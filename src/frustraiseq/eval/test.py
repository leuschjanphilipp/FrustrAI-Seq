import yaml
import argparse
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

from frustraiseq.data.dataloader import FunstrationDataModule
from frustraiseq.model.frustraiseq import FrustrAISeq
from frustraiseq.utils.utils import run_eval_metrics

parser = argparse.ArgumentParser(description="Test FrustrAI-Seq model")
parser.add_argument("--config", type=str, default="./config.yaml")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Model checkpoint. Defaults to checkpoint_path in the config, else ./<experiment_name>/best_val_model.ckpt")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

if args.checkpoint is not None:
    config["checkpoint_path"] = args.checkpoint
elif config.get("checkpoint_path") is None:
    config["checkpoint_path"] = f"./{config['experiment_name']}/best_val_model.ckpt"

#torch.set_float32_matmul_precision("high")
trainer_precision = "bf16-mixed" #"32"

data_module = FunstrationDataModule(config=config,
                                    fit_dataset=config["fit_dataset"], 
                                    batch_size=config["batch_size"],
                                    split_key=config["split_key"],
                                    max_seq_length=config["max_seq_length"], 
                                    num_workers=config["num_workers"], # 0 
                                    persistent_workers=True, # Flase
                                    pin_memory=True, # Flase
                                    prefetch_factor=2, #!
                                    sample_size=None,
                                    cath_sampling_n=config["cath_sampling_n"])
data_module.setup()

logger = CSVLogger(save_dir=f"./{config['experiment_name']}",
                    name="logs",
                    flush_logs_every_n_steps=5)


test_trainer = Trainer(accelerator="gpu", # gpu
                        devices=1, # only use one gpu for inference
                        max_epochs=2, 
                        logger=logger,
                        log_every_n_steps=10,
                        val_check_interval=0.2, 
                        precision=trainer_precision, #!, config["inference_precision"],
                        gradient_clip_val=1,
                        enable_progress_bar=config["verbose"],
                        )

model = FrustrAISeq.load_from_checkpoint(checkpoint_path=config["checkpoint_path"],
                                         config=config)
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict(set="test")

metrics = run_eval_metrics(np.load(f"./{config['experiment_name']}/test_preds.npz"), return_cls_report_dict=False)
print(f"Metrics for {config['experiment_name']} - Test Set:")
print(metrics["cls_report"])
print(metrics["pearson_r"])
print("MAE:", metrics["mean_absolute_error"])