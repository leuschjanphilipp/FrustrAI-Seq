import os
import sys
import torch
import argparse
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from frustraiseq.config.default_config import DEFAULT_CONFIG
from frustraiseq.data.dataloader import FunstrationDataModule
from frustraiseq.model.frustraiseq import FrustrAISeq
from frustraiseq.utils.utils import run_eval_metrics


parser = argparse.ArgumentParser(description="Train FrustrAI-Seq model")
parser.add_argument("--experiment_name", type=str, default="train",)
parser.add_argument("--fit_dataset", type=str, default="leuschj/Funstration",)
parser.add_argument("--batch_size", type=int, default=32,)
parser.add_argument("--plm_model", type=str, default="./data/protT5",)
parser.add_argument("--split_key", type=str, default="split_0",)
parser.add_argument("--num_workers", type=int, default=10,)
parser.add_argument("--cath_sampling_n", type=int, default=None)
args = parser.parse_args()

config = DEFAULT_CONFIG.copy()

config["experiment_name"] = args.experiment_name
config["fit_dataset"] = args.fit_dataset
config["batch_size"] = args.batch_size
config["pLM_model"] = args.plm_model
config["split_key"] = args.split_key
config["num_workers"] = args.num_workers
config["cath_sampling_n"] = args.cath_sampling_n

#torch.set_float32_matmul_precision("high")
trainer_precision = "bf16-mixed" #"32"
find_unused = False

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

early_stop = EarlyStopping(monitor="val_loss",
                            patience=5,
                            min_delta=0.001,
                            mode='min',
                            verbose=True)
checkpoint = ModelCheckpoint(monitor="val_loss",
                                dirpath=f"./{config['experiment_name']}",
                                filename=f"best_val_model",
                                save_top_k=1,
                                mode='min',
                                save_weights_only=False)
logger = WandbLogger(project="FrustrAI-Seq",
                        name=config["experiment_name"],
                        save_dir=f"./{config['experiment_name']}",
                        log_model=False,
                        offline=False, #lets see
                        )
lr_logger = LearningRateMonitor(logging_interval='step')

trainer = Trainer(default_root_dir=f"./{config['experiment_name']}",
                accelerator="gpu",
                devices=2,
                strategy=DDPStrategy(find_unused_parameters=find_unused),
                max_epochs=20,
                logger=logger,
                log_every_n_steps=10,
                val_check_interval=0.2,
                callbacks=[early_stop, checkpoint, lr_logger],
                precision=trainer_precision,
                gradient_clip_val=1,
                enable_progress_bar=False,
                deterministic=False,
                accumulate_grad_batches=8, # used 8 for FT, maybe less for for FT in future.
                )

ckpt_path = None
ckpt_file = f"{config['experiment_name']}/best_val_model.ckpt"

if os.path.exists(ckpt_file):
    ckpt_path = ckpt_file
    print(f"RANK {os.environ.get('RANK', -1)}: Resuming training from checkpoint: {ckpt_file}")
else:
    print(f"RANK {os.environ.get('RANK', -1)}: Starting new training run")

model = FrustrAISeq(config=config)

trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path=ckpt_path
)

test_trainer = Trainer(accelerator="gpu", # gpu
                        devices=1, # only use one gpu for inference
                        max_epochs=2, 
                        logger=logger,
                        log_every_n_steps=10,
                        val_check_interval=0.2, 
                        precision=trainer_precision, #!, config["inference_precision"],
                        gradient_clip_val=1,
                        enable_progress_bar=False,
                        )
#tune on val set
data_module.test_dataloader = data_module.val_dataloader
model = FrustrAISeq.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/best_val_model.ckpt",
                                        config=config)
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict(set="val")

metrics = run_eval_metrics(np.load(f"./{config['experiment_name']}/val_preds.npz"), return_cls_report_dict=False)
print(f"Metrics for {config['experiment_name']} - Validation Set:")
print(metrics["cls_report"])
print(metrics["pearson_r"])
print(metrics["mean_absolute_error"])