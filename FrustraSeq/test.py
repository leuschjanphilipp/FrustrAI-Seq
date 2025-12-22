import os
import sys
import yaml
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

sys.path.append('..')
sys.path.append('FrustraSeq')
from FrustraSeq.models.FrustraSeq import FrustraSeq
from FrustraSeq.dataloader import FrustrationDataModule
from FrustraSeq.utils import run_eval_metrics

with open(f"./it5_ABL_protT5_CW/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

torch.set_float32_matmul_precision("high")
trainer_precision = "32"

if config["finetune"]:
    find_unused = False
else:
    find_unused = True

data_module = FrustrationDataModule(df=None,
                                    parquet_path=config["parquet_path"], 
                                    batch_size=config["batch_size"],
                                    set_key=config["set_key"],
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

print(f"RANK {os.environ.get('RANK', -1)}: Initializing new model.")
model = FrustraSeq.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/best_val_model.ckpt",
                                        config=config)

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
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict(set="val")

metrics = run_eval_metrics(np.load(f"./{config['experiment_name']}/val_preds.npz"), return_cls_report_dict=False)
print(metrics["cls_report"])
print(metrics["pearson_r"])
print(metrics["mean_absolute_error"])