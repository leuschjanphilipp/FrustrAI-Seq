import os
import sys
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.models.frustraSeq import FrustraSeq
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.utils import run_eval_metrics

config = {
    "experiment_name": "it5_ABL_protT5_LORA",
    "parquet_path": "pLMtrainer/data/frustration/v8_frustration_v2.parquet.gzip",
    "set_key": "split_test", # split_test (gonzalos prots in test) or set_old (split for previous dataset) or split0-3
    "cath_sampling_n": None, # 100,  # None for no sampling
    "batch_size": 32, #16 for all modules for FT; 64 maybe more for no FT
    "num_workers": 10,
    "max_seq_length": 512,
    "precision": "full",
    "pLM_model": "pLMtrainer/data/protT5",
    "prefix_prostT5": "<AA2fold>",
    "pLM_dim": 1024, #1280
    "no_label_token": -100,
    "finetune": True,
    "lora_r": 4,
    "lora_alpha": 1,
    "lora_modules": ["q", "k", "v", "o"], #"wi", "wo", "w1", "w2", "w3", "fc1", "fc2", "fc3"], # ["query", "key", "value", "fc1", "fc2"] for esm
    "ce_weighting": None, #[2.65750085, 0.68876299, 0.8533673], #[10.0, 2.0, 2.5], [(1/0.13)/(1/0.13), (1/0.48)/(1/0.13), (1/0.39)/(1/0.13)]
    "notes": "",
}
architecture = {}
architecture["lr"] = 1e-4
architecture["dropout"] = 0.1
architecture["kernel_1"] = 7
architecture["padding_1"] = architecture["kernel_1"] // 2  # to keep same length
architecture["kernel_2"] = 7
architecture["padding_2"] = architecture["kernel_2"] // 2  # to keep same length
architecture["hidden_dim_0"] = 64
architecture["hidden_dim_1"] = 10
config["architecture"] = architecture

#torch.set_float32_matmul_precision("high")
trainer_precision = "bf16-mixed" #"32"

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

early_stop = EarlyStopping(monitor="val_loss",
                            patience=5,
                            min_delta=0.0001,
                            mode='min',
                            verbose=True)
checkpoint = ModelCheckpoint(monitor="val_loss",
                                dirpath=f"./{config['experiment_name']}",
                                filename=f"best_val_model",
                                save_top_k=1,
                                mode='min',
                                save_weights_only=False)
logger = WandbLogger(project="pLMtrainer_FrustraSeq_",
                        name=config["experiment_name"],
                        save_dir=f"./{config['experiment_name']}",
                        log_model=False,
                        offline=True,
                        )
lr_logger = LearningRateMonitor(logging_interval='step')

trainer = Trainer(default_root_dir=f"./{config['experiment_name']}",
                accelerator="gpu",
                devices=4,
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
                accumulate_grad_batches=8,
                )

ckpt_path = None
ckpt_file = f"{config['experiment_name']}/best_val_model.ckpt"

if os.path.exists(ckpt_file):
    ckpt_path = ckpt_file
    print(f"RANK {os.environ.get('RANK', -1)}: Resuming training from checkpoint: {ckpt_file}")
else:
    print(f"RANK {os.environ.get('RANK', -1)}: Starting new training run")

model = FrustraSeq(config=config)

trainer.fit(
    model,
    datamodule=data_module,
    ckpt_path=ckpt_path
)
"""
if os.path.exists(f"{config['experiment_name']}/best_val_model.ckpt"):
    print(f"RANK {os.environ.get('RANK', -1)}: Loading model from checkpoint.")
    model = FrustraSeq.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/best_val_model.ckpt",
                                            config=config)
else:
    print(f"RANK {os.environ.get('RANK', -1)}: Initializing new model.")
    model = FrustraSeq(config=config)
trainer.fit(model, datamodule=data_module)
"""

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
model = FrustraSeq.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/best_val_model.ckpt",
                                        config=config)
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict(set="val")

metrics = run_eval_metrics(np.load(f"./{config['experiment_name']}/val_preds.npz"), return_cls_report_dict=False)
print(metrics["cls_report"])
print(metrics["pearson_r"])
print(metrics["mean_absolute_error"])
