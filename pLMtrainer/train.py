import os
import sys
import torch
import signal
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.models import FrustrationCNN
from pLMtrainer.utils import set_signal, HaicoreCheckpointOnExit

config = {
    "experiment_name": "it4_joint_ftAll_Ce_noDropout",
    "parquet_path": "pLMtrainer/data/frustration/v7_frustration_v2.parquet.gzip",
    "set_key": "set", # split_test (gonzalos prots in test) or set_old (split for previous dataset) or split0-3
    "cath_sampling_n": None, # 100,  # None for no sampling
    "batch_size": 16, # 24 for QK, 16 for all modules for FT; 64 maybe more for no FT
    "num_workers": 10,
    "input_dim": 1024,
    "hidden_dims": [64, 10],
    "dropout": 0,
    "regression": True,
    "classification": True,
    "max_seq_length": 512,
    "precision": "full",
    "finetune": True,
    "pLM_model": "pLMtrainer/data/protT5",
    "prefix_prostT5": "<AA2fold>",
    "no_label_token": -100,
    "lora_r": 4,
    "lora_alpha": 1,
    "lora_modules": ["q", "k", "v", "o", "wi", "wo", "w1", "w2", "w3", "fc1", "fc2", "fc3"],
    "ce_weighting": [10.0, 2.0, 2.5],
    "mc_dropout_n": 0,  #0 for normal test; number of MC dropout samples during test time
    "notes": "",
}

if config["precision"] == "full":
    torch.set_float32_matmul_precision("high")
    trainer_precision = "32"
elif config["precision"] == "half":
    torch.set_float32_matmul_precision("medium")
    trainer_precision = "16-mixed"

if config["finetune"]:
    find_unused = False
else:
    find_unused = True

data_module = FrustrationDataModule(df=None,
                                    parquet_path=config["parquet_path"], 
                                    batch_size=config["batch_size"],
                                    set_key=config["set_key"],
                                    max_seq_length=config["max_seq_length"], 
                                    num_workers=config["num_workers"], 
                                    persistent_workers=True,
                                    sample_size=None,
                                    cath_sampling_n=config["cath_sampling_n"])

early_stop = EarlyStopping(monitor="val_loss",
                           patience=5,
                           mode='min',
                           verbose=True)
checkpoint = ModelCheckpoint(monitor="val_loss",
                             dirpath=f"./{config['experiment_name']}",
                             filename=f"best_val_model",
                             save_top_k=1,
                             mode='min',
                             save_weights_only=True)
checkpoint_epoch  = ModelCheckpoint(dirpath=f"./{config['experiment_name']}",
                                    filename='model_{epoch:03d}',
                                    save_weights_only=False,
                                    every_n_epochs=1,)
logger = WandbLogger(project="pLMtrainer_frustration",
                     name=config["experiment_name"],
                     save_dir=f"./{config['experiment_name']}",
                     log_model=False,
                     offline=True)

plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1, auto_requeue=False)]

trainer = Trainer(default_root_dir=f"./{config['experiment_name']}",
                  accelerator='auto',
                  devices=-1,
                  #strategy=DDPStrategy(find_unused_parameters=find_unused),
                  max_epochs=50,
                  logger=logger,
                  log_every_n_steps=10,
                  val_check_interval=0.2,
                  callbacks=[early_stop, checkpoint, checkpoint_epoch, HaicoreCheckpointOnExit()],
                  plugins=plugins,
                  precision=trainer_precision,
                  gradient_clip_val=1,
                  enable_progress_bar=False,
                  deterministic=False,
                  accumulate_grad_batches=4, # if batch size gets too small
                  )
set_signal(trainer)

if os.path.exists(f"./{config['experiment_name']}/hpc_ckpt.ckpt"):
    print("Resuming from HPC checkpoint.")
    model = FrustrationCNN.load_from_checkpoint(checkpoint_path=f"./{config['experiment_name']}/hpc_ckpt.ckpt",
                                               config=config)
    trainer.fit(model, ckpt_path=f"./{config['experiment_name']}/hpc_ckpt.ckpt", datamodule=data_module)
else:
    print("Initializing new model.")
    model = FrustrationCNN(config=config)
    trainer.fit(model, datamodule=data_module)

if trainer.should_stop:
    sys.exit(0)