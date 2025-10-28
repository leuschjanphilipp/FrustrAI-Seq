import sys
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.models import FrustrationCNN

config = {
    "experiment_name": "it3-5_loraAll_small",
    "parquet_path": "pLMtrainer/data/frustration/v4_frustration.parquet.gzip",
    "cath_sampling_n": 100,  # None for no sampling
    "batch_size": 64, # 24 for QK, 16 for all modules for FT; 64 maybe more for no FT
    "num_workers": 10,
    "input_dim": 1024,
    "hidden_dims": [64, 10],
    "output_dim": 4, # 3 classes + 1 for regression 
    "dropout": 0.15,
    "max_seq_length": 512,
    "precision": "full",
    "finetune": False,
    "pLM_model": "Rostlab/ProtT5",
    "prefix_prostT5": "<AA2fold>",
    "no_label_token": -100,
    "lora_r": 4,
    "lora_alpha": 1,
    "lora_modules": ["q", "k", "v", "o", "wi", "wo", "w1", "w2", "w3", "fc1", "fc2", "fc3"],
    "ce_weighting": None,
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
                                    max_seq_length=config["max_seq_length"], 
                                    num_workers=config["num_workers"], 
                                    persistent_workers=True,
                                    sample_size=None,
                                    cath_sampling_n=config["cath_sampling_n"])

model = FrustrationCNN(config=config)

early_stop = EarlyStopping(monitor="val_loss",
                           patience=5,
                           mode='min',
                           verbose=True)
checkpoint = ModelCheckpoint(monitor="val_loss",
                             dirpath=f"./{config['experiment_name']}",
                             filename=f"train_best",
                             save_top_k=1,
                             mode='min',
                             save_weights_only=True)
logger = WandbLogger(project="pLMtrainer_frustration",
                     name=config["experiment_name"],
                     save_dir=f"./{config['experiment_name']}",
                     log_model=False,
                     offline=True)

#logger = CSVLogger(f"./{exp_name}", name="train_logs")

trainer = Trainer(accelerator='auto', # gpu
                  devices=-1, # 4 for one node on haicore
                  strategy=DDPStrategy(find_unused_parameters=find_unused),
                  max_epochs=50,
                  logger=logger,
                  log_every_n_steps=10,
                  val_check_interval=0.2, #3500 batches in total 
                  callbacks=[early_stop, checkpoint],
                  precision=trainer_precision,
                  gradient_clip_val=1,
                  enable_progress_bar=False,
                  deterministic=False, # for reproducibility disable on cluster
                  accumulate_grad_batches=4, # if batch size gets too small
                  )
trainer.fit(model, datamodule=data_module)

test_trainer = Trainer(accelerator='auto', # gpu
                       devices=1, # 4 for one node on haicore
                       max_epochs=2,
                       logger=logger,
                       log_every_n_steps=5,
                       val_check_interval=50, #3500 batches in total 
                       callbacks=[early_stop, checkpoint],
                       precision=trainer_precision,
                       gradient_clip_val=1,
                       enable_progress_bar=False,
                       deterministic=False, # for reproducibility disable on cluster
                       #accumulate_grad_batches=4, # if batch size gets too small
                       )
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict()