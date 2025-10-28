import sys
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.models import FrustrationCNN

with open(f"./it3-5_loraAll_small/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

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
logger = CSVLogger(f"./{config['experiment_name']}", name="test_logs")

model = FrustrationCNN.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/train_best.ckpt",
                                            config=config)
test_trainer = Trainer(accelerator='auto', # gpu
                       devices=1, # 4 for one node on haicore
                       max_epochs=2,
                       logger=logger,
                       log_every_n_steps=5,
                       val_check_interval=50, 
                       precision=trainer_precision,
                       gradient_clip_val=1,
                       enable_progress_bar=False,
                       deterministic=False,
                       )
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict()