import sys
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.models import FrustrationCNN

with open(f"./it4_joint_ftAll_Ce_noDropout/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

#config["mc_dropout_n"] = 100  #0 for normal test; manually adding since didnt exist when model was trained.
#config["cath_sampling_n"] = 100  # 100,  # None for no sampling; subsample for testing MCD

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
data_module.setup()
logger = CSVLogger(f"./{config['experiment_name']}", name="test_logs")

model = FrustrationCNN.load_from_checkpoint(checkpoint_path=f"{config['experiment_name']}/best_val_model.ckpt",
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
#hacky solution to use val dataloader for evaluation
#data_module.test_dataloader = data_module.val_dataloader
test_trainer.test(model, datamodule=data_module)
model.save_preds_dict(set="test")