import sys
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule
from pLMtrainer.models import FrustrationFNN

parquet_path = "pLMtrainer/data/frustration/v3_frustration.parquet.gzip"
max_seq_length = 700
batch_size = 64
num_workers = 16

data_module = FrustrationDataModule(parquet_path=parquet_path, 
                                    batch_size=batch_size, 
                                    max_seq_length=max_seq_length, 
                                    num_workers=num_workers, 
                                    persistent_workers=True)

model = FrustrationFNN(input_dim=1024, 
                       hidden_dim=32, 
                       output_dim=22, 
                       dropout=0.15, 
                       max_seq_length=max_seq_length,
                       pLM_model="../data/ProstT5", 
                       pLM_precision="half", 
                       prefix_prostT5="<AA2fold>",
                       no_label_token=0)

torch.set_float32_matmul_precision('medium')

early_stop = EarlyStopping(monitor="val_loss",
                           patience=5,
                           mode='min',
                           verbose=True)
checkpoint = ModelCheckpoint(monitor="val_loss",
                             dirpath="./checkpoints",
                             filename=f"train_reg_cls",
                             save_top_k=1,
                             mode='min',
                             save_weights_only=True)
logger = CSVLogger("./checkpoints", name="train_reg_cls_logs")

trainer = Trainer(accelerator='auto', # gpu
                  devices=-1, # 4 for one node on haicore
                  strategy='ddp',
                  max_epochs=100,
                  logger=logger,
                  log_every_n_steps=50,
                  callbacks=[early_stop, checkpoint],
                  precision="16-mixed",
                  gradient_clip_val=1,
                  enable_progress_bar=True,
                  deterministic=False, # for reproducibility disable on cluster 
                  #num_sanity_val_steps=0,
                  #accumulate_grad_batches=2, # if batch size gets too small --> test on H100/A100
                  )

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)
model.save_preds_targets(path="./checkpoints/preds_targets_reg_cls.npz")