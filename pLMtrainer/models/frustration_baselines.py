import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from transformers import T5Tokenizer, T5EncoderModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

sys.path.append('..') 
from pLMtrainer.dataloader import FrustrationDataset, FrustrationDataModule

torch.set_float32_matmul_precision('medium')

class FrustrationBaseline(pl.LightningModule):
    def __init__(self, 
                 input_dim=1024,
                 output_dim=22, # 20 classes + 1 for no info + 1 for regression 
                 max_seq_length=700,
                 precision="full",
                 pLM_model="Rostlab/ProstT5", 
                 prefix_prostT5="<AA2fold>",
                 no_label_token=0,):
        super(FrustrationBaseline, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(pLM_model).to(self.device)
        self.prefix_prostT5 = prefix_prostT5
        self.precision = precision

        self.weights = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        if self.precision == "half":
            self.encoder.half()
            self.weights.half()
            print("Using half precision for the pLM encoder")
        self.encoder.eval()  # Freeze the encoder

        self.max_seq_length = max_seq_length + 1 # +1 for prefix_prostT5 token; later trunc to max_seq_length

        self.mse_loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=no_label_token) # TODO look at weight param for class imbalance

    
    def forward(self, full_seq):
        #start_time = time.time()
        full_seq = [self.prefix_prostT5 + " " + " ".join(seq) for seq in full_seq]  # Add spaces between amino acids and prefix
        ids = self.tokenizer.batch_encode_plus(full_seq, 
                                               add_special_tokens=True, 
                                               max_length=self.max_seq_length,
                                               padding="max_length",
                                               truncation="longest_first",
                                               return_tensors='pt'
                                               ).to(self.device)
        
        with torch.no_grad():
            embedding_rpr = self.encoder(
                ids.input_ids, 
                attention_mask=ids.attention_mask
            )
        embeddings = embedding_rpr.last_hidden_state[:, 1:] # remove the aa token bos and bring to shape
        
        #embeddings = embeddings.float()
        res = self.FNN(embeddings)
        #end_time = time.time()
        #print(f"Forward pass time: {end_time - start_time} seconds")
        return res.float()

    def training_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Training batch with no valid residues - skipping")
            return None  # Skip this batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        mse_loss = self.mse_loss_fn(preds[..., -1][res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 21)
        loss = mse_loss + ce_loss
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Validation batch with no valid residues - skipping") 
            return None  # Skip this batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        mse_loss = self.mse_loss_fn(preds[..., -1][res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 21)
        loss = mse_loss + ce_loss
        self.log('val_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Test batch with no valid residues - skipping") 
            return None  # Skip this batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        mse_loss = self.mse_loss_fn(preds[..., -1][res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 21)
        loss = mse_loss + ce_loss
        self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if not self.regression:
            preds = torch.cat((torch.argmax(preds[:, :-1], dim=-1), preds[:, -1].unsqueeze(-1)), dim=-1)  # keep regression output as is

        self.all_preds.append(preds.cpu().numpy())
        self.all_targets.append(frst_vals.cpu().numpy())
        self.masked_all_preds.append(preds[res_mask].flatten().cpu().numpy())
        self.masked_all_targets.append(frst_vals[res_mask].flatten().cpu().numpy())

        return loss

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.Adam(self.weights.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_start(self):
        self.all_preds = []
        self.all_targets = []
        self.masked_all_preds = []
        self.masked_all_targets = []

    def on_test_epoch_end(self):
        self.all_preds = np.concatenate(self.all_preds)
        self.all_targets = np.concatenate(self.all_targets)
        self.masked_all_preds = np.concatenate(self.masked_all_preds)
        self.masked_all_targets = np.concatenate(self.masked_all_targets)

        #print("Classification Report:\n", classification_report(self.masked_all_targets, self.masked_all_preds, labels=list(range(1, 21)), zero_division=0))

    def save_preds_targets(self, path="./preds_targets.npz"):
        np.savez_compressed(path, 
                            all_preds=self.all_preds, 
                            all_targets=self.all_targets, 
                            masked_preds=self.masked_all_preds, 
                            masked_targets=self.masked_all_targets)


if __name__ == "__main__":
    
    parquet_path = "pLMtrainer/data/frustration/v3_frustration.parquet.gzip"
    data_module = FrustrationDataModule(parquet_path=parquet_path, batch_size=10, max_seq_length=100, num_workers=1, persistent_workers=True)
    early_stop = EarlyStopping(monitor="val_loss",
                           patience=5,
                           mode='min',
                           verbose=True)
    checkpoint = ModelCheckpoint(monitor="val_loss",
                                dirpath="./checkpoints",
                                filename=f"debug",
                                save_top_k=1,
                                mode='min',
                                save_weights_only=True)
    logger = CSVLogger("./checkpoints", name="debug_logs")

    model = FrustrationBaseline(input_dim=1024,
                       output_dim=22, 
                       max_seq_length=100,
                       pLM_model="pLMtrainer/data/ProstT5", 
                       pLM_precision="half", 
                       prefix_prostT5="<AA2fold>",
                       no_label_token=0)
    
    trainer = Trainer(accelerator='auto', # gpu
                  devices=-1,
                  max_epochs=5,
                  logger=logger,
                  log_every_n_steps=10, # 50 for haicore default
                  callbacks=[early_stop, checkpoint],
                  precision="16-mixed",
                  gradient_clip_val=1,
                  enable_progress_bar=True,
                  deterministic=False,)

    trainer.fit(model, datamodule=data_module)




