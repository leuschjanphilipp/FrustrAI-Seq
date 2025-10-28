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
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule

torch.set_float32_matmul_precision('medium')

class FrustrationFNN(pl.LightningModule):
    def __init__(self, 
                 input_dim=1024, 
                 hidden_dim=32, 
                 output_dim=21, # 20 classes + 1 for regression 
                 dropout=0.15,
                 max_seq_length=512,
                 precision="full",
                 pLM_model="Rostlab/ProstT5", 
                 prefix_prostT5="<AA2fold>",
                 no_label_token=-100,):
        super(FrustrationFNN, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(pLM_model).to(self.device)
        self.prefix_prostT5 = prefix_prostT5
        self.precision = precision

        self.FNN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        if self.precision == "half":
            self.encoder.half()
            #self.FNN.half()
            print("Using half precision")
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
        
        embeddings = embeddings.float()
        res = self.FNN(embeddings)
        #end_time = time.time()
        #print(f"Forward pass time: {end_time - start_time} seconds")
        return res

    def training_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Training batch with no valid residues - skipping")
            return None  # Skip this batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        mse_loss = self.mse_loss_fn(preds[..., -1][res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
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
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
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
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
        loss = mse_loss + ce_loss
        self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        regr_preds = preds[..., -1]
        #class_preds = torch.argmax(preds[..., :-1], dim=-1)
        class_preds = preds[..., :-1]

        self.test_dict["regr_preds"].append(regr_preds.cpu().numpy())
        self.test_dict["cls_preds"].append(class_preds.cpu().numpy())
        self.test_dict["regr_targets"].append(frst_vals.cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.cpu().numpy())

        self.test_dict["masked_regr_preds"].append(regr_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_preds"].append(class_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].flatten().cpu().numpy())

        return loss

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.Adam(self.FNN.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_start(self):
        self.test_dict = {"regr_preds": [], "cls_preds": [], "regr_targets": [], "cls_targets": [],
                          "masked_regr_preds": [], "masked_cls_preds": [], "masked_regr_targets": [], "masked_cls_targets": []}


    def on_test_epoch_end(self):
        self.test_dict["regr_preds"] = np.concatenate(self.test_dict["regr_preds"])
        self.test_dict["cls_preds"] = np.concatenate(self.test_dict["cls_preds"])
        self.test_dict["regr_targets"] = np.concatenate(self.test_dict["regr_targets"])
        self.test_dict["cls_targets"] = np.concatenate(self.test_dict["cls_targets"])
        self.test_dict["masked_regr_preds"] = np.concatenate(self.test_dict["masked_regr_preds"])
        self.test_dict["masked_cls_preds"] = np.concatenate(self.test_dict["masked_cls_preds"])
        self.test_dict["masked_regr_targets"] = np.concatenate(self.test_dict["masked_regr_targets"])
        self.test_dict["masked_cls_targets"] = np.concatenate(self.test_dict["masked_cls_targets"])


    def save_preds_targets(self, path="./preds_targets.npz"):
        np.savez_compressed(path, **self.test_dict)

    @staticmethod
    def suggest_params():
        #TODO model selection
        pass

class FrustrationCNN(pl.LightningModule):
    def __init__(self, 
                 input_dim=1024, 
                 hidden_dims=[64, 10], 
                 output_dim=3, # 3 classes + 1 for regression 
                 dropout=0.15,
                 max_seq_length=512,
                 precision="full", 
                 pLM_model="Rostlab/ProtT5",
                 prefix_prostT5="<AA2fold>",
                 no_label_token=-100,):
        super(FrustrationCNN, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(pLM_model).to(self.device)
        self.prefix_prostT5 = prefix_prostT5
        self.max_seq_length = max_seq_length + 1 # +1 for prefix_prostT5 token; later trunc to max_seq_length
        self.precision = precision

        #https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
        self.CNN = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dims[0], kernel_size=(7, 1), padding=(3, 0)),  # 7x64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=(7, 1), padding=(3, 0))
        )

        self.cls_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim-1),
        ) 

        self.reg_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

        if self.precision == "half":
            self.encoder.half()
            #self.CNN.half()
            print("Using half precision")
        self.encoder.eval()  # Freeze the encoder

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
        
        embeddings = embeddings.float()
        embeddings = embeddings.permute(0, 2, 1).unsqueeze(-1)  # (batch_size, input_dim, seq_length, 1)
        res = self.CNN(embeddings).squeeze(-1).permute(0, 2, 1)  # (batch_size, seq_length, output_dim)
        cls_res = self.cls_head(res)
        reg_res = self.reg_head(res)
        res = torch.cat((cls_res, reg_res), dim=-1)
        # TODO return as tuple of (class_preds, regr_preds) and change in other methods for better readability
        #end_time = time.time()
        #print(f"Forward pass time: {end_time - start_time} seconds")
        return res

    def training_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Training batch with no valid residues - skipping")
            return None  # Skip this batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        mse_loss = self.mse_loss_fn(preds[..., -1][res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
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
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
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
        ce_loss = self.ce_loss_fn(preds[..., :-1].flatten(0,1), frst_classes.flatten()) # shape (batch_size, 20)
        loss = mse_loss + ce_loss
        self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        regr_preds = preds[..., -1]
        #class_preds = torch.argmax(preds[..., :-1], dim=-1
        class_preds = preds[..., :-1]

        self.test_dict["regr_preds"].append(regr_preds.cpu().numpy())
        self.test_dict["cls_preds"].append(class_preds.cpu().numpy())
        self.test_dict["regr_targets"].append(frst_vals.cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.cpu().numpy())

        self.test_dict["masked_regr_preds"].append(regr_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_preds"].append(class_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].flatten().cpu().numpy())

        return loss
    
    def predict_step(self, batch, batch_idx):
        full_seq, _, _, _ = batch
        preds = self.forward(full_seq)
        self.preds_list.append(preds.cpu().numpy())

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.Adam(self.CNN.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_start(self):
        self.test_dict = {"regr_preds": [], "cls_preds": [], "regr_targets": [], "cls_targets": [],
                          "masked_regr_preds": [], "masked_cls_preds": [], "masked_regr_targets": [], "masked_cls_targets": []}

    def on_test_epoch_end(self):
        self.test_dict["regr_preds"] = np.concatenate(self.test_dict["regr_preds"])
        self.test_dict["cls_preds"] = np.concatenate(self.test_dict["cls_preds"])
        self.test_dict["regr_targets"] = np.concatenate(self.test_dict["regr_targets"])
        self.test_dict["cls_targets"] = np.concatenate(self.test_dict["cls_targets"])
        self.test_dict["masked_regr_preds"] = np.concatenate(self.test_dict["masked_regr_preds"])
        self.test_dict["masked_cls_preds"] = np.concatenate(self.test_dict["masked_cls_preds"])
        self.test_dict["masked_regr_targets"] = np.concatenate(self.test_dict["masked_regr_targets"])
        self.test_dict["masked_cls_targets"] = np.concatenate(self.test_dict["masked_cls_targets"])

    def on_predict_start(self):
        self.preds_list = []
    
    def on_predict_end(self):
        self.preds_list = np.concatenate(self.preds_list)

    def save_preds_targets(self, path="./preds_targets.npz"):
        np.savez_compressed(path, **self.test_dict)

    @staticmethod
    def suggest_params():
        #TODO model selection
        pass

class FrustrationBaseline(pl.LightningModule):
    def __init__(self, 
                 input_dim=1024, 
                 output_dim=21, # 20 classes + 1 for regression 
                 max_seq_length=512,
                 precision="full",
                 pLM_model="Rostlab/ProstT5", 
                 prefix_prostT5="<AA2fold>",
                 no_label_token=-100,):
        super(FrustrationBaseline, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(pLM_model).to(self.device)
        self.prefix_prostT5 = prefix_prostT5
        self.precision = precision

        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

        if self.precision == "half":
            self.encoder.half()
            #self.FNN.half()
            print("Using half precision")
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
        
        embeddings = embeddings.float()
        res = self.linear(embeddings)
        #end_time = time.time()
        #print(f"Forward pass time: {end_time - start_time} seconds")
        return res

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

        regr_preds = preds[..., -1]
        #class_preds = torch.argmax(preds[..., :-1], dim=-1)
        class_preds = preds[..., :-1]

        self.test_dict["regr_preds"].append(regr_preds.cpu().numpy())
        self.test_dict["cls_preds"].append(class_preds.cpu().numpy())
        self.test_dict["regr_targets"].append(frst_vals.cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.cpu().numpy())

        self.test_dict["masked_regr_preds"].append(regr_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_preds"].append(class_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].flatten().cpu().numpy())

        return loss

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_start(self):
        self.test_dict = {"regr_preds": [], "cls_preds": [], "regr_targets": [], "cls_targets": [],
                          "masked_regr_preds": [], "masked_cls_preds": [], "masked_regr_targets": [], "masked_cls_targets": []}

    def on_test_epoch_end(self):
        self.test_dict["regr_preds"] = np.concatenate(self.test_dict["regr_preds"])
        self.test_dict["cls_preds"] = np.concatenate(self.test_dict["cls_preds"])
        self.test_dict["regr_targets"] = np.concatenate(self.test_dict["regr_targets"])
        self.test_dict["cls_targets"] = np.concatenate(self.test_dict["cls_targets"])
        self.test_dict["masked_regr_preds"] = np.concatenate(self.test_dict["masked_regr_preds"])
        self.test_dict["masked_cls_preds"] = np.concatenate(self.test_dict["masked_cls_preds"])
        self.test_dict["masked_regr_targets"] = np.concatenate(self.test_dict["masked_regr_targets"])
        self.test_dict["masked_cls_targets"] = np.concatenate(self.test_dict["masked_cls_targets"])


    def save_preds_targets(self, path="./preds_targets.npz"):
        np.savez_compressed(path, **self.test_dict)


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
                       output_dim=21, 
                       max_seq_length=100,
                       pLM_model="pLMtrainer/data/ProstT5", 
                       precision="half", 
                       prefix_prostT5="<AA2fold>",
                       no_label_token=-100)
    
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

if __name__ == "__main__":
    
    parquet_path = "pLMtrainer/data/frustration/v3_frustration.parquet.gzip"
    data_module = FrustrationDataModule(parquet_path=parquet_path, batch_size=10, max_seq_length=100, num_workers=1, persistent_workers=True, sample_size=1000,)
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

    model = FrustrationCNN(input_dim=1024, 
                       hidden_dim=32, 
                       output_dim=21, 
                       dropout=0.15, 
                       max_seq_length=100,
                       precision="half",
                       pLM_model="pLMtrainer/data/ProstT5",  
                       prefix_prostT5="<AA2fold>",
                       no_label_token=-100)

    trainer = Trainer(accelerator='auto', # gpu
                  devices=-1,
                  max_epochs=5,
                  logger=logger,
                  log_every_n_steps=10, # 50 for haicore default
                  callbacks=[early_stop, checkpoint],
                  precision="16-mixed",
                  gradient_clip_val=1,
                  enable_progress_bar=True,
                  deterministic=False,
                  )

    trainer.fit(model, datamodule=data_module)




