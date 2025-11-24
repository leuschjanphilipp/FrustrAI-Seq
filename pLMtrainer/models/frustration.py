import sys
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from transformers import T5Tokenizer, T5EncoderModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from peft import LoraConfig, get_peft_model

sys.path.append('..')
sys.path.append('pLMtrainer')
from pLMtrainer.dataloader import FrustrationDataModule

class FrustrationCNN(pl.LightningModule):
    def __init__(self, config):
        super(FrustrationCNN, self).__init__()

        self.config = config
        self.experiment_name = config["experiment_name"]
        self.plm_model = config["pLM_model"]
        self.prefix_prostT5 = config["prefix_prostT5"]
        self.max_seq_length = config["max_seq_length"]
        if "prostt5" in self.plm_model.lower():
            self.max_seq_length += 1  # for prostT5, add 1 for bos token <AA2fold>
        self.precision = config["precision"]
        self.finetune = config["finetune"]
        self.tokenizer = T5Tokenizer.from_pretrained(self.plm_model, do_lower_case=False, max_length=self.max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(self.plm_model).to(self.device)
        self.regression = config["regression"]
        self.classification = config["classification"]
        self.mc_dropout_n = config["mc_dropout_n"]

        assert self.regression or self.classification, "At least one of regression or classification must be True. Both can be true for joint training."
        assert (self.finetune and self.precision == "full") or not self.finetune, "LoRA fine-tuning only supported with full precision"

        if self.precision == "half":
            self.encoder.half()
            #self.CNN.half()
            print("Using half precision")

        if self.finetune:
            print(f"Using LoRA fine-tuning for {self.config['lora_modules']} layers")
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                inference_mode=False,
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                bias="all",
                target_modules=self.config['lora_modules'],
                #lora_dropout=0.1,
            )
            self.encoder.train()
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.print_trainable_parameters()
            # https://github.com/RSchmirler/ProtT5-EvoTuning/blob/main/notebook/PT5_EvoTuning.ipynb 
            #peft_config = LoraConfig(r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"], task_type = "SEQ_2_SEQ_LM",)
            # https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
        else:
            self.encoder.eval()  # Freeze the encoder   

        self.CNN = nn.Sequential(
            nn.Conv2d(config["input_dim"], config["hidden_dims"][0], kernel_size=(7, 1), padding=(3, 0)),  # 7x64
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Conv2d(config["hidden_dims"][0], config["hidden_dims"][1], kernel_size=(7, 1), padding=(3, 0))
        )

        if self.regression:
            print(f"Adding regression head")
            self.reg_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dims"][1], 1),
            )
            self.mse_loss_fn = nn.MSELoss()
            
        if self.classification:
            print(f"Adding classification head")
            self.cls_head = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["hidden_dims"][1], 3),  # 3 classes
            ) 
            if config["ce_weighting"] is not None:
                self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config["no_label_token"], weight=torch.Tensor(config["ce_weighting"])) 
            else:
                self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config["no_label_token"]) 

    def forward(self, full_seq):
        emb = self._plm_forward(full_seq)  # (batch_size, seq_length, input_dim)
        out = self._cnn_forward(emb)  # dict with regression and/or classification outputs
        return out
        
    def _plm_forward(self, full_seq):
        #start_time = time.time()
        if "prostt5" in self.plm_model.lower():
            full_seq = [self.prefix_prostT5 + " " + " ".join(seq) for seq in full_seq]  # Add spaces between amino acids and prefix
        else:
            full_seq = [" ".join(seq) for seq in full_seq]

        ids = self.tokenizer.batch_encode_plus(full_seq, 
                                               add_special_tokens=True, 
                                               max_length=self.max_seq_length,
                                               padding="max_length",
                                               truncation="longest_first",
                                               return_tensors='pt'
                                               ).to(self.device)

        if self.finetune:
            embedding_rpr = self.encoder(
                input_ids=ids.input_ids, 
                attention_mask=ids.attention_mask
            )
        else:
            with torch.no_grad():
                embedding_rpr = self.encoder(
                    input_ids=ids.input_ids, 
                    attention_mask=ids.attention_mask
                )
                
        if "prostt5" in self.plm_model.lower():
            embeddings = embedding_rpr.last_hidden_state[:, 1:].float() # remove the aa token bos and eos and bring to shape
        else:
            embeddings = embedding_rpr.last_hidden_state.float() # remove the aa token bos and bring to shape

        embeddings = embeddings.permute(0, 2, 1).unsqueeze(-1)  # (batch_size, input_dim, seq_length, 1)
        #end_time = time.time()
        #print(f"pLM forward pass time: {end_time - start_time} seconds")
        return embeddings
    
    def _cnn_forward(self, embeddings):
        #start_time = time.time()
        res = self.CNN(embeddings).squeeze(-1).permute(0, 2, 1)  # (batch_size, seq_length, output_dim)
        out = {}
        if self.regression:
            out["regression"] = self.reg_head(res)
        if self.classification:
            out["classification"] = self.cls_head(res)
            #end_time = time.time()
            #print(f"Forward pass time: {end_time - start_time} seconds")
        return out

    def general_step(self, batch, stage):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print(f"{stage.capitalize()} batch with no valid residues - skipping") 
            return None  # Skip this batch
        
        outputs = self.forward(full_seq)
        loss = torch.tensor(0.0, device=self.device)
        if self.regression:
            reg_preds = outputs["regression"].squeeze(-1)
            mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
            self.log(f'{stage}_mse_loss', mse_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=True)
            loss += mse_loss

        if self.classification:
            cls_preds = outputs["classification"].squeeze(-1) if self.classification else None
            ce_loss = self.ce_loss_fn(cls_preds.flatten(0, 1), frst_classes.flatten()) # shape (batch_size, n_classes(3))
            self.log(f'{stage}_ce_loss', ce_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=True)
            loss += ce_loss

        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, 'val')
        
    def test_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Test batch with no valid residues - skipping") 
            return None  # Skip this batch

        embeddings = self._plm_forward(full_seq)
        #loss = torch.tensor(0.0, device=self.device)

        if self.mc_dropout_n > 0:
            self.CNN.train()
            self.reg_head.train() if self.regression else None
            self.cls_head.train() if self.classification else None
            test_mc_reg_preds = []  # to store mc dropout regression preds
            test_mc_cls_preds = []  # to store mc dropout classification preds
            #res_mask = res_mask.cpu().numpy()

            for _ in range(self.mc_dropout_n):
                outputs = self._cnn_forward(embeddings)
                if self.regression:
                    reg_preds = outputs["regression"].squeeze(-1)
                    test_mc_reg_preds.append(reg_preds)  # shape (batch_size, seq_length)

                if self.classification:
                    cls_preds = outputs["classification"].squeeze(-1)
                    test_mc_cls_preds.append(cls_preds)  # shape (batch_size, seq_length, n_classes)
            
            if self.regression:
                test_mc_reg_preds = torch.stack(test_mc_reg_preds).permute(1, 2, 0)  # to shape (seq_length, batch_size, mc_dropout_n)
                self.test_dict["mcd_regr_preds"].append(test_mc_reg_preds.cpu().numpy())  # shape (seq_length, batch_size, mc_dropout_n)
                self.test_dict["mcd_masked_regr_preds"].append(test_mc_reg_preds[res_mask].cpu().numpy())  # shape (seq_length, batch_size, mc_dropout_n)
                self.reg_head.eval()
            if self.classification:
                test_mc_cls_preds = torch.stack(test_mc_cls_preds).permute(1, 2, 0, 3)  # to shape (batch_size, seq_length, mc_dropout_n, n_classes)
                self.test_dict["mcd_cls_preds"].append(test_mc_cls_preds.cpu().numpy())  # shape (batch_size, seq_length, n_classes, mc_dropout_n)
                self.test_dict["mcd_masked_cls_preds"].append(test_mc_cls_preds[res_mask].cpu().numpy())  # shape (batch_size, seq_length, n_classes, mc_dropout_n)
                self.cls_head.eval()

        self.CNN.eval()
        outputs = self._cnn_forward(embeddings) 
        if self.regression:
            reg_preds = outputs["regression"].squeeze(-1)
            #mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)
            #self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            #loss += mse_loss
            self.test_dict["regr_preds"].append(reg_preds.cpu().numpy())
            self.test_dict["masked_regr_preds"].append(reg_preds[res_mask].flatten().cpu().numpy())

        if self.classification:
            cls_preds = outputs["classification"].squeeze(-1) 
            #ce_loss = self.ce_loss_fn(cls_preds.flatten(0, 1), frst_classes.flatten()) # shape (batch_size, n_classes(3))
            #self.log('test_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            #loss += ce_loss
            self.test_dict["cls_preds"].append(cls_preds.cpu().numpy())
            #self.test_dict["masked_cls_preds"].append(torch.argmax(cls_preds, dim=-1)[res_mask].flatten().cpu().numpy())
            self.test_dict["masked_cls_preds"].append(cls_preds[res_mask].flatten().cpu().numpy())

            #self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_dict["regr_targets"].append(frst_vals.cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].flatten().cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].flatten().cpu().numpy())

        self.test_dict["full_seqs"].append(np.array(full_seq))
        self.test_dict["masks"].append(res_mask.cpu().numpy())

        return None
    
    def predict_step(self, batch, batch_idx):
        full_seq, _, _, _ = batch
        preds = self.forward(full_seq)
        self.preds_list.append(preds.cpu().numpy())

    def configure_optimizers(self):
        print("Configuring optimizer...")
        params = list(self.CNN.parameters())
        print(f"Added CNN parameters. total num of tensors so far: {len(params)}")
        if self.finetune:
            params += list(self.encoder.parameters())
            print(f"Added encoder parameters. total num of tensors so far: {len(params)}")
        if self.regression:
            params += list(self.reg_head.parameters())
            print(f"Added regression head parameters. total num of tensors so far: {len(params)}")
        if self.classification:
            params += list(self.cls_head.parameters())
            print(f"Added classification head parameters. total num of tensors so far: {len(params)}")
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer

    def on_train_start(self):
        with open(f"./{self.experiment_name}/config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def on_train_end(self):
        pass
        # if self.use_loraFT is not None:
            # save the lora adapters
            #lora_save_path = f"./{self.experiment_name}/lora_adapters"
            #print(f"Saving LoRA adapters to {lora_save_path}")
            #self.encoder.save_pretrained(lora_save_path)
    
    def on_train_epoch_start(self):
        print(f"Starting training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def on_train_epoch_end(self):
        print(f"Ending training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def on_validation_start(self):
        print(f"Starting validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.val_dataloader
    
    def on_validation_end(self):
        print(f"Ending validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def on_test_epoch_start(self):
        self.test_dict = {"full_seqs": [],
                          "masks": [],
                          "regr_preds": [], 
                          "cls_preds": [], 
                          "regr_targets": [], 
                          "cls_targets": [],
                          "mcd_regr_preds": [],
                          "mcd_cls_preds": [],
                          "masked_regr_preds": [],
                          "mcd_masked_regr_preds": [],
                          "masked_cls_preds": [], 
                          "masked_regr_targets": [], 
                          "masked_cls_targets": [],
                          "mcd_masked_cls_preds": [],}

    def on_test_epoch_end(self):
        #concat the batches
        self.test_dict = {key: np.concatenate(value) for key, value in self.test_dict.items() if len(value) > 0}
        '''
        self.test_dict["full_seqs"] = np.concatenate(self.test_dict["full_seqs"])
        self.test_dict["masks"] = np.concatenate(self.test_dict["masks"])
        self.test_dict["regr_preds"] = np.concatenate(self.test_dict["regr_preds"])
        self.test_dict["cls_preds"] = np.concatenate(self.test_dict["cls_preds"])
        self.test_dict["regr_targets"] = np.concatenate(self.test_dict["regr_targets"])
        self.test_dict["cls_targets"] = np.concatenate(self.test_dict["cls_targets"])
        self.test_dict["masked_regr_preds"] = np.concatenate(self.test_dict["masked_regr_preds"])
        self.test_dict["masked_cls_preds"] = np.concatenate(self.test_dict["masked_cls_preds"])
        self.test_dict["masked_regr_targets"] = np.concatenate(self.test_dict["masked_regr_targets"])
        self.test_dict["masked_cls_targets"] = np.concatenate(self.test_dict["masked_cls_targets"])
        '''

    def on_predict_start(self):
        self.preds_list = []
    
    def on_predict_end(self):
        self.preds_list = np.concatenate(self.preds_list)

    def save_preds_dict(self, set="test"):
        np.savez_compressed(f"./{self.experiment_name}/{self.experiment_name}_{set}_preds.npz", **self.test_dict)

    @staticmethod
    def suggest_params():
        #TODO model selection
        pass



if __name__ == "__main__":
    
    parquet_path = "pLMtrainer/data/frustration/v4_frustration.parquet.gzip"
    data_module = FrustrationDataModule(df=None,
                                        parquet_path=parquet_path, 
                                        batch_size=10, 
                                        max_seq_length=100, 
                                        num_workers=1, 
                                        persistent_workers=True, 
                                        sample_size=1000,)
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
                       hidden_dims=[64,10], 
                       output_dim=4, 
                       dropout=0.15, 
                       max_seq_length=100,
                       precision="half",
                       pLM_model="pLMtrainer/data/ProtT5",  
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