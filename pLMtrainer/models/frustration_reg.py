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

class FrustrationCNN_REG(pl.LightningModule):
    def __init__(self, 
                 config):
        super(FrustrationCNN_REG, self).__init__()

        self.config = config
        self.experiment_name = config["experiment_name"]
        self.plm_model = config["pLM_model"]
        self.prefix_prostT5 = config["prefix_prostT5"]
        self.max_seq_length = config["max_seq_length"]  # if prost seq len will be max_seq_length-1 later since we prepend the prefix
        self.precision = config["precision"]
        self.finetune = config["finetune"]
        self.tokenizer = T5Tokenizer.from_pretrained(self.plm_model, do_lower_case=False, max_length=self.max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(self.plm_model).to(self.device)

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
        else:
            self.encoder.eval()  # Freeze the encoder   

        # https://github.com/RSchmirler/ProtT5-EvoTuning/blob/main/notebook/PT5_EvoTuning.ipynb 
        # lora modification
        #peft_config = LoraConfig(
        #    r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"], task_type = "SEQ_2_SEQ_LM",
        #)

        # https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
        self.CNN = nn.Sequential(
            nn.Conv2d(config["input_dim"], config["hidden_dims"][0], kernel_size=(7, 1), padding=(3, 0)),  # 7x64
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Conv2d(config["hidden_dims"][0], config["hidden_dims"][1], kernel_size=(7, 1), padding=(3, 0))
        )

        self.reg_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dims"][1], 1),
        )

        self.mse_loss_fn = nn.MSELoss()

    def forward(self, full_seq):
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

        res = self.CNN(embeddings).squeeze(-1).permute(0, 2, 1)  # (batch_size, seq_length, output_dim)
        reg_res = self.reg_head(res)
        #end_time = time.time()
        #print(f"Forward pass time: {end_time - start_time} seconds")
        return reg_res
    
    def general_step(self, batch, stage):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print(f"{stage.capitalize()} batch with no valid residues - skipping") 
            return None  # Skip this batch

        reg_preds = self.forward(full_seq)
        reg_preds = reg_preds.squeeze(-1)

        mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)

        self.log(f'{stage}_mse_loss', mse_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        return mse_loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, 'val')
        
    def test_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print("Test batch with no valid residues - skipping") 
            return None  # Skip this batch

        reg_preds = self.forward(full_seq)
        reg_preds = reg_preds.squeeze(-1)

        mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)

        self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_dict["full_seqs"].append(np.array(full_seq))
        self.test_dict["masks"].append(res_mask.cpu().numpy())

        self.test_dict["regr_preds"].append(reg_preds.cpu().numpy())
        self.test_dict["regr_targets"].append(frst_vals.cpu().numpy())

        self.test_dict["masked_regr_preds"].append(reg_preds[res_mask].flatten().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].flatten().cpu().numpy())

        return mse_loss

    def predict_step(self, batch, batch_idx):
        full_seq, _, _, _ = batch
        preds = self.forward(full_seq)
        self.preds_list.append(preds.cpu().numpy())

    def configure_optimizers(self):
        if self.finetune:
            params = list(self.CNN.parameters()) + \
                     list(self.encoder.parameters()) + \
                     list(self.reg_head.parameters())
            optimizer = torch.optim.Adam(params, lr=1e-3)
        else:
            params = list(self.CNN.parameters()) + \
                     list(self.reg_head.parameters())
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
    
    def on_validation_end(self):
        print(f"Ending validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def on_test_epoch_start(self):
        self.test_dict = {"full_seqs": [],
                          "masks": [],
                          "regr_preds": [],
                          "regr_targets": [], 
                          "masked_regr_preds": [],  
                          "masked_regr_targets": [], }

    def on_test_epoch_end(self):
        #concat the batches
        self.test_dict["full_seqs"] = np.concatenate(self.test_dict["full_seqs"])
        self.test_dict["masks"] = np.concatenate(self.test_dict["masks"])
        self.test_dict["regr_preds"] = np.concatenate(self.test_dict["regr_preds"])
        self.test_dict["regr_targets"] = np.concatenate(self.test_dict["regr_targets"])
        self.test_dict["masked_regr_preds"] = np.concatenate(self.test_dict["masked_regr_preds"])
        self.test_dict["masked_regr_targets"] = np.concatenate(self.test_dict["masked_regr_targets"])

    def on_predict_start(self):
        self.preds_list = []
    
    def on_predict_end(self):
        self.preds_list = np.concatenate(self.preds_list)

    def save_preds_dict(self):
        np.savez_compressed(f"./{self.experiment_name}/{self.experiment_name}_test_preds.npz", **self.test_dict)

    @staticmethod
    def suggest_params():
        #TODO model selection
        pass



if __name__ == "__main__":
    pass