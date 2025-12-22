
import os
import sys
import yaml
import time
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import entropy
from scipy.special import softmax

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import LinearLR
from transformers import T5Tokenizer, T5EncoderModel, EsmModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.append('..')
sys.path.append('pLMtrainer')

class FrustraSeq(pl.LightningModule):
    def __init__(self, config):
        super(FrustraSeq, self).__init__()

        self.config = config
        self.experiment_name = config["experiment_name"]
        self.plm_model = config["pLM_model"]
        self.prefix_prostT5 = config["prefix_prostT5"]
        self.max_seq_length = config["max_seq_length"]
        
        if "esm" in self.plm_model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.plm_model, max_length=self.max_seq_length)
            self.encoder = EsmModel.from_pretrained(self.plm_model).to(self.device)

        elif "t5" in self.plm_model.lower():
            if "prostt5" in self.plm_model.lower():
                self.max_seq_length += 1  # for prostT5, add 1 for bos token <AA2fold> #TODO use 511 aas to stay at 512 (div by 8)
            self.tokenizer = T5Tokenizer.from_pretrained(self.plm_model, do_lower_case=False, max_length=self.max_seq_length)
            self.encoder = T5EncoderModel.from_pretrained(self.plm_model).to(self.device)
        
        if self.config["finetune"]:
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
            # peft_config = LoraConfig(r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"], task_type = "SEQ_2_SEQ_LM",)
            # https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
            #! future look into self.encoder.gradient_checkpointing_enable() - but appartently not working easily with DDP
        else:
            self.encoder.eval()

        self.CNN = nn.Sequential(
            nn.Conv2d(config["pLM_dim"], 
                      config["architecture"]["hidden_dim_0"], 
                      kernel_size=config["architecture"]["kernel_1"], 
                      padding=config["architecture"]["padding_1"]),  # 7x64
            nn.ReLU(),
            nn.Dropout(config["architecture"]["dropout"]),
            nn.Conv2d(config["architecture"]["hidden_dim_0"], 
                      config["architecture"]["hidden_dim_1"], 
                      kernel_size=config["architecture"]["kernel_2"], 
                      padding=config["architecture"]["padding_2"])
        )
        #TODO separate reg head dim and cls head dim maybe?
        self.reg_head = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(config["architecture"]["dropout"]),
        nn.Linear(config["architecture"]["hidden_dim_1"], 1),
        )
        self.cls_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config["architecture"]["dropout"]),
            nn.Linear(config["architecture"]["hidden_dim_1"], 3),  # 3 classes
        )
        self.mse_loss_fn = nn.MSELoss()
        if config["ce_weighting"] is not None:
            self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config["no_label_token"], 
                                                  weight=torch.Tensor(config["ce_weighting"])) 
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config["no_label_token"])

        if self.config["use_focal_loss_instead_of_ce"]:
            print("Using focal loss instead of cross-entropy loss for classification head. Overrides ce_weighting if set.")
            self.ce_loss_fn = self.focal_loss

        print(f"RANK {os.environ.get('RANK', -1)}: Model initialized.")

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
        if self.config["finetune"]:
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
        #end_time = time.time()
        #print(f"CNN forward pass time: {end_time - start_time} seconds")
        return {"regression": self.reg_head(res), "classification": self.cls_head(res)}

    def general_step(self, batch, stage):
        full_seq, res_mask, frst_vals, frst_classes = batch
        if res_mask.sum() == 0:
            print(f"{stage.capitalize()} batch with no valid residues - skipping") 
            return None
        
        outputs = self.forward(full_seq)

        reg_preds = outputs["regression"].squeeze(-1)
        mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)

        cls_preds = outputs["classification"].squeeze(-1)
        ce_loss = self.ce_loss_fn(cls_preds.flatten(0, 1), frst_classes.flatten()) # shape (batch_size, n_classes(3))

        loss = mse_loss + ce_loss

        self.log(f'{stage}_mse_loss', mse_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}_ce_loss', ce_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=True)
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
        outputs = self._cnn_forward(embeddings) 

        reg_preds = outputs["regression"].squeeze(-1)
        self.test_dict["regr_preds"].append(reg_preds.detach().float().cpu().numpy())
        self.test_dict["masked_regr_preds"].append(reg_preds[res_mask].detach().float().cpu().numpy())

        cls_preds = outputs["classification"].squeeze(-1) 
        self.test_dict["cls_preds_logits"].append(cls_preds.detach().float().cpu().numpy())
        self.test_dict["masked_cls_preds_logits"].append(cls_preds[res_mask].detach().float().cpu().numpy())
        self.test_dict["cls_preds"].append(torch.argmax(cls_preds, dim=-1).detach().int().cpu().numpy())
        self.test_dict["masked_cls_preds"].append(torch.argmax(cls_preds, dim=-1)[res_mask].detach().int().cpu().numpy())

        self.test_dict["regr_targets"].append(frst_vals.detach().float().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].detach().float().cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.detach().int().cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].detach().int().cpu().numpy())

        self.test_dict["full_seqs"].append(np.array(full_seq))
        self.test_dict["masks"].append(res_mask.detach().bool().cpu().numpy())

        return None
    
    def predict_step(self, batch, batch_idx):
        #if self.inference_precision == "half":
            #self.encoder.half()
            #self.CNN.half()
            #print("Using half precision for pLM encoder")
        full_seq, _, _, _ = batch
        embeddings = self._plm_forward(full_seq)
        outputs = self._cnn_forward(embeddings)
        reg_preds = outputs["regression"].squeeze(-1) # shape (batch_size, seq_length)
        cls_preds = outputs["classification"].squeeze(-1) # shape (batch_size, seq_length, n_classes(3))

        for seq, reg_pred, cls_pred in zip(full_seq, reg_preds, cls_preds):
            idx = len(seq)
            reg_pred = reg_pred[:idx].detach().float().cpu().numpy()
            entropies = entropy(softmax(cls_pred[:idx].detach().float().cpu().numpy(), axis=-1), axis=-1) / np.log(cls_preds.shape[-1])
            cls_pred = torch.argmax(cls_pred[:idx], dim=-1)[:idx].detach().int().cpu().numpy()
            res = {
                "residue": list(seq),
                "frustration_index": reg_pred,
                "frustration_class": cls_pred,
                "entropy": entropies,
            }
            if self.surprisal_dict is not None:
                z_score = []
                for aa, val in zip(seq, reg_pred):
                    mean = self.surprisal_dict[aa]["mean"]
                    std = self.surprisal_dict[aa]["std"]
                    z = (val - mean) / std
                    z_score.append(z)
                res["surprisal"] = z_score
            else:
                print("No surprisal dictionary provided - skipping surprisal z-score calculation.")
            self.pred_list.append(res)
        return None

    def configure_optimizers(self):

        lora_params = [p for n,p in self.encoder.named_parameters() if p.requires_grad and "lora" in n]
        head_params = [p for n,p in self.named_parameters() if p.requires_grad and "cls_head" in n or "reg_head" in n or "CNN" in n]

        optimizer_grouped_parameters = [
            {"params": lora_params,
            "lr": self.config["architecture"]["lr"] / 3, "weight_decay": 0.0},
            {"params": head_params,
            "lr": self.config["architecture"]["lr"], "weight_decay": 0.01},
            ]
        print(f"RANK {os.environ.get('RANK', -1)}: lora params: {len(lora_params)}, head params: {len(head_params)}")

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

        warmup_steps = 500
        total_steps = self.trainer.estimated_stepping_batches

        warmup_scheduler = LinearLR(optimizer, start_factor=0.3, end_factor=1, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_steps - warmup_steps), eta_min=self.config["architecture"]["lr"] * 0.1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine"
            }
        }

    def on_train_start(self):
        if self.trainer.global_rank == 0:
            with open(f"./{self.experiment_name}/config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def on_train_end(self):
        pass
    
    def on_train_epoch_start(self):
        print(f"Starting training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_train_epoch_end(self):
        print(f"Ending training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_validation_start(self):
        print(f"Starting validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
        self.val_dataloader
    
    def on_validation_end(self):
        print(f"Ending validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_test_epoch_start(self):
        self.test_dict = {"full_seqs": [],
                          "masks": [],
                          "regr_preds": [], 
                          "cls_preds": [], 
                          "regr_targets": [], 
                          "cls_targets": [],
                          "masked_regr_preds": [],
                          "masked_cls_preds": [], 
                          "masked_regr_targets": [], 
                          "masked_cls_targets": [],
                          "cls_preds_logits": [],
                          "masked_cls_preds_logits": []}

    def on_test_epoch_end(self):
        #concat the batches
        self.test_dict = {key: np.concatenate(value) for key, value in self.test_dict.items() if len(value) > 0}

    def on_predict_start(self):
        self.pred_list = []
    
    def on_predict_end(self):
        #self.preds_dict = {key: np.concatenate(value) for key, value in self.pred_dict.items() if len(value) > 0}
        pass

    @rank_zero_only
    def save_preds_dict(self, set="test"):
        print(f"TRAINER RANK {self.trainer.global_rank}. Save preds.")
        print(f"OS RANK {os.environ.get('RANK', -1)}. Save preds.")

        if self.trainer.global_rank == 0:
            np.savez_compressed(f"./{self.experiment_name}/{set}_preds.npz", **self.test_dict)

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config["no_label_token"], reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    @staticmethod
    #@rank_zero_only
    def suggest_params(trial):
        print(f"OS RANK {os.environ.get('RANK', -1)}. Suggesting params.")
        architecture = {}
        architecture["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        # Architecture
        architecture["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        architecture["kernel_1"] = trial.suggest_categorical("kernel_1", [3,5,7,9])
        architecture["padding_1"] = architecture["kernel_1"] // 2  # to keep same length
        architecture["kernel_2"] = trial.suggest_categorical("kernel_2", [3,5,7,9])
        architecture["padding_2"] = architecture["kernel_2"] // 2  # to keep same length
        architecture["hidden_dim_0"] = trial.suggest_categorical("hidden_dim_0", [32, 64, 128, 256, 512])
        architecture["hidden_dim_1"] = trial.suggest_categorical("hidden_dim_1", [8, 16, 32, 64])
        return architecture