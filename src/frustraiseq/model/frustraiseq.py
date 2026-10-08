import json
import os
import yaml
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl

from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import T5Config, T5EncoderModel, T5Tokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from frustraiseq.data.dataset import InferenceDataset, map_nonstandard_residues

# Frustration index thresholds for binning regression outputs into classes,
# equivalent to pd.cut(bins=[-inf, -1, 0.55, inf], labels=[0, 1, 2]):
# <= -1: highly frustrated (0), (-1, 0.55]: neutral (1), > 0.55: minimally frustrated (2)
FRUSTRATION_BIN_EDGES = [-1.0, 0.55]

HF_MODEL_REPO = "leuschj/FrustrAI-Seq"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# local paths that are meaningless outside the machine a model was saved on
_LOCAL_CONFIG_KEYS = ("pLM_model", "checkpoint_path", "inference_dataset")

class FrustrAISeq(pl.LightningModule):
    def __init__(self, config, encoder=None):
        """
        Args:
            config: model config dict (see DEFAULT_CONFIG)
            encoder: optional ready-to-use (merged) T5 encoder. If None, the pLM is loaded from
                config["pLM_model"] and wrapped with LoRA adapters for training.
        """
        super(FrustrAISeq, self).__init__()

        self.config = config
        self.experiment_name = config["experiment_name"]
        self.plm_model = config.get("pLM_model", "Rostlab/prot_t5_xl_uniref50")
        self.max_seq_length = config["max_seq_length"]

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = T5EncoderModel.from_pretrained(self.plm_model).to(self.device)
            if self.config["precision"] == "half":
                self.encoder.half()

            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                inference_mode=False,
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                bias="all", #! TODO
                target_modules=self.config['lora_modules'],
                #lora_dropout=0.1,
            )
            self.encoder.train()
            self.encoder = get_peft_model(self.encoder, peft_config)
        # https://github.com/RSchmirler/ProtT5-EvoTuning/blob/main/notebook/PT5_EvoTuning.ipynb 
        # peft_config = LoraConfig(r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"], task_type = "SEQ_2_SEQ_LM",)
        # https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
        #! future look into self.encoder.gradient_checkpointing_enable() - but appartently not working easily with DDP

        self.CNN = nn.Sequential(
            nn.Conv2d(config["architecture"]["pLM_dim"], 
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
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config["no_label_token"], 
                                              weight=torch.Tensor(config["ce_weighting"])) 
        
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/reg_aa_mean_preds_trainset.json'), 'r') as f:
            self.surprisal_dict = json.load(f)
        self.surprisal_mean = torch.tensor([self.surprisal_dict[aa]["mean"] for aa in self.surprisal_dict], device=self.device)
        self.surprisal_std = torch.tensor([self.surprisal_dict[aa]["std"] for aa in self.surprisal_dict], device=self.device)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.surprisal_dict)}

        if config["verbose"] and encoder is None:
            print(f"RANK {os.environ.get('RANK', -1)}: Loaded pLM model {self.plm_model}")
            print(f"RANK {os.environ.get('RANK', -1)}: Using LoRA fine-tuning for {self.config['lora_modules']} layers")
            self.encoder.print_trainable_parameters()
            print(f"RANK {os.environ.get('RANK', -1)}: Using {self.config['precision']} precision.")
            print(f"RANK {os.environ.get('RANK', -1)}: Applying class weights for CrossEntropyLoss:", config["ce_weighting"])
            print(f"RANK {os.environ.get('RANK', -1)}: Loaded surprisal dictionary.",)
            print(f"RANK {os.environ.get('RANK', -1)}: Model initialized.")

    def _plm_forward(self, input_ids, attention_mask):
        embeddings = self.encoder(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device)
            ).last_hidden_state.float()
        embeddings = embeddings.permute(0, 2, 1).unsqueeze(-1)  # (batch_size, input_dim, seq_length, 1)
        return embeddings
    
    def _cnn_forward(self, embeddings):
        res = self.CNN(embeddings).squeeze(-1).permute(0, 2, 1)  # (batch_size, seq_length, output_dim)
        return {"regression": self.reg_head(res), "classification": self.cls_head(res)}

    def forward(self, batch, stage, sync_dist=False):
        input_ids, attention_mask, res_mask, frst_vals, frst_classes, _, _, _ = batch
        if res_mask.sum() == 0:
            print(f"{stage.capitalize()} batch with no valid residues - skipping") 
            return None

        emb = self._plm_forward(input_ids, attention_mask)  # (batch_size, seq_length, input_dim)
        out = self._cnn_forward(emb)  # dict with regression and/or classification outputs
        
        reg_preds = out["regression"].squeeze(-1)
        mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)

        cls_preds = out["classification"].squeeze(-1)
        ce_loss = self.ce_loss_fn(cls_preds.flatten(0, 1), frst_classes.flatten()) # shape (batch_size, n_classes(3))

        loss = mse_loss + ce_loss

        self.log(f'{stage}_mse_loss', mse_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=(stage!='train'))
        self.log(f'{stage}_ce_loss', ce_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, sync_dist=(stage!='train'))
        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=(stage!='train'))
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, 'val')

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, res_mask, frst_vals, frst_classes, _, _, _ = batch
        if res_mask.sum() == 0:
            print("Test batch with no valid residues - skipping") 
            return None 

        embeddings = self._plm_forward(input_ids, attention_mask)
        outputs = self._cnn_forward(embeddings)

        #regr preds
        reg_preds = outputs["regression"].squeeze(-1)
        self.test_dict["regr_preds"].append(reg_preds.detach().float().cpu().numpy())
        self.test_dict["masked_regr_preds"].append(reg_preds[res_mask].detach().float().cpu().numpy())
        #cls preds
        cls_preds = outputs["classification"].squeeze(-1) 
        self.test_dict["cls_preds_logits"].append(cls_preds.detach().float().cpu().numpy())
        self.test_dict["masked_cls_preds_logits"].append(cls_preds[res_mask].detach().float().cpu().numpy())
        self.test_dict["cls_preds"].append(torch.argmax(cls_preds, dim=-1).detach().int().cpu().numpy())
        self.test_dict["masked_cls_preds"].append(torch.argmax(cls_preds, dim=-1)[res_mask].detach().int().cpu().numpy())
        #targets
        self.test_dict["regr_targets"].append(frst_vals.detach().float().cpu().numpy())
        self.test_dict["masked_regr_targets"].append(frst_vals[res_mask].detach().float().cpu().numpy())
        self.test_dict["cls_targets"].append(frst_classes.detach().int().cpu().numpy())
        self.test_dict["masked_cls_targets"].append(frst_classes[res_mask].detach().int().cpu().numpy())
        #misc
        self.test_dict["input_ids"].append(np.array(input_ids.detach().cpu()))
        self.test_dict["full_seqs"].append(np.array(self.tokenizer.batch_decode(input_ids)))
        self.test_dict["masks"].append(res_mask.detach().bool().cpu().numpy())
        
        #test loss
        reg_preds = outputs["regression"].squeeze(-1)
        mse_loss = self.mse_loss_fn(reg_preds[res_mask], frst_vals[res_mask]) # shape (batch_size, 1)

        cls_preds = outputs["classification"].squeeze(-1)
        ce_loss = self.ce_loss_fn(cls_preds.flatten(0, 1), frst_classes.flatten()) # shape (batch_size, n_classes(3))

        loss = mse_loss + ce_loss

        self.log(f'test_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'test_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, full_seq, ids = batch

        try:
            embeddings = self._plm_forward(input_ids, attention_mask)
        except RuntimeError as e:
            print(f"RuntimeError during PLM forward pass: {e}")
            return None

        outputs   = self._cnn_forward(embeddings)
        reg_preds = outputs["regression"].squeeze(-1)   # (B, L)
        cls_logits = outputs["classification"]          # (B, L, 3)

        # entropy of the cls head is only reported when the cls head also provides the class
        use_cls_heads = self.config.get("use_cls_heads_output_for_class_pred", False)
        if use_cls_heads:
            probs     = torch.softmax(cls_logits, dim=-1)   # (B, L, 3)
            cls_preds = probs.argmax(dim=-1)                # (B, L)
            log_n     = torch.log(torch.tensor(float(probs.shape[-1]), device=self.device))
            entropies = -(probs * (probs + 1e-9).log()).sum(dim=-1) / log_n  # (B, L)
        else:
            bin_edges = torch.tensor(FRUSTRATION_BIN_EDGES, device=reg_preds.device)
            cls_preds = torch.bucketize(reg_preds.float(), bin_edges)  # (B, L)

        surp_mean = self.surprisal_mean.to(self.device)
        surp_std  = self.surprisal_std.to(self.device)

        rows = []
        for i, (seq, seq_id) in enumerate(zip(full_seq, ids)):
            L = min(len(seq), reg_preds.shape[1])  # clamp for truncated sequences

            reg_i = reg_preds[i, :L].float()
            cls_i = cls_preds[i, :L]

            aa_idx  = torch.tensor([self.aa_to_idx[aa] for aa in seq],
                                    dtype=torch.long, device=self.device)
            surp_i  = (reg_i - surp_mean[aa_idx]) / surp_std[aa_idx]

            # single CPU transfer per sequence
            reg_np  = reg_i.cpu().numpy()
            cls_np  = cls_i.cpu().numpy()
            ent_np  = entropies[i, :L].float().cpu().numpy() if use_cls_heads else None
            surp_np = surp_i.cpu().numpy()

            for j, aa in enumerate(seq):
                row = {
                    "id":                seq_id,
                    "residue":           aa,
                    "frustration_index": float(reg_np[j]),
                    "frustration_class": int(cls_np[j]),
                }
                if use_cls_heads:
                    row["entropy"] = float(ent_np[j])
                row["surprisal"] = float(surp_np[j])
                rows.append(row)

        return rows

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

    @rank_zero_only
    def on_train_start(self):
        if self.trainer.global_rank == 0:
            with open(f"./{self.experiment_name}/config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def on_train_epoch_start(self):
        if self.config["verbose"]:
            print(f"Starting training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_train_epoch_end(self):
        if self.config["verbose"]:
            print(f"Ending training epoch {self.current_epoch} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_validation_start(self):
        if self.config["verbose"]:
            print(f"Starting validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
    
    def on_validation_end(self):
        if self.config["verbose"]:
            print(f"Ending validation {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

    def on_test_epoch_start(self):
        self.test_dict = {"full_seqs": [],
                          "input_ids": [],
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

        self.tokenizer = T5Tokenizer.from_pretrained(self.config["pLM_model"], 
                                                     do_lower_case=False, 
                                                     max_length=self.config["max_seq_length"])

    def on_test_epoch_end(self):
        #concat the batches
        self.test_dict = {key: np.concatenate(value) for key, value in self.test_dict.items() if len(value) > 0}

    @rank_zero_only
    def save_preds_dict(self, set="test"):
        print(f"TRAINER RANK {self.trainer.global_rank}. Saving preds.")
        print(f"OS RANK {os.environ.get('RANK', -1)}. Saving preds.")
        if self.trainer.global_rank == 0:
            np.savez_compressed(f"./{self.experiment_name}/{set}_preds.npz", **self.test_dict)

    def save_pretrained(self, save_directory):
        """
        Save config, weights and tokenizer so the model can be loaded with `FrustrAISeq.from_pretrained`.
        The encoder must be a plain (LoRA-merged) T5 encoder, see `frustraiseq/utils/export_pretrained.py`.
        """
        if isinstance(self.encoder, PeftModel):
            raise ValueError("Merge the LoRA adapters first (self.encoder = self.encoder.merge_and_unload()).")
        os.makedirs(save_directory, exist_ok=True)

        config = {k: v for k, v in self.config.items() if k not in _LOCAL_CONFIG_KEYS}
        config["plm_config"] = self.encoder.config.to_dict()
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump(config, f, indent=2)

        # store tied tensors (shared / embed_tokens) only once; the loss weights are not needed
        state_dict, seen = {}, set()
        for key, tensor in self.state_dict().items():
            if key.startswith("ce_loss_fn") or tensor.data_ptr() in seen:
                continue
            seen.add(tensor.data_ptr())
            state_dict[key] = tensor.contiguous()
        save_file(state_dict, os.path.join(save_directory, WEIGHTS_NAME), metadata={"format": "pt"})

        T5Tokenizer.from_pretrained(self.plm_model, do_lower_case=False).save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path=HF_MODEL_REPO, revision=None, **config_overrides):
        """
        Load a model saved with `save_pretrained` from a local directory or the HuggingFace Hub.

        Args:
            model_name_or_path: HF repo id (default: leuschj/FrustrAI-Seq) or local directory
            revision: optional HF revision (branch, tag or commit)
            **config_overrides: config entries to override, e.g. use_cls_heads_output_for_class_pred=True
        """
        if os.path.isdir(model_name_or_path):
            model_dir = model_name_or_path
        else:
            model_dir = snapshot_download(model_name_or_path, revision=revision)

        with open(os.path.join(model_dir, CONFIG_NAME), "r") as f:
            config = json.load(f)
        plm_config = T5Config.from_dict(config.pop("plm_config"))
        config.update(config_overrides)
        config["pLM_model"] = model_dir  # tokenizer files live next to the weights

        # build the encoder without allocating memory; the weights are assigned from the safetensors file
        with torch.device("meta"):
            encoder = T5EncoderModel(plm_config)
        model = cls(config, encoder=encoder)
        result = model.load_state_dict(load_file(os.path.join(model_dir, WEIGHTS_NAME)), strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected keys in {WEIGHTS_NAME}: {result.unexpected_keys}")
        model.encoder.tie_weights()  # embed_tokens shares the (stored once) shared embedding
        still_meta = [name for name, p in model.named_parameters() if p.is_meta]
        if still_meta:
            raise ValueError(f"Missing weights in {WEIGHTS_NAME}: {still_meta}")
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, sequences, batch_size=1, device=None):
        """
        Predict per-residue frustration for protein sequences.

        Args:
            sequences: dict {id: sequence}, list of sequences, or DataFrame with columns 'id' and 'sequence'
            batch_size: number of sequences per forward pass (sequences are padded to the longest in the batch)
            device: torch device; default cuda > mps > cpu

        Returns:
            DataFrame with one row per residue: id, residue, frustration_index, frustration_class,
            [entropy,] surprisal
        """
        if isinstance(sequences, pd.DataFrame):
            ids, seqs = sequences["id"].tolist(), sequences["sequence"].tolist()
        elif isinstance(sequences, dict):
            ids, seqs = list(sequences.keys()), list(sequences.values())
        else:
            seqs = list(sequences)
            ids = [f"seq_{i}" for i in range(len(seqs))]
        seqs = map_nonstandard_residues(ids, seqs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.to(device)
        self.eval()

        dataset = InferenceDataset(self.config, ids, seqs)
        rows = []
        starts = range(0, len(dataset), batch_size)
        for batch_idx, start in enumerate(tqdm(starts, desc="Predicting", disable=not self.config.get("verbose", True))):
            batch = dataset.collate_fn([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            batch_rows = self.predict_step(batch, batch_idx)
            if batch_rows is not None:
                rows.extend(batch_rows)
        return pd.DataFrame(rows)
