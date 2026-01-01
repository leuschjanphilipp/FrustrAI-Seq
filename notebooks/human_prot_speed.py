import sys
import yaml
import json
import tqdm
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import pyarrow.parquet as pq
from pytorch_lightning import Trainer

from matplotlib import pyplot as plt
from transformers import T5Tokenizer, T5EncoderModel

sys.path.append('..')
sys.path.append('FrustraSeq')
from FrustraSeq.models.FrustraSeq import FrustraSeq
from FrustraSeq.dataloader import FrustrationDataModule

fasta_file_path = "../data/frustration/uniprotkb_human_AND_model_organism_9606_2025_12_31.fasta"

seqs = {}
with open(fasta_file_path, 'r') as f:
    fasta_data = f.read()
    for line in fasta_data.splitlines():
        if line.startswith(">"):
            header = line[1:].split('|')[1]
            seqs[header] = ""
        else:
            seqs[header] += line.strip()

seqs = dict(sorted(seqs.items(), key=lambda item: len(item[1])))
df = pd.DataFrame.from_dict(seqs, orient='index', columns=["sequence"]).reset_index().rename(columns={'index':'id'})
df["sequence"] = df["sequence"].apply(lambda x: x.replace("U", "X").replace("O", "X").replace("B", "X").replace("Z", "X")).replace("X", "A")


# load config
with open(f"../data/it5_ABL_protT5_CW_LORA/config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config["pLM_model"] = "../data/protT5"
config["max_seq_length"] = len(seqs[list(seqs.keys())[-1]])

model = FrustraSeq.load_from_checkpoint(checkpoint_path=f"../data/{config['experiment_name']}/best_val_model.ckpt",
                                        config=config)

with open('../data/frustration/reg_heuristic.json', 'r') as f:
    model.surprisal_dict = json.load(f)

trainer = Trainer(accelerator='gpu',
                  devices=1, 
                  precision="bf16-mixed")

predict_dataloader = FrustrationDataModule(df=df,
                                        max_seq_length=df["sequence"].str.len().max(),
                                        batch_size=1,
                                        num_workers=10,
                                        persistent_workers=True,)
start_time = time.time()

trainer.predict(model, predict_dataloader)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time for evaluating {len(df)} sequences: {total_time:.2f} seconds")
