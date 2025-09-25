import sys
import tqdm
import torch
import numpy as np
import scanpy as sc
import pyarrow.parquet as pq

from transformers import T5Tokenizer, T5EncoderModel

torch.set_float32_matmul_precision('medium')

parquet_path = "../data/frustration/v3_frustration.parquet.gzip"
df = pq.read_table(parquet_path).to_pandas()
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pLM_model = "../data/prostT5"
prefix_prostT5 = "<AA2fold>"
max_seq_length = 700 + 1
tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
encoder = T5EncoderModel.from_pretrained(pLM_model).to(device)

if "cuda" in device:
    encoder.half()
    print("Using half precision for the pLM encoder")

frust_avail = []
embeddings = []
res_labels = []
frst_val = []
frst_class = []

for row in tqdm.tqdm(df.iterrows()):
    if row[0] % 100 == 0:
        print(row[0])
    idx = min(len(row[1]["full_seq"]), max_seq_length)

    full_seq = [prefix_prostT5 + " " + " ".join(seq) for seq in [row[1]["full_seq"]]]  # Add spaces between amino acids and prefix
    ids = tokenizer.batch_encode_plus(full_seq, 
                                    add_special_tokens=True, 
                                    max_length=max_seq_length,
                                    padding="max_length",
                                    truncation="longest_first",
                                    return_tensors='pt'
                                    ).to(device)
        
    with torch.no_grad():
        embedding_rpr = encoder(
            ids.input_ids, 
            attention_mask=ids.attention_mask
        )
    emb = embedding_rpr.last_hidden_state[0, 1:idx,]
    embeddings.append(emb)
    res_labels.append(list(row[1]["full_seq"])[:idx-1])

    frsts = torch.zeros((len(row[1]["full_seq"])), dtype=torch.bool)
    frsts[row[1]["res_idx"]] = True
    frust_avail.append(frsts[:idx-1])
    vals = np.zeros((len(row[1]["full_seq"])))
    vals[:] = np.nan
    vals[row[1]["res_idx"]] = row[1]["frst_idx"]
    frst_val.append(vals[:idx-1])
    classes = np.zeros((len(row[1]["full_seq"])), dtype=object)
    classes[row[1]["res_idx"]] = row[1]["frst_class"]
    frst_class.append(classes[:idx-1])

adata = sc.AnnData(X=torch.cat(embeddings, dim=0).cpu().numpy())
adata.obs['frustration'] = torch.cat(frust_avail, dim=0).cpu().numpy()
adata.obs['residue'] = np.concatenate(res_labels, axis=0)
adata.obs['frst_val'] = np.concatenate(frst_val, axis=0)
adata.obs['frst_class'] = np.concatenate(frst_class, axis=0)
adata.obs['frst_class'] = adata.obs['frst_class'].astype('category')

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.write(f"frustration_adata_{max_seq_length-1}.h5ad", adata)