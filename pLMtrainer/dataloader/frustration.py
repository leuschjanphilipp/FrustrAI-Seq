import torch
import pyarrow.parquet as pq

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class FrustrationDataModule(LightningDataModule):
    def __init__(self, 
                 parquet_path,
                 max_seq_length=700,
                 regression=True,
                 batch_size=64, 
                 num_workers=1,
                 persistent_workers=True):
        super().__init__()
        self.parquet_path = parquet_path
        self.max_seq_length = max_seq_length
        self.regression = regression

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        df = pq.read_table(self.parquet_path).to_pandas()
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)  # Shuffle the dataframe
        print(f"Loaded {len(df)} samples from {self.parquet_path}")
        #Masking
        train_mask = df["set"] == "train"
        val_mask = df["set"] == "val"
        test_mask = df["set"] == "test"
        print("Created train/val/test masks")

        max_len = df["full_seq"].str.len().max()
        res_idx_mask = torch.zeros((len(df), max_len), dtype=torch.bool)
        if self.regression:
            frst_vals = torch.zeros((len(df), max_len), dtype=torch.float)
        else:
            frst_vals = torch.zeros((len(df), max_len), dtype=torch.long)
        print("Initialized res_idx_mask and frst_vals tensors")
        for i, idx_list in enumerate(df["res_idx"]):
            res_idx_mask[i, idx_list] = True
            if self.regression:
                frst_vals[i, idx_list] = torch.Tensor(df["frst_idx"][i])
            else:
                frst_vals[i, idx_list] = torch.LongTensor(df["frst_class"][i]) + 1 # frst classes range from 1-20 with 0 being no info.

        frst_vals = frst_vals[:, :self.max_seq_length]
        res_idx_mask = res_idx_mask[:, :self.max_seq_length]
        print("Populated res_idx_mask and frst_vals tensors")

        self.train_dataset = FrustrationDataset(df[train_mask]["full_seq"].tolist(),
                                                res_idx_mask[train_mask],
                                                frst_vals[train_mask])
        print("Created train dataset")
        self.val_dataset = FrustrationDataset(df[val_mask]["full_seq"].tolist(),
                                                res_idx_mask[val_mask],
                                                frst_vals[val_mask])
        print("Created val dataset")
        self.test_dataset = FrustrationDataset(df[test_mask]["full_seq"].tolist(),
                                                res_idx_mask[test_mask],
                                                frst_vals[test_mask])
        print("Created test dataset")
        print(f"Train/Val/Test split: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

class FrustrationDataset(Dataset):
    def __init__(self, full_seq, res_idx, frst_idx):
        self.full_seq = full_seq
        self.res_idx = res_idx
        self.frst_idx = frst_idx

    def __len__(self):
        return len(self.full_seq)

    def __getitem__(self, idx):
        return self.full_seq[idx], self.res_idx[idx], self.frst_idx[idx]