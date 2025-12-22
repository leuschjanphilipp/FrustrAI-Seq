import torch
import pyarrow.parquet as pq

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class FrustrationDataModule(LightningDataModule):
    def __init__(self,
                 df, 
                 parquet_path=None,
                 max_seq_length=512,
                 batch_size=16,
                 set_key="set", 
                 num_workers=1,
                 persistent_workers=True,
                 pin_memory=True,
                 prefetch_factor=2,
                 sample_size=None,
                 cath_sampling_n=None):
        super().__init__()
        self.df = df
        self.parquet_path = parquet_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.set_key = set_key
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.sample_size = sample_size
        self.cath_sampling_n = cath_sampling_n

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.parquet_path is not None:
                self.df = pq.read_table(self.parquet_path).to_pandas()
            if self.cath_sampling_n is not None:
                grouped = self.df.groupby("cath_T_id")
                self.df = grouped.sample(n=self.cath_sampling_n, replace=True, random_state=42).reset_index(drop=True)
            if self.sample_size is not None:
                self.df = self.df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)  # Shuffle the dataframe
            print(f"Loaded {len(self.df)} samples from {self.parquet_path}")
            #Masking
            train_mask = self.df[self.set_key] == "train"
            val_mask = self.df[self.set_key] == "val"
            test_mask = self.df[self.set_key] == "test"
            print("Created train/val/test masks")

            max_len = self.df["full_seq"].str.len().max()
            res_idx_mask = torch.zeros((len(self.df), max_len), dtype=torch.bool)
            frst_vals = torch.zeros((len(self.df), max_len), dtype=torch.float)
            frst_vals[:] = -100  # Initialize to -100 for no frst value info
            frst_classes = torch.zeros((len(self.df), max_len), dtype=torch.long)
            frst_classes[:] = -100  # Initialize to -100 for no frst class info
            print("Initialized res_idx_mask and frst_vals tensors")
            for i, idx_list in enumerate(self.df["res_idx"]):
                res_idx_mask[i, idx_list] = True
                frst_vals[i, idx_list] = torch.Tensor(self.df["frst_idx"][i]) # regression values
                frst_classes[i, idx_list] = torch.LongTensor(self.df["frst_class_3"][i]) 

            res_idx_mask = res_idx_mask[:, :self.max_seq_length]
            frst_vals = frst_vals[:, :self.max_seq_length]
            frst_classes = frst_classes[:, :self.max_seq_length]
            print("Populated res_idx_mask and frst_vals tensors")

            self.train_dataset = FrustrationDataset(self.df[train_mask]["full_seq"].tolist(),
                                                    res_idx_mask[train_mask],
                                                    frst_vals[train_mask],
                                                    frst_classes[train_mask])
            print("Created train dataset")
            self.val_dataset = FrustrationDataset(self.df[val_mask]["full_seq"].tolist(),
                                                    res_idx_mask[val_mask],
                                                    frst_vals[val_mask],
                                                    frst_classes[val_mask])
            print("Created val dataset")
            self.test_dataset = FrustrationDataset(self.df[test_mask]["full_seq"].tolist(),
                                                    res_idx_mask[test_mask],
                                                    frst_vals[test_mask],
                                                    frst_classes[test_mask])
            print("Created test dataset")
            print(f"Train/Val/Test split: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)} samples")
        
        if stage == "predict":
            # Shuffle the dataframe
            assert self.df is not None, "Dataframe must be provided for prediction stage."
            assert self.df["sequence"].notnull().all(), "All sequences must be non-null for prediction."
            assert self.df["id"].notnull().all(), "All ids must be non-null for prediction."
            print(f"Loaded {len(self.df)} sequences for prediction.")
            max_len = self.df["sequence"].str.len().max() # get max length of sequences in self.df
            
            res_idx_mask = torch.zeros((len(self.df), max_len), dtype=torch.bool)
            frst_vals = torch.zeros((len(self.df), max_len), dtype=torch.float)
            frst_classes = torch.zeros((len(self.df), max_len), dtype=torch.long)

            res_idx_mask = res_idx_mask[:, :self.max_seq_length]
            frst_vals = frst_vals[:, :self.max_seq_length]
            frst_classes = frst_classes[:, :self.max_seq_length]

            self.predict_dataset = FrustrationDataset(self.df["sequence"].tolist(),
                                                        res_idx_mask,
                                                        frst_vals,
                                                        frst_classes)
            print("Created test dataset for prediction")
            print(f"Test dataset size: {len(self.predict_dataset)} samples")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          prefetch_factor=self.prefetch_factor)

class FrustrationDataset(Dataset):
    def __init__(self, full_seq, res_idx, frst_vals, frst_classes):
        self.full_seq = full_seq
        self.res_idx = res_idx
        self.frst_vals = frst_vals
        self.frst_classes = frst_classes

    def __len__(self):
        return len(self.full_seq)

    def __getitem__(self, idx):
        return self.full_seq[idx], self.res_idx[idx], self.frst_vals[idx], self.frst_classes[idx]