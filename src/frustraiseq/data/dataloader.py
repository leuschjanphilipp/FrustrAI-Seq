import torch
import pandas as pd
import pyarrow.parquet as pq

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import FunstrationDataset, InferenceDataset
from datasets import load_dataset

class FunstrationDataModule(LightningDataModule):
    def __init__(self,
                 config,
                 fit_dataset="leuschj/Funstration",
                 inference_dataset=None,
                 max_seq_length=512,
                 batch_size=16,
                 split_key="split_0", 
                 num_workers=1,
                 persistent_workers=True,
                 pin_memory=True,
                 prefetch_factor=2,
                 sample_size=None,
                 cath_sampling_n=None):
        super().__init__()

        self.config = config
        self.fit_dataset = fit_dataset
        self.inference_dataset = inference_dataset
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.split_key = split_key
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.sample_size = sample_size
        self.cath_sampling_n = cath_sampling_n

    def _prepare_funstration_fit_dataset(self):
        if isinstance(self.fit_dataset, pd.DataFrame):
            print(f"Using provided DataFrame with {len(self.fit_dataset)} samples.")
        elif isinstance(self.fit_dataset, str):
            try:
                self.fit_dataset = load_dataset(self.fit_dataset)
                print(f"Loaded dataset with {len(self.fit_dataset)} samples from Hugging Face Datasets library.")
            except Exception as e:
                print(f"Error loading dataset {self.fit_dataset} from Hugging Face Datasets library: {e}.\nTrying to load as Parquet file...")
            try:
                self.fit_dataset = pq.read_table(self.fit_dataset).to_pandas()
                print(f"Loaded dataset from Parquet file with {len(self.fit_dataset)} samples.")
            except Exception as e:
                print(f"Error loading dataset from Parquet file {self.fit_dataset}: {e}")
        else:
            raise ValueError(f"Dataset must be a pandas DataFrame, a Hugging Face dataset name, or a path to a Parquet file. Got {type(self.fit_dataset)} instead.")
        
        # For subsampling dataset - mostly for debugging purposes
        if self.cath_sampling_n is not None:
            grouped = self.fit_dataset.groupby("cath_T_id")
            self.fit_dataset = grouped.sample(n=self.cath_sampling_n, replace=True, random_state=42).reset_index(drop=True)
        if self.sample_size is not None:
            self.fit_dataset = self.fit_dataset.sample(n=self.sample_size, random_state=42).reset_index(drop=True)  # Shuffle the dataframe

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self._prepare_funstration_fit_dataset()
            #Masking
            train_mask = self.fit_dataset[self.split_key] == "train"
            val_mask = self.fit_dataset[self.split_key] == "val"
            test_mask = self.fit_dataset[self.split_key] == "test"
            print("Created train/val/test masks")

            max_len = self.fit_dataset["full_seq"].str.len().max()
            res_idx_mask = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.bool)

            frst_vals = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.float)
            frst_vals[:] = -100  # Initialize to -100 for no frst value info

            frst_classes = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.long)
            frst_classes[:] = -100  # Initialize to -100 for no frst class info

            res_reg_means = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.float)
            res_reg_stds = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.float)
            res_cls_majority_classes = torch.zeros((len(self.fit_dataset), max_len), dtype=torch.long)

            print("Initialized res_idx_mask and frst_vals tensors")
            
            for i, idx_list in enumerate(self.fit_dataset["res_idx"]):
                res_idx_mask[i, idx_list] = True
                frst_vals[i, idx_list] = torch.Tensor(self.fit_dataset["frst_idx"][i])
                frst_classes[i, idx_list] = torch.LongTensor(self.fit_dataset["frst_class"][i])
                res_reg_means[i, idx_list] = torch.Tensor(self.fit_dataset["res_reg_means"][i])
                res_reg_stds[i, idx_list] = torch.Tensor(self.fit_dataset["res_reg_stds"][i])
                res_cls_majority_classes[i, idx_list] = torch.LongTensor(self.fit_dataset["res_cls_majority_classes"][i])

            res_idx_mask = res_idx_mask[:, :self.max_seq_length]
            frst_vals = frst_vals[:, :self.max_seq_length]
            frst_classes = frst_classes[:, :self.max_seq_length]
            res_reg_means = res_reg_means[:, :self.max_seq_length]
            res_reg_stds = res_reg_stds[:, :self.max_seq_length]
            res_cls_majority_classes = res_cls_majority_classes[:, :self.max_seq_length]
            print("Populated res_idx_mask and frst_vals tensors")

            self.train_dataset = FunstrationDataset(self.config,
                                                    self.fit_dataset[train_mask]["full_seq"].tolist(),
                                                    res_idx_mask[train_mask],
                                                    frst_vals[train_mask],
                                                    frst_classes[train_mask],
                                                    res_reg_means[train_mask],
                                                    res_reg_stds[train_mask],
                                                    res_cls_majority_classes[train_mask])
            print("Created train dataset")
            self.val_dataset = FunstrationDataset(self.config,
                                                  self.fit_dataset[val_mask]["full_seq"].tolist(),
                                                  res_idx_mask[val_mask],
                                                  frst_vals[val_mask],
                                                  frst_classes[val_mask],
                                                  res_reg_means[val_mask],
                                                  res_reg_stds[val_mask],
                                                  res_cls_majority_classes[val_mask])
            print("Created val dataset")
            self.test_dataset = FunstrationDataset(self.config,
                                                   self.fit_dataset[test_mask]["full_seq"].tolist(),
                                                   res_idx_mask[test_mask],
                                                   frst_vals[test_mask],
                                                   frst_classes[test_mask],
                                                   res_reg_means[test_mask],
                                                   res_reg_stds[test_mask],
                                                   res_cls_majority_classes[test_mask])
            print("Created test dataset")
            print(f"Train/Val/Test split: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)} samples")
        
        if stage == "predict":
            assert self.inference_dataset is not None, "inference_dataset must be provided for prediction stage."
            assert self.inference_dataset["sequence"].notnull().all(), "All sequences must be non-null for prediction."
            assert self.inference_dataset["id"].notnull().all(), "All ids must be non-null for prediction."

            self.predict_dataset = InferenceDataset(self.config,
                                                    self.inference_dataset["id"].tolist(),
                                                    self.inference_dataset["sequence"].tolist())
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
