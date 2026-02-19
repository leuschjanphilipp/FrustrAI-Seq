from torch.utils.data import Dataset
from transformers import T5Tokenizer


class FunstrationDataset(Dataset):
    def __init__(self,
                 config, 
                 full_seq, 
                 res_idx, 
                 frst_vals, 
                 frst_classes,
                 res_reg_means, 
                 res_reg_stds, 
                 res_cls_majority_classes):
        
        self.config = config
        self.full_seq = full_seq
        self.res_idx = res_idx
        self.frst_vals = frst_vals
        self.frst_classes = frst_classes
        self.res_reg_means = res_reg_means
        self.res_reg_stds = res_reg_stds
        self.res_cls_majority_classes = res_cls_majority_classes

        self.tokenizer = T5Tokenizer.from_pretrained(config["pLM_model"], 
                                                     do_lower_case=False, 
                                                     max_length=config["max_seq_length"])

    def __len__(self):
        return len(self.full_seq)
    
    def _tokenize_seqs(self, seqs):

        full_seq_idx = [" ".join(seq) for seq in seqs]
        
        # Use tokenizer's __call__ method (batch_encode_plus was removed in transformers v5.x)
        seqs_tokenized = self.tokenizer(full_seq_idx, 
                                        add_special_tokens=True, 
                                        max_length=self.config["max_seq_length"],
                                        padding="max_length",
                                        truncation="longest_first",
                                        return_tensors='pt')
        return seqs_tokenized

    def __getitem__(self, idx):

        seqs_tokenized = self._tokenize_seqs([self.full_seq[idx]])

        return (seqs_tokenized["input_ids"].squeeze(0), 
                seqs_tokenized["attention_mask"].squeeze(0), 
                self.res_idx[idx], 
                self.frst_vals[idx], 
                self.frst_classes[idx], 
                self.res_reg_means[idx], 
                self.res_reg_stds[idx], 
                self.res_cls_majority_classes[idx])

class InferenceDataset(Dataset):
    def __init__(self, 
                 config,
                 id, 
                 full_seq,):

        self.config = config
        self.id = id
        self.full_seq = full_seq

        self.tokenizer = T5Tokenizer.from_pretrained(config["pLM_model"], 
                                                     do_lower_case=False, 
                                                     max_length=config["max_seq_length"])

    def __len__(self):
        return len(self.full_seq)
    
    def _tokenize_seqs(self, seqs):
        # when in inference mode dont apply max length restriction and pad to longest in batch
        max_seq_length = max([len(seq) for seq in seqs])
        full_seq_idx = [" ".join(seq) for seq in seqs]
        
        # Use tokenizer's __call__ method (batch_encode_plus was removed in transformers v5.x)
        seqs_tokenized = self.tokenizer(full_seq_idx, 
                                        add_special_tokens=True, 
                                        max_length=max_seq_length,
                                        padding="max_length",
                                        truncation="longest_first",
                                        return_tensors='pt')
        return seqs_tokenized

    def __getitem__(self, idx):

        seqs_tokenized = self._tokenize_seqs([self.full_seq[idx]])

        return (seqs_tokenized["input_ids"].squeeze(0), 
                seqs_tokenized["attention_mask"].squeeze(0),
                self.full_seq[idx],
                self.id[idx])