from torch.utils.data import Dataset
from transformers import T5Tokenizer

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def map_nonstandard_residues(ids, sequences):
    """Uppercase sequences and map every non-standard residue to X, warning about each affected sequence."""
    mapped_seqs = []
    for seq_id, seq in zip(ids, sequences):
        seq = seq.upper()
        mapped = "".join(aa if aa in STANDARD_AAS else "X" for aa in seq)
        nonstandard = sorted({aa for aa in seq if aa not in STANDARD_AAS and aa != "X"})
        if nonstandard:
            n = sum(aa in nonstandard for aa in seq)
            print(f"WARNING: {seq_id}: mapped {n} non-standard residue(s) {nonstandard} to X.")
        mapped_seqs.append(mapped)
    return mapped_seqs



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
    
    def __getitem__(self, idx):
        return self.full_seq[idx], self.id[idx]

    def collate_fn(self, batch):
        seqs, ids = zip(*batch)
        spaced = [" ".join(seq) for seq in seqs]
        tokenized = self.tokenizer(
            list(spaced),
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"], list(seqs), list(ids)