import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import T5Tokenizer, T5EncoderModel

class FrustrationFNN(pl.LightningModule):
    def __init__(self, input_dim=1024, hidden_dim=32, output_dim=1, dropout=0.15, max_seq_length=500, pLM_model="Rostlab/ProstT5", pLM_precision="full", prefix_prostT5="<AA2fold>"):
        super(FrustrationFNN, self).__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(pLM_model, do_lower_case=False, max_length=max_seq_length)
        self.encoder = T5EncoderModel.from_pretrained(pLM_model)
        self.prefix_prostT5 = prefix_prostT5

        if pLM_precision == "half":
            self.encoder = self.encoder.half()
        self.encoder.eval()  # Freeze the encoder

        self.max_seq_length = 2753 + 2 #for aa2fold + eos token. later trunc longest first strat and use arg for that

        self.FNN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, full_seq):
        full_seq = [self.prefix_prostT5 + " " + " ".join(seq) for seq in full_seq]  # Add spaces between amino acids and prefix

        ids = self.tokenizer.batch_encode_plus(full_seq, 
                                               add_special_tokens=True, 
                                               padding="max_length", 
                                               max_length=self.max_seq_length,
                                               return_tensors='pt'
                                               ).to(self.device)
        with torch.no_grad():
            embedding_rpr = self.encoder(
                ids.input_ids, 
                attention_mask=ids.attention_mask
            )
        embeddings = embedding_rpr.last_hidden_state[:, 1:-1] # remove the aa token bos
        #embeddings = torch.gather(embeddings, dim=1, index=res_idx)
        return self.FNN(embeddings)
    
    def training_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals = batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds[res_mask], frst_vals[res_mask])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals = batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds[res_mask], frst_vals[res_mask])
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        full_seq, res_mask, frst_vals = batch
        preds = self.forward(full_seq)
        preds = preds.squeeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds[res_mask], frst_vals[res_mask])
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.FNN.parameters(), lr=1e-3)
        return optimizer
    
    @staticmethod
    def suggest_params():
        #TODO model selection
        pass


