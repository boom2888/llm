import torch
import torch.nn as nn
from config import config
from layernorm import LayerNorm
from transformer import Transformer
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.pos_emb=nn.Embedding(config["context_length"],config["emb_dim"])
        self.dropout=nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[Transformer() for _ in range(config["n_layers"])])
        self.finalnorm=LayerNorm(config["emb_dim"])
        self.out=nn.Linear(config["emb_dim"],config["vocab_size"],bias=False)
    def forward(self,x):
        batch_size, seq_len = x.shape
        x1=self.tok_emb(x)
        x2=self.pos_emb(torch.arange(seq_len, device=x.device))
        x=x1+x2
        x=self.dropout(x)
        x=self.trf_blocks(x)
        x= self.finalnorm(x)
        x=self.out(x)
        return x

        
