from layernorm import LayerNorm
import torch 
import torch.nn as nn
from config import config
from attention import Multiheadattn
from feedfoward import Feddforward
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=LayerNorm(config["emb_dim"])
        self.attn=Multiheadattn(config["n_heads"],config["emb_dim"],config["emb_dim"])
        self.dropout=nn.Dropout(config["drop_rate"])
        self.feedforward=Feddforward()

       
    def forward(self,x):
      temp=x
      x=self.norm(x)
      x=self.attn(x)
      x=self.dropout(x)
      x=temp+x
      temp=x
      x=self.norm(x)
      x=self.feedforward(x)
      x=self.dropout(x)
      x=temp+x
      return x