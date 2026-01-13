import torch
import torch.nn as nn
from gelu import GELU
from config import config
class Feddforward(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
          
                nn.Linear(config["emb_dim"],4*config["emb_dim"]),
                GELU(),
                nn.Linear(4*config["emb_dim"],config["emb_dim"])
            
        )
    def forward(self,x):
        return self.layer(x)

