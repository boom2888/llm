import torch 
import torch.nn as nn
import math
from torch.nn import functional as F
class CasualAttention(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query=nn.Linear(d_in,d_out)
        self.W_key=nn.Linear(d_in,d_out)
        self.w_value=nn.Linear(d_in,d_out)
    def forward(self,x,mask=True):
        q=self.W_query(x)
        k=self.W_key(x)
        
        v=self.w_value(x)
        attn=q @ k.transpose(1,2)
        d_k = q.size(-1)
        if mask:
            mask=torch.ones_like(attn,dtype=torch.bool).triu(1)
            attn.masked_fill_(mask,-torch.inf)
        attn /= math.sqrt(d_k) 
        attn_scores=F.softmax(attn,dim=-1)
        return attn_scores @v

class Multiheadattn(nn.Module):
    def __init__(self,heads,d_in,d_out):
        super().__init__()
        self.d_out=d_out//heads
        self.models=nn.ModuleList(
            [ CasualAttention(d_in,self.d_out) for _ in range(heads)]
        )
    def forward(self,x):
       return torch.cat([head(x) for head in self.models],dim=-1)

        
