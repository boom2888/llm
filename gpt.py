import torch
import torch.nn as nn
from config import config
from layernorm import LayerNorm
from  transformer import Transformer
EOT_TOKEN_ID=50256
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
    @torch.no_grad
    def generate(self, idx, max_new_tokens=250, context_size=128, 
                         temperature=0.0, top_k=None, eos_id=EOT_TOKEN_ID):
                """
                Generate text with early stopping at EOT token.
                Stops immediately when EOT token is generated.
                """
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -context_size:]
                    
                    with torch.no_grad():
                        logits = self(idx_cond)
                    
                    logits = logits[:, -1, :]
                    
                    # Top-k filtering
                    if top_k is not None:
                        top_logits, _ = torch.topk(logits, top_k)
                        min_val = top_logits[:, -1]
                        logits = torch.where(
                            logits < min_val,
                            torch.tensor(float("-inf")).to(logits.device),
                            logits
                        )
                    
                    # Temperature sampling
                    if temperature > 0.0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    else:
                        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # STOP AT EOT TOKEN
                    if eos_id is not None and idx_next.item() == eos_id:
                        idx = torch.cat((idx, idx_next), dim=1)
                        break
                    
                    idx = torch.cat((idx, idx_next), dim=1)
                
                return idx

        
