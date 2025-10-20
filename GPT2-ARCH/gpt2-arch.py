import torch.nn as nn
import torch
from layers import *
from transformer import *
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
class GPT2(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])

        self.drop = nn.Dropout(cfg["drop_rate"])
        self.transformers = nn.ModuleList([Transformer(cfg=cfg) for _ in range(cfg["n_layers"])])
        self.norm = LayerNorm(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"],cfg["vocab_size"])
    def forward(self,x):
        _ , seq_len = x.shape
        x = self.pos_emb(torch.arange(seq_len)) + self.token_emb(x)
        x = self.drop(x)
        for layer in self.transformers:
            x = layer(x)
        x = self.norm(x)
        return self.out(x)
gpt2 = GPT2(GPT_CONFIG_124M)
inputs = torch.randint(low=0, high=50257, size=(2, 4))  
print(gpt2(inputs).shape)