from layers import *
import torch.nn as nn
import torch
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
    def forward(self,x):
        #shortcut for attenttoin block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut
        return x
