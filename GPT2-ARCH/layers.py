import torch.nn as nn
import torch
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5
    def forward(self,x):
        means = x.mean(dim=-1,keepdim=True)
        vars = x.var(dim=-1,keepdim=True)
        norm_vals = (x - means)/torch.sqrt(vars+self.eps)
        return self.scale*norm_vals + self.shift*norm_vals
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(cfg["emb_dim"],cfg["emb_dim"]*4),nn.GELU(),nn.Linear(cfg["emb_dim"]*4,cfg["emb_dim"]))
    def forward(self,x):
        return self.feed_forward(x)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
      super().__init__()
      assert (d_out % num_heads == 0), \
          "d_out must be divisible by num_heads"
      self.d_out = d_out
      self.num_heads = num_heads
      self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim
      self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.out_proj = nn.Linear(d_out,d_out)
      self.dropout = nn.Dropout(dropout)
      self.register_buffer(
          "mask",
          torch.triu(torch.ones(context_length,context_length),diagonal=1)
      )
    def forward(self,x):
      b , num_tokens, d_in = x.shape
      queries = self.W_query(x).view(b,num_tokens,self.num_heads,self.head_dim)
      keys = self.W_key(x).view(b,num_tokens,self.num_heads,self.head_dim)
      values = self.W_value(x).view(b,num_tokens,self.num_heads,self.head_dim)
      # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
      keys = keys.transpose(1,2)
      values = values.transpose(1,2)
      queries = queries.transpose(1,2)


      attention_scores = queries @ keys.transpose(-1,-2)
      # Original mask truncated to the number of tokens and converted to boolean
      mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
      attention_scores.masked_fill_(mask_bool, -torch.inf)
      attn_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

      attn_weights = self.dropout(attn_weights)

      # Shape: (b, num_tokens, num_heads, head_dim)
      context_vec = (attn_weights @ values).transpose(1, 2)

      # Combine heads, where self.d_out = self.num_heads * self.head_dim
      context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
      context_vec = self.out_proj(context_vec) # optional projection

      return context_vec
