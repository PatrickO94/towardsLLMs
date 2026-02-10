import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .config import cfg

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads, n_embd, decode=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.decode = decode
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = self.n_embd // num_heads

        assert self.head_dim * num_heads == self.n_embd, \
            "n_embd must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query = nn.Linear(n_embd, self.n_embd)
        self.key = nn.Linear(n_embd, self.n_embd)
        self.value = nn.Linear(n_embd, self.n_embd)

        # Output projection
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        self.dropout_att = nn.Dropout(cfg.DROPOUT)
        self.dropout_res = nn.Dropout(cfg.DROPOUT)
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)
        # buffer mask
        if decode:
            self.register_buffer("bias", torch.tril(torch.ones(cfg.CONTEXT_LEN, cfg.CONTEXT_LEN)).view(1, 1, cfg.CONTEXT_LEN, cfg.CONTEXT_LEN))


    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len) - optional attention mask
        """
        B, T, C = x.shape

        # Linear projections
        q = self.query(x)  # (B, T, n_embd = C)
        k = self.key(x)  # (B, T, n_embd = C)
        v = self.value(x)  # (B, T, n_embd = C)

        # Split into heads and reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, headD)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, headD)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, headD)

        # Scaled dot-product attention
        att_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # causal mask, preventing to attend to future tokens. masking beneath diagonal (lower triangle).
        if self.decode:
            att_scores = att_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # softmax, last step of attention calculations


        att_scores = F.softmax(att_scores, dim=-1)  # (B, H, T, T)
        att_scores = self.dropout_att(att_scores)

        # Apply attention to values
        out = att_scores @ v  # (B, H, T, D)

        # Concatenate heads: -> (B, T, H, D) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)


        # Final linear projection
        out = self.out_proj(out)
        out = self.dropout_res(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd),
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd),
                                 nn.Dropout(cfg.DROPOUT),)
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(cfg.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(cfg.N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.CONTEXT_LEN, cfg.CONTEXT_LEN)))
        self.dropout = nn.Dropout(cfg.DROPOUT)

    def forward(self, x, decoder=True):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C-headsize)
        q = self.query(x) # (B, T, C-headsize)
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        if decoder:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask influence of future nodes/tokens
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x) # (B, T, C)
        out = weights @ v # (B, T, T) @ (B, T, C-head)---> (B, T, C-head)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(cfg.N_EMBD, cfg.N_EMBD)
        self.dropout = nn.Dropout(cfg.DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BlockNew(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa = MultiHeadAttentionLayer(n_heads, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

