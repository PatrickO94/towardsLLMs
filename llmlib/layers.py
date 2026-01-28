import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .config import cfg

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None, num_heads=8, dropout=0.1):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_heads = num_heads
        self.head_dim = self.output_dim // num_heads

        assert self.head_dim * num_heads == self.output_dim, \
            "output_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query = nn.Linear(input_dim, self.output_dim)
        self.key = nn.Linear(input_dim, self.output_dim)
        self.value = nn.Linear(input_dim, self.output_dim)

        # Output projection
        self.out_proj = nn.Linear(self.output_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len) - optional attention mask
        """
        B, T, C = x.shape

        # Linear projections
        q = self.query(x)  # (B, T, output_dim)
        k = self.key(x)  # (B, T, output_dim)
        v = self.value(x)  # (B, T, output_dim)

        # Split into heads and reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Scaled dot-product attention
        att_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # masking of diagonal of att scores if used, reshaping if necessary
        if mask is not None:
            # handle different mask shapes
            if mask.dim() == 2:  # (B, T)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            elif mask.dim() == 3:  # (B, 1, T) or (B, T, T)
                mask = mask.unsqueeze(1)
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
        """
        # causal mask, preventing to attend to future tokens. masking beneath diagonal (lower triangle).
        # just pass torch.tril mask to attention layer, to preserve flexibility of the mask as seen above.
        if mask is None:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            mask = causal_mask  # prevents attending to future tokens
        # softmax, last step of attention calculations
        """

        att_weights = F.softmax(att_scores, dim=-1)  # (B, H, T, T)
        att_weights = self.dropout(att_weights)

        # Apply attention to values
        out = att_weights @ v  # (B, H, T, D)

        # Concatenate heads: -> (B, T, H, D) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.output_dim)

        # Final linear projection
        out = self.out_proj(out)

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

