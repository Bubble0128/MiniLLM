"""Character GPT adapted from angelos-p/llm-from-scratch, tutorial Part 2.

The baseline keeps learned positions, pre-LN blocks, GELU, causal SDPA and
weight tying. Optional dropout is the explicitly labelled model experiment.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    # Course ablations; defaults preserve the original architecture.
    residual_scale: float = 1.0
    weight_tying: bool = True

    def __post_init__(self):
        """Reject invalid model settings before allocating a model."""
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    """Multi-head attention with a causal mask: no future-token leakage."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Project Q/K/V, attend to the prefix, then concatenate the heads."""
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q, k, v = [t.view(batch, length, self.n_head, width // self.n_head)
                   .transpose(1, 2) for t in (q, k, v)]
        # EXPERIMENT: attention and residual dropout; disabled in baseline/eval.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward layer with 4x expansion and GELU."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Transform each token independently; optionally regularize output."""
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Two pre-normalized residual sublayers, following the source tutorial."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.residual_scale = config.residual_scale
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Mix prefix information, then apply the position-wise MLP."""
        x = x + self.residual_scale * self.attn(self.ln_1(x))
        return x + self.residual_scale * self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Autoregressive character model returning logits and optional CE loss."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.transformer.wte.weight = self.lm_head.weight
        self.embedding_dropout = nn.Dropout(config.dropout)

    def forward(self, idx, targets=None):
        """Predict the next character at every input position (B,T,V)."""
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError("Input must have shape (B,T), with 1 <= T <= block_size")
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.embedding_dropout(
            self.transformer.wte(idx) + self.transformer.wpe(positions)
        )
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss
