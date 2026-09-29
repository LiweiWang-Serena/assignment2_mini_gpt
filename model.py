"""Mini-GPT: a small decoder-only transformer language model.

Architecture (pre-LayerNorm GPT style):

    token ids -> token embedding + learned positional embedding -> dropout
              -> N x [LayerNorm -> causal multi-head self-attention -> residual
                      LayerNorm -> MLP (Linear -> GELU -> Linear)  -> residual]
              -> final LayerNorm -> linear head -> next-token logits

The output head shares its weights with the token embedding ("weight tying"),
which removes ~vocab_size * n_embd parameters from the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters that define the model architecture."""

    vocab_size: int = 50257  # GPT-2 BPE vocabulary used in Assignment 1
    block_size: int = 128  # maximum context length (= training sequence length)
    n_layer: int = 2  # number of transformer blocks (assignment: 1-2)
    n_head: int = 4  # attention heads per block (assignment: 2-4)
    n_embd: int = 128  # embedding dimension (assignment: 64-256)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-back-only) mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One linear layer produces queries, keys and values for all heads.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask: position t may only attend to positions <= t.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape

        q, k, v = self.qkv(x).split(n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        out = weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)  # merge heads
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x hidden expansion."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only transformer that outputs logits for next-token prediction."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # learned positions
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count parameters (the tied embedding/head weight is counted once)."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.token_emb.weight.numel() + self.pos_emb.weight.numel()
        return total

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss). Loss is cross-entropy when targets are given."""
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}."
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, C)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens tokens after the prompt idx."""
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
