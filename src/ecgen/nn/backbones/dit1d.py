"""
DiT-style transformer for 1D signals (Peebles & Xie, 2023).

The signal (plus dense conditions) is cut into non-overlapping patches of
``patch_size`` samples, embedded as tokens with fixed sinusoidal positions, and
processed by transformer blocks whose LayerNorms are modulated by the time +
condition embedding (AdaLN-Zero); condition tokens enter through
cross-attention in every block.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.embeddings import TimestepEmbedding
from ecgen.nn.injection import AdaLNZero, concat_dense
from ecgen.registry import BACKBONES
from ecgen.types import ConditionOutput


def _positions(n: int, dim: int, device) -> Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        context_dim: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.ada_attn = AdaLNZero(emb_dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ada_mlp = AdaLNZero(emb_dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.cross = None
        if context_dim:
            self.cross_norm = nn.LayerNorm(dim)
            self.cross = nn.MultiheadAttention(
                dim, heads, kdim=context_dim, vdim=context_dim, batch_first=True
            )
            self.cross_gate = nn.Parameter(torch.zeros(dim))

    def forward(
        self, x: Tensor, emb: Tensor, tokens: Optional[Tensor], mask: Optional[Tensor]
    ) -> Tensor:
        h, gate = self.ada_attn(x, emb)
        x = x + gate * self.attn(h, h, h, need_weights=False)[0]
        if self.cross is not None and tokens is not None:
            key_padding = None
            if mask is not None:
                key_padding = ~mask.bool()
                key_padding = key_padding & ~key_padding.all(
                    dim=1, keepdim=True
                )  # keep >= 1 key
            out = self.cross(
                self.cross_norm(x),
                tokens,
                tokens,
                key_padding_mask=key_padding,
                need_weights=False,
            )[0]
            x = x + self.cross_gate * out
        h, gate = self.ada_mlp(x, emb)
        return x + gate * self.mlp(h)


@BACKBONES.register("dit1d")
class DiT1d(nn.Module):
    """
    Args:
        in_channels: Signal (or latent) channels.
        out_channels: Default ``in_channels``.
        patch_size: Samples per token; the length must be divisible by it.
        dim: Token width.
        depth: Transformer blocks.
        heads: Attention heads.
        mlp_ratio: MLP width / ``dim``.
        cond_dim, context_dim, dense_channels: Condition sizes (see ``UNet1d``).
    """

    accepts = frozenset({"vector", "tokens", "dense"})

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        patch_size: int = 25,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: int = 0,
        context_dim: int = 0,
        dense_channels: int = 0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.dim = dim
        self.patch = nn.Conv1d(
            in_channels + dense_channels, dim, patch_size, stride=patch_size
        )
        self.time_embed = TimestepEmbedding(dim)
        emb_dim = self.time_embed.out_dim
        self.cond_proj = nn.Linear(cond_dim, emb_dim) if cond_dim else None
        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        self.blocks = nn.ModuleList(
            DiTBlock(dim, heads, emb_dim, context_dim, mlp_ratio) for _ in range(depth)
        )
        self.final = AdaLNZero(emb_dim, dim)
        self.out = nn.Linear(dim, patch_size * out_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, x: Tensor, t: Tensor, cond: Optional[ConditionOutput] = None
    ) -> Tensor:
        b, _, length = x.shape
        if length % self.patch_size:
            raise ValueError(
                f"Signal length {length} must be divisible by patch_size={self.patch_size}"
            )
        emb = self.time_embed(t)
        tokens = mask = None
        if cond is not None:
            if self.cond_proj is not None and cond.vector is not None:
                emb = emb + self.cond_proj(cond.vector)
            tokens, mask = cond.tokens, cond.tokens_mask
            x = concat_dense(x, cond.dense)
        emb = F.silu(emb)

        h = self.patch(x).transpose(1, 2)  # [B, N, dim]
        h = h + _positions(h.shape[1], self.dim, h.device)
        for block in self.blocks:
            h = block(h, emb, tokens, mask)
        h, _ = self.final(h, emb)
        out = self.out(h)  # [B, N, patch * C]
        out = out.view(b, -1, self.patch_size, self.out_channels).permute(0, 3, 1, 2)
        return out.reshape(b, self.out_channels, length)
