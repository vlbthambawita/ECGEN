"""
Layers that inject a condition into a backbone.

All modulation layers are zero-initialised, so a freshly built conditional
network computes exactly what its unconditional counterpart would; the
condition's influence is learned from there.

Shapes: signals/feature maps are ``[B, C, T]``, global conditions ``[B, D]``,
token conditions ``[B, N, D]``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def concat_dense(x: Tensor, dense: Optional[Tensor]) -> Tensor:
    """``ConcatInput``: stack time-aligned conditions onto the input channels."""
    if dense is None:
        return x
    if dense.shape[-1] != x.shape[-1]:
        dense = F.interpolate(
            dense, size=x.shape[-1], mode="linear", align_corners=False
        )
    return torch.cat([x, dense], dim=1)


class ConcatInput(nn.Module):
    """Module form of :func:`concat_dense`; ``out_channels = in_channels + cond_channels``."""

    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.out_channels = in_channels + cond_channels

    def forward(self, x: Tensor, dense: Optional[Tensor]) -> Tensor:
        return concat_dense(x, dense)


class FiLM(nn.Module):
    """``h * (1 + gamma(c)) + beta(c)`` per channel."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = _zero(nn.Linear(cond_dim, 2 * channels))

    def forward(self, h: Tensor, c: Optional[Tensor]) -> Tensor:
        if c is None:
            return h
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        shape = (h.shape[0], -1) + (1,) * (h.dim() - 2)
        return h * (1 + gamma.view(shape)) + beta.view(shape)


class AdaGN(nn.Module):
    """GroupNorm without affine parameters, then FiLM from the condition."""

    def __init__(self, cond_dim: int, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, affine=True)
        self.film = FiLM(cond_dim, channels)

    def forward(self, h: Tensor, c: Optional[Tensor]) -> Tensor:
        return self.film(self.norm(h), c)


class AdaLNZero(nn.Module):
    """
    DiT-style adaptive LayerNorm for ``[B, N, D]`` tokens.

    ``forward`` returns ``(modulated_x, gate)``; the block adds
    ``gate * f(modulated_x)`` to its residual. Gates start at 0.
    """

    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = _zero(nn.Linear(cond_dim, 3 * dim))

    def forward(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale, gate = self.proj(F.silu(c)).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class AddToTimeEmb(nn.Module):
    """Project a global condition and add it to the time embedding."""

    def __init__(self, cond_dim: int, temb_dim: int) -> None:
        super().__init__()
        self.proj = _zero(nn.Linear(cond_dim, temb_dim))

    def forward(self, temb: Tensor, c: Optional[Tensor]) -> Tensor:
        return temb if c is None else temb + self.proj(c)


class CrossAttention1d(nn.Module):
    """
    Feature map ``[B, C, T]`` attends to condition tokens ``[B, N, D]``.

    Residual: ``x + out(attn(norm(x), tokens))``, with ``out`` zero-initialised.
    """

    def __init__(
        self, channels: int, context_dim: int, heads: int = 4, groups: int = 8
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(context_dim, 2 * channels)
        self.out = _zero(nn.Linear(channels, channels))

    def forward(
        self, x: Tensor, tokens: Optional[Tensor], mask: Optional[Tensor] = None
    ) -> Tensor:
        if tokens is None:
            return x
        b, c, t = x.shape
        h = self.norm(x).transpose(1, 2)  # [B, T, C]
        q = self.q(h).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = self.kv(tokens).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(torch.bool)  # True = attend
            # rows with no valid token would give NaN; let them attend to everything
            attn_mask = attn_mask | ~attn_mask.any(dim=-1, keepdim=True)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, c)
        return x + self.out(out).transpose(1, 2)
