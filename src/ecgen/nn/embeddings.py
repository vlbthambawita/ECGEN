"""
Embeddings for diffusion time steps and continuous scalars.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """
    Transformer-style sinusoidal embedding of (possibly fractional) positions.

    Args:
        t: ``[B]`` time steps or noise levels.
        dim: Embedding size.

    Returns:
        ``[B, dim]``
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP (DDPM / ADM style)."""

    def __init__(self, dim: int, hidden: int | None = None, scale: float = 1.0) -> None:
        super().__init__()
        hidden = hidden or 4 * dim
        self.dim = dim
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.out_dim = hidden

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(sinusoidal_embedding(t * self.scale, self.dim))


class FourierFeatures(nn.Module):
    """
    Random Fourier features of scalars: ``[B, F] -> [B, F * 2 * n_freqs]``.

    Frequencies are fixed (a buffer), drawn once with a fixed seed so models
    built twice from the same config are identical.
    """

    def __init__(
        self, num_features: int, n_freqs: int = 16, sigma: float = 1.0
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.register_buffer(
            "freqs", torch.randn(num_features, n_freqs, generator=gen) * sigma
        )
        self.out_dim = num_features * 2 * n_freqs

    def forward(self, x: Tensor) -> Tensor:
        args = 2 * math.pi * x[..., None] * self.freqs  # [B, F, n]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).flatten(1)
