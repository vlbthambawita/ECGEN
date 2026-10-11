"""
Field encoders: one condition field → one :class:`ConditionOutput` slot.

Each encoder declares ``slot`` (``"vector"``, ``"tokens"`` or ``"dense"``) and
``out_dim`` (feature size, or channels for dense). ``forward`` gets the field's
value and its missing mask (``None`` if the field has none) and returns:

- vector: ``[B, out_dim]``
- tokens: ``([B, N, out_dim], mask [B, N] bool)``
- dense:  ``[B, out_dim, T]``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ecgen.nn.embeddings import FourierFeatures
from ecgen.registry import CONDITION_ENCODERS


class FieldEncoder(nn.Module):
    slot: str = "vector"
    out_dim: int

    def forward(self, value: Tensor, missing: Optional[Tensor] = None):  # noqa: D401
        raise NotImplementedError


def _mlp(in_dim: int, dim: int, hidden: Optional[int]) -> nn.Sequential:
    hidden = hidden or dim
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))


@CONDITION_ENCODERS.register("categorical")
class CategoricalEncoder(FieldEncoder):
    """
    Single class label ``[B]`` (int) → embedding ``[B, dim]``.

    A missing label (mask ``True``, or value ``< 0``) maps to a learned
    "unknown" embedding.
    """

    def __init__(self, num_classes: int, dim: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = dim
        self.embed = nn.Embedding(num_classes + 1, dim)  # last row = unknown

    def forward(self, value: Tensor, missing: Optional[Tensor] = None) -> Tensor:
        idx = value.long().reshape(-1)
        unknown = idx < 0
        if missing is not None:
            unknown = unknown | missing.reshape(-1).bool()
        idx = torch.where(unknown, torch.full_like(idx, self.num_classes), idx)
        return self.embed(idx)


@CONDITION_ENCODERS.register("multilabel")
class MultiLabelEncoder(FieldEncoder):
    """
    Multi-hot labels ``[B, L]`` → ``[B, dim]`` (linear projection, SSSD-ECG style),
    or with ``as_tokens=True`` one token per *present* label for cross-attention.
    Missing labels (mask ``[B, L]``) are treated as absent.
    """

    def __init__(
        self, num_labels: int, dim: int = 128, as_tokens: bool = False
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.out_dim = dim
        self.as_tokens = as_tokens
        self.slot = "tokens" if as_tokens else "vector"
        if as_tokens:
            self.label_emb = nn.Parameter(torch.randn(num_labels, dim) * 0.02)
        else:
            self.proj = _mlp(num_labels, dim, None)

    def forward(self, value: Tensor, missing: Optional[Tensor] = None):
        y = value.float()
        if missing is not None:
            y = y * (~missing.bool()).float()
        if not self.as_tokens:
            return self.proj(y)
        tokens = self.label_emb[None] * y[..., None]  # [B, L, D]
        return tokens, y > 0.5


@CONDITION_ENCODERS.register("numeric")
class NumericEncoder(FieldEncoder):
    """
    Normalized numeric features ``[B, F]`` with a missing mask → ``[B, dim]``.

    The MLP sees ``(value * observed, missing)``, so missingness is explicit.
    With ``as_tokens=True`` each feature becomes one token
    ``value * w_f + b_f`` (or a learned missing token), for cross-attention.
    """

    def __init__(
        self,
        num_features: int,
        dim: int = 128,
        hidden: Optional[int] = None,
        as_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.out_dim = dim
        self.as_tokens = as_tokens
        self.slot = "tokens" if as_tokens else "vector"
        if as_tokens:
            self.weight = nn.Parameter(torch.randn(num_features, dim) * 0.02)
            self.bias = nn.Parameter(torch.randn(num_features, dim) * 0.02)
            self.missing_token = nn.Parameter(torch.randn(num_features, dim) * 0.02)
        else:
            self.mlp = _mlp(2 * num_features, dim, hidden)

    def forward(self, value: Tensor, missing: Optional[Tensor] = None):
        x = value.float()
        m = torch.zeros_like(x, dtype=torch.bool) if missing is None else missing.bool()
        x = torch.where(m, torch.zeros_like(x), x)
        if not self.as_tokens:
            return self.mlp(torch.cat([x, m.float()], dim=-1))
        tokens = x[..., None] * self.weight + self.bias
        tokens = torch.where(m[..., None], self.missing_token.expand_as(tokens), tokens)
        return tokens, torch.ones_like(m)


@CONDITION_ENCODERS.register("fourier")
class FourierScalarEncoder(FieldEncoder):
    """Numeric features ``[B, F]`` → random Fourier features → MLP ``[B, dim]``."""

    def __init__(
        self, num_features: int, dim: int = 128, n_freqs: int = 16, sigma: float = 1.0
    ) -> None:
        super().__init__()
        self.out_dim = dim
        self.fourier = FourierFeatures(num_features, n_freqs, sigma)
        self.mlp = _mlp(self.fourier.out_dim + num_features, dim, None)

    def forward(self, value: Tensor, missing: Optional[Tensor] = None) -> Tensor:
        x = value.float()
        m = torch.zeros_like(x, dtype=torch.bool) if missing is None else missing.bool()
        x = torch.where(m, torch.zeros_like(x), x)
        feats = self.fourier(x) * (~m).float().repeat_interleave(
            self.fourier.out_dim // x.shape[1], dim=1
        )
        return self.mlp(torch.cat([feats, m.float()], dim=-1))


@CONDITION_ENCODERS.register("masked_signal")
class MaskedSignalEncoder(FieldEncoder):
    """
    Partially observed ECG → dense ``[B, 2C, T]`` = ``[x * mask, mask]``.

    Reads ``obs`` / ``obs_mask`` (not a ``cond`` field). With ``channels`` set,
    a 1x1 conv projects to that many channels.
    """

    slot = "dense"

    def __init__(self, num_leads: int, channels: Optional[int] = None) -> None:
        super().__init__()
        self.num_leads = num_leads
        self.proj = nn.Conv1d(2 * num_leads, channels, 1) if channels else None
        self.out_dim = channels or 2 * num_leads

    def forward(self, value: Tensor, missing: Optional[Tensor] = None) -> Tensor:
        obs, mask = value
        mask = mask.to(obs.dtype)
        dense = torch.cat([obs * mask, mask], dim=1)
        return self.proj(dense) if self.proj is not None else dense
