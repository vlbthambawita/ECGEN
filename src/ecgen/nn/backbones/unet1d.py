"""
1D U-Net denoiser / velocity network (ADM-style) for ECGs.

``forward(x, t, cond)`` takes the noisy signal ``[B, C, T]``, the time ``[B]``
(any float scale; it is embedded sinusoidally) and an optional
:class:`~ecgen.types.ConditionOutput`:

- ``cond.vector`` is added to the time embedding (``AddToTimeEmb``), which
  modulates every residual block through AdaGN;
- ``cond.tokens`` are attended to by cross-attention at the attention levels;
- ``cond.dense`` is concatenated to the input.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.embeddings import TimestepEmbedding
from ecgen.nn.injection import AdaGN, AddToTimeEmb, CrossAttention1d, concat_dense
from ecgen.registry import BACKBONES
from ecgen.types import ConditionOutput


def _zero(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ResBlock(nn.Module):
    """GN-SiLU-Conv, AdaGN(time+condition)-SiLU-Dropout-Conv(zero init), residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.0,
        groups: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(emb_dim, out_ch, groups)
        self.drop = nn.Dropout(dropout)
        self.conv2 = _zero(nn.Conv1d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h, emb))))
        return self.skip(x) + h


class SelfAttention1d(nn.Module):
    def __init__(self, channels: int, heads: int = 4, groups: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} not divisible by heads={heads}")
        self.heads = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.out = _zero(nn.Conv1d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, t = x.shape
        q, k, v = (
            self.qkv(self.norm(x)).view(b, 3, self.heads, c // self.heads, t).unbind(1)
        )
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.out(out.transpose(-1, -2).reshape(b, c, t))


class _Level(nn.Module):
    """Residual blocks (+ optional self/cross attention after each) at one resolution."""

    def __init__(self, chans, emb_dim, dropout, attn, heads, context_dim):
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock(i, o, emb_dim, dropout) for i, o in chans)
        out = [o for _, o in chans]
        self.attn = nn.ModuleList(
            SelfAttention1d(o, heads) if attn else nn.Identity() for o in out
        )
        self.cross = nn.ModuleList(
            CrossAttention1d(o, context_dim, heads)
            if (attn and context_dim)
            else nn.Identity()
            for o in out
        )


@BACKBONES.register("unet1d")
class UNet1d(nn.Module):
    """
    Args:
        in_channels: Signal channels (leads).
        out_channels: Output channels (default ``in_channels``).
        base_channels: Width of the first level.
        channel_mults: Per-level width multipliers; the signal length must be
            divisible by ``2 ** (len(channel_mults) - 1)``.
        num_res_blocks: Residual blocks per level.
        attn_levels: Levels (0 = full resolution) with self-attention (and
            cross-attention when ``context_dim`` > 0).
        heads: Attention heads.
        dropout: Dropout inside residual blocks.
        cond_dim: Size of ``ConditionOutput.vector`` (0 = none).
        context_dim: Size of ``ConditionOutput.tokens`` (0 = none).
        dense_channels: Channels of ``ConditionOutput.dense`` (0 = none).
    """

    accepts = frozenset({"vector", "tokens", "dense"})

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_levels: Sequence[int] = (3,),
        heads: int = 4,
        dropout: float = 0.0,
        cond_dim: int = 0,
        context_dim: int = 0,
        dense_channels: int = 0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.levels = len(channel_mults)
        self.downsample_factor = 2 ** (self.levels - 1)
        self.time_embed = TimestepEmbedding(base_channels)
        emb_dim = self.time_embed.out_dim
        self.cond_embed = AddToTimeEmb(cond_dim, emb_dim) if cond_dim else None
        self.conv_in = nn.Conv1d(
            in_channels + dense_channels, base_channels, 3, padding=1
        )

        widths = [base_channels * m for m in channel_mults]
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips = [base_channels]
        ch = base_channels
        for lvl, w in enumerate(widths):
            chans = []
            for _ in range(num_res_blocks):
                chans.append((ch, w))
                ch = w
                skips.append(ch)
            self.down.append(
                _Level(chans, emb_dim, dropout, lvl in attn_levels, heads, context_dim)
            )
            if lvl < self.levels - 1:
                self.downsample.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)

        self.mid = _Level(
            [(ch, ch), (ch, ch)], emb_dim, dropout, True, heads, context_dim
        )

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for lvl, w in reversed(list(enumerate(widths))):
            chans = []
            for _ in range(num_res_blocks + 1):
                chans.append((ch + skips.pop(), w))
                ch = w
            self.up.append(
                _Level(chans, emb_dim, dropout, lvl in attn_levels, heads, context_dim)
            )
            if lvl > 0:
                self.upsample.append(nn.Conv1d(ch, ch, 3, padding=1))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = _zero(nn.Conv1d(ch, out_channels, 3, padding=1))

    @staticmethod
    def _run(level: _Level, i: int, h: Tensor, emb: Tensor, tokens, mask) -> Tensor:
        h = level.blocks[i](h, emb)
        h = level.attn[i](h)
        if isinstance(level.cross[i], CrossAttention1d):
            h = level.cross[i](h, tokens, mask)
        return h

    def forward(
        self, x: Tensor, t: Tensor, cond: Optional[ConditionOutput] = None
    ) -> Tensor:
        if x.shape[-1] % self.downsample_factor:
            raise ValueError(
                f"Signal length {x.shape[-1]} must be divisible by {self.downsample_factor} "
                f"(2 ** (len(channel_mults) - 1))"
            )
        emb = self.time_embed(t)
        tokens = mask = None
        if cond is not None:
            if self.cond_embed is not None:
                emb = self.cond_embed(emb, cond.vector)
            tokens, mask = cond.tokens, cond.tokens_mask
            x = concat_dense(x, cond.dense)

        h = self.conv_in(x)
        hs = [h]
        for lvl, level in enumerate(self.down):
            for i in range(len(level.blocks)):
                h = self._run(level, i, h, emb, tokens, mask)
                hs.append(h)
            if lvl < self.levels - 1:
                h = self.downsample[lvl](h)
                hs.append(h)

        for i in range(len(self.mid.blocks)):
            h = self._run(self.mid, i, h, emb, tokens, mask)

        for j, level in enumerate(self.up):
            for i in range(len(level.blocks)):
                h = self._run(
                    level, i, torch.cat([h, hs.pop()], dim=1), emb, tokens, mask
                )
            if j < self.levels - 1:
                h = self.upsample[j](F.interpolate(h, scale_factor=2.0, mode="nearest"))

        return self.conv_out(F.silu(self.norm_out(h)))
