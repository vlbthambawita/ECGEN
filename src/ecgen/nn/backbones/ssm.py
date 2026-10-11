"""
State-space backbone (SSSD-style; Alcaraz & Strodthoff, 2023) built from S4D
layers (Gu et al., 2022) in pure PyTorch.

An S4D layer is a diagonal linear state-space model applied per channel as a
long convolution: its kernel ``K[l] = 2 Re(sum_n C_n (exp(dt A_n) - 1) / A_n * exp(dt A_n l))``
is computed once per forward pass and applied with an FFT, so the cost is
``O(L log L)`` and the receptive field is the whole signal. ``bidirectional``
adds a second kernel run over the time-reversed input (a non-causal model, as
denoisers should be).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.embeddings import TimestepEmbedding
from ecgen.nn.injection import AddToTimeEmb, concat_dense
from ecgen.registry import BACKBONES
from ecgen.types import ConditionOutput


class S4D(nn.Module):
    """
    Diagonal state-space layer ``[B, H, L] -> [B, H, L]``.

    Args:
        channels: ``H``.
        state_size: ``N``; ``N / 2`` complex conjugate pairs are kept.
        dt_min, dt_max: Range of the (learned, log-uniform initialised) step sizes.
        bidirectional: Also run a kernel over the reversed sequence.
    """

    def __init__(
        self,
        channels: int,
        state_size: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        n = state_size // 2
        k = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        log_dt = torch.rand(k, channels) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        # S4D-Lin initialisation: A_n = -1/2 + i * pi * n
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(k, channels, n)))
        self.a_imag = nn.Parameter(
            math.pi * torch.arange(n).float().expand(k, channels, n).clone()
        )
        self.c = nn.Parameter(
            torch.view_as_real(
                torch.randn(k, channels, n, dtype=torch.cfloat) * (0.5**0.5)
            )
        )
        self.d = nn.Parameter(torch.randn(channels))

    def kernel(self, length: int) -> Tensor:
        """``[k, H, L]`` convolution kernels."""
        dt = self.log_dt.exp()[..., None]  # [k, H, 1]
        a = -self.log_a_real.exp() + 1j * self.a_imag  # [k, H, N]
        c = torch.view_as_complex(self.c)
        dta = dt * a
        coef = c * (torch.exp(dta) - 1.0) / a  # [k, H, N]
        vander = torch.exp(
            dta[..., None] * torch.arange(length, device=a.device)
        )  # [k, H, N, L]
        return 2 * torch.einsum("khn,khnl->khl", coef, vander).real

    def forward(self, u: Tensor) -> Tensor:
        length = u.shape[-1]
        k = self.kernel(length).to(u.dtype)
        n_fft = 2 * length
        u_f = torch.fft.rfft(u.float(), n=n_fft)
        y = torch.fft.irfft(u_f * torch.fft.rfft(k[0].float(), n=n_fft), n=n_fft)[
            ..., :length
        ]
        if self.bidirectional:
            u_rev = torch.fft.rfft(u.flip(-1).float(), n=n_fft)
            y_rev = torch.fft.irfft(
                u_rev * torch.fft.rfft(k[1].float(), n=n_fft), n=n_fft
            )[..., :length]
            y = y + y_rev.flip(-1)
        return (y + u.float() * self.d[:, None]).to(u.dtype)


class SSMResidualBlock(nn.Module):
    """SSSD residual block: (+ time/condition) -> S4D -> gated activation -> residual + skip."""

    def __init__(
        self,
        channels: int,
        emb_dim: int,
        state_size: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.emb = nn.Linear(emb_dim, channels)
        self.norm = nn.GroupNorm(8, channels)
        self.ssm = S4D(channels, state_size, bidirectional=bidirectional)
        self.mix = nn.Conv1d(channels, 2 * channels, 1)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, Tensor]:
        h = self.norm(x + self.emb(emb)[..., None])
        h = self.mix(self.ssm(h))
        a, g = h.chunk(2, dim=1)
        h = self.drop(torch.tanh(a) * torch.sigmoid(g))
        return (x + self.res(h)) / math.sqrt(2.0), self.skip(h)


@BACKBONES.register("ssm")
class SSMBackbone(nn.Module):
    """
    Stack of S4D residual blocks at full resolution (no down-sampling).

    Args:
        in_channels: Signal (or latent) channels.
        out_channels: Default ``in_channels``.
        channels: Width.
        depth: Residual blocks.
        state_size: S4D state size ``N``.
        bidirectional: Non-causal (two-sided) kernels.
        dropout: Dropout after the gate.
        cond_dim, dense_channels: Condition sizes (vector added to the time
            embedding; dense concatenated to the input). Tokens are not supported.
    """

    accepts = frozenset({"vector", "dense"})

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: Optional[int] = None,
        channels: int = 128,
        depth: int = 8,
        state_size: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.0,
        cond_dim: int = 0,
        context_dim: int = 0,
        dense_channels: int = 0,
    ) -> None:
        super().__init__()
        if context_dim:
            raise ValueError(
                "The ssm backbone does not take token conditions; use unet1d or dit1d"
            )
        out_channels = out_channels or in_channels
        self.time_embed = TimestepEmbedding(channels)
        emb_dim = self.time_embed.out_dim
        self.cond_embed = AddToTimeEmb(cond_dim, emb_dim) if cond_dim else None
        self.conv_in = nn.Conv1d(in_channels + dense_channels, channels, 1)
        self.blocks = nn.ModuleList(
            SSMResidualBlock(channels, emb_dim, state_size, bidirectional, dropout)
            for _ in range(depth)
        )
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv1d(channels, out_channels, 1),
        )
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def forward(
        self, x: Tensor, t: Tensor, cond: Optional[ConditionOutput] = None
    ) -> Tensor:
        emb = self.time_embed(t)
        if cond is not None:
            if self.cond_embed is not None:
                emb = self.cond_embed(emb, cond.vector)
            x = concat_dense(x, cond.dense)
        emb = F.silu(emb)
        h = self.conv_in(x)
        skip = torch.zeros_like(h)
        for block in self.blocks:
            h, s = block(h, emb)
            skip = skip + s
        return self.conv_out(skip / math.sqrt(len(self.blocks)))


class _BiMamba(nn.Module):
    """A Mamba layer over the sequence and another over its reverse, summed."""

    def __init__(self, channels: int, state_size: int) -> None:
        super().__init__()
        from mamba_ssm import Mamba

        self.fwd = Mamba(d_model=channels, d_state=state_size, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=channels, d_state=state_size, d_conv=4, expand=2)

    def forward(self, u: Tensor) -> Tensor:  # [B, H, L]
        x = u.transpose(1, 2)
        return (self.fwd(x) + self.bwd(x.flip(1)).flip(1)).transpose(1, 2)


@BACKBONES.register("mamba")
class MambaBackbone(SSMBackbone):
    """
    :class:`SSMBackbone` with bidirectional Mamba layers (Gu & Dao, 2023) in
    place of S4D. Needs the ``mamba-ssm`` package (CUDA only; ``ecgen[ssm]``).
    """

    def __init__(self, *args, state_size: int = 16, **kwargs) -> None:
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The mamba backbone needs `pip install mamba-ssm` (CUDA)"
            ) from e
        super().__init__(*args, state_size=state_size, **kwargs)
        channels = self.conv_in.out_channels
        for block in self.blocks:
            block.ssm = _BiMamba(channels, state_size)
