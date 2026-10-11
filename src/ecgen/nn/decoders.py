from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.blocks import ResidualBlock1D
from ecgen.nn.injection import FiLM


class Decoder1D(nn.Module):
    """1D Decoder for ECG signals"""

    def __init__(
        self,
        out_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 0,
        extra_in_channels: int = 0,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels

        channel_multipliers = tuple(reversed(channel_multipliers))

        ch = base_channels * channel_multipliers[0]
        # extra_in_channels: dense conditions (at latent resolution) concatenated
        # to z; cond_dim > 0: FiLM after the mid blocks and every residual block
        self.conv_in = nn.Conv1d(
            latent_channels + extra_in_channels, ch, kernel_size=3, padding=1
        )

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)

        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch

            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)
                )

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

        self.films = nn.ModuleDict()
        if cond_dim > 0:
            self.films["mid"] = FiLM(cond_dim, base_channels * channel_multipliers[0])
            for i, block in enumerate(self.up_blocks):
                if isinstance(block, ResidualBlock1D):
                    self.films[str(i)] = FiLM(cond_dim, block.conv2.out_channels)

    def forward(
        self, z: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> Tensor:
        if dense is not None:
            z = torch.cat([z, dense], dim=1)
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_block2(h)
        if "mid" in self.films:
            h = self.films["mid"](h, c)

        for i, block in enumerate(self.up_blocks):
            h = block(h)
            if str(i) in self.films:
                h = self.films[str(i)](h, c)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
