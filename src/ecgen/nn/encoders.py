from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.blocks import ResidualBlock1D
from ecgen.nn.injection import FiLM


class Encoder1D(nn.Module):
    """1D Encoder for ECG signals"""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 0,
        extra_in_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # extra_in_channels: dense conditions concatenated to the input;
        # cond_dim > 0: FiLM after every residual block (both default off, so
        # unconditional models keep their original parameters)
        self.conv_in = nn.Conv1d(
            in_channels + extra_in_channels, base_channels, kernel_size=7, padding=3
        )

        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch

            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(
                    nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)
                )

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, latent_channels * 2, kernel_size=3, padding=1)

        self.films = nn.ModuleDict()
        if cond_dim > 0:
            for i, block in enumerate(self.down_blocks):
                if isinstance(block, ResidualBlock1D):
                    self.films[str(i)] = FiLM(cond_dim, block.conv2.out_channels)
            self.films["mid"] = FiLM(cond_dim, ch)

    def forward(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if dense is not None:
            x = torch.cat([x, dense], dim=1)
        h = self.conv_in(x)

        for i, block in enumerate(self.down_blocks):
            h = block(h)
            if str(i) in self.films:
                h = self.films[str(i)](h, c)

        h = self.mid_block1(h)
        h = self.mid_block2(h)
        if "mid" in self.films:
            h = self.films["mid"](h, c)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)

        return mean, logvar
