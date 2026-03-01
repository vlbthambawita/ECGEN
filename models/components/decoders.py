from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.components.blocks import ResidualBlock1D


class Decoder1D(nn.Module):
    """1D Decoder for ECG signals"""

    def __init__(
        self,
        out_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels

        channel_multipliers = tuple(reversed(channel_multipliers))

        ch = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)

        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch

            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
