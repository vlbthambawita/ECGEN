from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock1D(nn.Module):
    """1D Residual block with group normalization"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual
