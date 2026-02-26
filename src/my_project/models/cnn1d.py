from typing import Any

import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN baseline for ECG signals.
    """

    def __init__(self, input_channels: int = 12, num_classes: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> Any:
        features = self.net(x).squeeze(-1)
        return self.head(features)
