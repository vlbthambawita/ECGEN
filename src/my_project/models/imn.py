from typing import Any

import torch
from torch import nn


class IMN(nn.Module):
    """
    Stub ECG model (e.g., Inception-style network).
    """

    def __init__(self, input_channels: int = 12, num_classes: int = 1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> Any:
        features = self.backbone(x).squeeze(-1)
        return self.head(features)
