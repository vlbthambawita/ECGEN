"""Reconstruction quality metrics."""

import torch
from torch import Tensor


def reconstruction_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error for reconstruction quality."""
    return torch.mean((pred - target) ** 2)


def reconstruction_mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error for reconstruction quality."""
    return torch.mean(torch.abs(pred - target))
