"""Common loss functions shared across model types."""

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss."""
    return F.l1_loss(pred, target)
