"""Generation quality metrics."""

import torch
from torch import Tensor


def signal_to_noise_ratio(signal: Tensor, noise: Tensor) -> Tensor:
    """Calculate signal-to-noise ratio."""
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-8))
