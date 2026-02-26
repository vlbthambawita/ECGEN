"""
Metrics for ECG generation evaluation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def wasserstein_distance(d_real: Tensor, d_fake: Tensor) -> Tensor:
    """
    Compute Wasserstein distance estimate from discriminator outputs.
    
    Args:
        d_real: Discriminator output on real samples
        d_fake: Discriminator output on fake samples
    
    Returns:
        Wasserstein distance estimate (higher is better)
    """
    return d_real.mean() - d_fake.mean()


def gradient_penalty_loss(
    discriminator: torch.nn.Module,
    real_data: Tensor,
    fake_data: Tensor,
    lmbda: float = 10.0,
) -> Tensor:
    """
    Compute WGAN-GP gradient penalty.
    
    Args:
        discriminator: Discriminator network
        real_data: Real samples (B, C, L)
        fake_data: Fake samples (B, C, L)
        lmbda: Gradient penalty coefficient
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    disc_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=real_data.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    
    return gradient_penalty
