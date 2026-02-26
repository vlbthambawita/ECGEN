"""
Loss functions for ECG generation models.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def wgan_discriminator_loss(d_real: Tensor, d_fake: Tensor, gp: Tensor) -> Tensor:
    """
    WGAN-GP discriminator loss.
    
    Args:
        d_real: Discriminator output on real samples
        d_fake: Discriminator output on fake samples
        gp: Gradient penalty
    
    Returns:
        Discriminator loss (minimize: fake - real + gp)
    """
    return d_fake - d_real + gp


def wgan_generator_loss(d_fake: Tensor) -> Tensor:
    """
    WGAN-GP generator loss.
    
    Args:
        d_fake: Discriminator output on fake samples
    
    Returns:
        Generator loss (minimize: -d_fake)
    """
    return -d_fake


def adversarial_loss(pred: Tensor, target_is_real: bool) -> Tensor:
    """
    Standard adversarial loss (BCE).
    
    Args:
        pred: Discriminator predictions
        target_is_real: Whether targets should be real (1) or fake (0)
    
    Returns:
        Binary cross-entropy loss
    """
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)


def least_squares_discriminator_loss(d_real: Tensor, d_fake: Tensor) -> Tensor:
    """
    Least squares GAN discriminator loss.
    
    Args:
        d_real: Discriminator output on real samples
        d_fake: Discriminator output on fake samples
    
    Returns:
        LSGAN discriminator loss
    """
    return 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))


def least_squares_generator_loss(d_fake: Tensor) -> Tensor:
    """
    Least squares GAN generator loss.
    
    Args:
        d_fake: Discriminator output on fake samples
    
    Returns:
        LSGAN generator loss
    """
    return 0.5 * torch.mean((d_fake - 1) ** 2)
