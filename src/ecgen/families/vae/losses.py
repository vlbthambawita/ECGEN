from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def vae_loss(
    recon: Tensor,
    x: Tensor,
    mean: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.0001,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    VAE loss function

    Args:
        recon: Reconstructed signal
        x: Original signal
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / (x.size(0) * x.size(1) * x.size(2))

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
