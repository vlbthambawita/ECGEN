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


def gaussian_kl(
    mean: Tensor, logvar: Tensor, prior_mean: Tensor, prior_logvar: Tensor
) -> Tensor:
    """Summed ``KL(N(mean, var) || N(prior_mean, prior_var))``."""
    return 0.5 * torch.sum(
        prior_logvar
        - logvar
        + (logvar.exp() + (mean - prior_mean).pow(2)) / prior_logvar.exp()
        - 1
    )


def cvae_loss(
    recon: Tensor,
    x: Tensor,
    mean: Tensor,
    logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
    kl_weight: float = 0.0001,
    recon_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Conditional VAE loss: MSE + ``kl_weight`` * KL to a (learned) prior.

    The KL is normalized like :func:`vae_loss` (by ``B * C * T``). With
    ``recon_mask`` (1 = count), the MSE is averaged over masked entries only.
    """
    if recon_mask is None:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
    else:
        w = recon_mask.to(x.dtype)
        recon_loss = ((recon - x) ** 2 * w).sum() / w.sum().clamp_min(1.0)
    kl = gaussian_kl(mean, logvar, prior_mean, prior_logvar) / (
        x.size(0) * x.size(1) * x.size(2)
    )
    return recon_loss + kl_weight * kl, recon_loss, kl
