"""Loss functions for different model types."""

from training.losses.vae_losses import vae_loss
from training.losses.vqvae_losses import vqvae_loss, prior_loss

__all__ = ["vae_loss", "vqvae_loss", "prior_loss"]
