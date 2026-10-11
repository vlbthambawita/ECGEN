"""Deprecated: use :mod:`ecgen.families.vae` (and :mod:`ecgen.nn` for the blocks)."""

from ecgen.families.vae import VAE1D, VAEConfig, VAELightning, vae_loss
from ecgen.nn import Decoder1D, Encoder1D, ResidualBlock1D

__all__ = [
    "ResidualBlock1D",
    "Encoder1D",
    "Decoder1D",
    "VAE1D",
    "VAEConfig",
    "VAELightning",
    "vae_loss",
]
