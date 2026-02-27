"""
ECG generation models.
"""

from ecgen.models.pulse2pulse import (
    Pulse2PulseConfig,
    Pulse2PulseGAN,
    WaveGANDiscriminator,
    WaveGANGenerator,
)
from ecgen.models.vae import (
    Decoder1D,
    Encoder1D,
    ResidualBlock1D,
    VAE1D,
    VAEConfig,
    VAELightning,
    vae_loss,
)

__all__ = [
    "Pulse2PulseConfig",
    "Pulse2PulseGAN",
    "WaveGANDiscriminator",
    "WaveGANGenerator",
    "ResidualBlock1D",
    "Encoder1D",
    "Decoder1D",
    "VAE1D",
    "VAEConfig",
    "VAELightning",
    "vae_loss",
]
