"""
Deprecated since 1.1.0 and removed in 2.0: use :mod:`ecgen.families` (and
:mod:`ecgen.nn` for the building blocks).
"""

import warnings

warnings.warn(
    "ecgen.models is deprecated and will be removed in ecgen 2.0; "
    "use ecgen.families (and ecgen.nn for the building blocks).",
    DeprecationWarning,
    stacklevel=2,
)

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
