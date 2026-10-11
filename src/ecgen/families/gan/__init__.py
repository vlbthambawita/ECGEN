"""Generative Adversarial Network (GAN) family."""

from ecgen.families.gan.pulse2pulse import (
    Pulse2PulseConfig,
    Pulse2PulseGAN,
    WaveGANDiscriminator,
    WaveGANGenerator,
    calc_gradient_penalty,
)

__all__ = [
    "Pulse2PulseConfig",
    "Pulse2PulseGAN",
    "WaveGANDiscriminator",
    "WaveGANGenerator",
    "calc_gradient_penalty",
]
