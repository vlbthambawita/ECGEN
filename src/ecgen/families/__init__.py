"""
Generative-model families.

Importing this package registers the built-in families in
``ecgen.registry.FAMILIES``: ``vae``, ``vqvae``, ``vqvae_prior``, ``pulse2pulse``, ``diffusion``, ``flow_matching``.
"""

from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.families.continuous import (
    ContinuousConfig,
    ContinuousGenerativeModule,
    DiffusionModule,
    FlowMatchingModule,
)
from ecgen.families.gan import Pulse2PulseConfig, Pulse2PulseGAN
from ecgen.families.vae import VAEConfig, VAELightning
from ecgen.families.vqvae import (
    PriorConfig,
    PriorLightning,
    VQVAEConfig,
    VQVAELightning,
)

__all__ = [
    "Capabilities",
    "ContinuousConfig",
    "ContinuousGenerativeModule",
    "DiffusionModule",
    "FlowMatchingModule",
    "GenerativeModule",
    "Pulse2PulseConfig",
    "Pulse2PulseGAN",
    "PriorConfig",
    "PriorLightning",
    "VAEConfig",
    "VAELightning",
    "VQVAEConfig",
    "VQVAELightning",
]
