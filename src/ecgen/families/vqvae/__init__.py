"""Vector Quantized VAE (VQ-VAE) family: stage-1 autoencoder and stage-2 PixelCNN prior."""

from ecgen.families.vqvae.config import PriorConfig, VQVAEConfig
from ecgen.families.vqvae.losses import prior_loss, vqvae_loss
from ecgen.families.vqvae.pixelcnn_prior import CausalGatedBlock, PixelCNNPrior
from ecgen.families.vqvae.prior_lightning import PriorLightning
from ecgen.families.vqvae.vector_quantizer import VectorQuantizer
from ecgen.families.vqvae.vqvae_1d import VQVAE1D
from ecgen.families.vqvae.vqvae_lightning import VQVAELightning

__all__ = [
    "VectorQuantizer",
    "VQVAE1D",
    "PixelCNNPrior",
    "CausalGatedBlock",
    "VQVAELightning",
    "PriorLightning",
    "VQVAEConfig",
    "PriorConfig",
    "vqvae_loss",
    "prior_loss",
]
