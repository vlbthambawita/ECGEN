from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE model."""

    # Model architecture
    in_channels: int = 12
    base_channels: int = 64
    latent_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2

    # Vector quantization
    num_embeddings: int = 512
    commitment_cost: float = 0.25

    # Training hyperparameters
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999


@dataclass
class PriorConfig:
    """Configuration for PixelCNN prior model."""

    # Model architecture
    num_embeddings: int = 512
    hidden_dim: int = 128
    num_layers: int = 10
    kernel_size: int = 2
    dilation_cycle: int = (
        10  # dilations 1, 2, 4, ..., 512 -> receptive field of 1024 codes
    )

    # Training hyperparameters
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999

    # VQ-VAE checkpoint
    vqvae_checkpoint: str = ""
    # VQ-VAE architecture, filled in when vqvae_checkpoint is loaded so that
    # prior checkpoints can be restored without the stage-1 file
    vqvae_config: Optional[dict[str, Any]] = None

    # Conditional prior (the stage-1 VQ-VAE stays unconditional); see ecgen.conditioning
    conditioning: Optional[dict[str, Any]] = None
