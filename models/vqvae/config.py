from __future__ import annotations

from dataclasses import dataclass


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
    num_layers: int = 3
    
    # Training hyperparameters
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    
    # VQ-VAE checkpoint
    vqvae_checkpoint: str = ""
