from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.components.encoders import Encoder1D
from models.components.decoders import Decoder1D
from models.vqvae.vector_quantizer import VectorQuantizer


class VQVAE1D(nn.Module):
    """
    Vector Quantized Variational Autoencoder for ECG signals.
    
    Uses discrete latent representations instead of continuous ones.
    Architecture follows the existing VAE1D design but with vector quantization.
    
    Args:
        in_channels: Number of input channels (ECG leads)
        base_channels: Base number of channels in encoder/decoder
        latent_channels: Number of channels in latent space (embedding_dim)
        channel_multipliers: Channel multipliers for each downsampling stage
        num_res_blocks: Number of residual blocks per stage
        num_embeddings: Size of the discrete codebook
        commitment_cost: Weight for commitment loss in VQ
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_embeddings = num_embeddings

        # Encoder: ECG -> continuous latent
        self.encoder = Encoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

        # Vector Quantizer: continuous -> discrete
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            commitment_cost=commitment_cost,
        )

        # Decoder: discrete latent -> ECG
        self.decoder = Decoder1D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input to continuous latent representation.
        
        Note: For VQ-VAE, encoder outputs a single tensor (not mean/logvar like VAE).
        We return it twice for API compatibility with VAE.
        
        Args:
            x: Input ECG signals [B, C, L]
            
        Returns:
            z: Continuous latent [B, latent_channels, L//downsample_factor]
            z: Same tensor (for compatibility)
        """
        z, _ = self.encoder(x)  # Encoder returns (mean, logvar), we only use mean
        return z, z

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Quantize continuous latent to discrete codes.
        
        Args:
            z: Continuous latent [B, latent_channels, L_latent]
            
        Returns:
            vq_loss: Vector quantization loss
            z_q: Quantized latent [B, latent_channels, L_latent]
            indices: Codebook indices [B, L_latent]
        """
        return self.vq(z)

    def decode(self, z_q: Tensor) -> Tensor:
        """
        Decode quantized latent to reconstruction.
        
        Args:
            z_q: Quantized latent [B, latent_channels, L_latent]
            
        Returns:
            Reconstructed ECG [B, in_channels, L]
        """
        return self.decoder(z_q)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.
        
        Args:
            x: Input ECG signals [B, in_channels, L]
            
        Returns:
            recon: Reconstructed ECG [B, in_channels, L]
            vq_loss: Vector quantization loss
            indices: Codebook indices [B, L_latent]
        """
        # Encode
        z, _ = self.encode(x)

        # Quantize
        vq_loss, z_q, indices = self.quantize(z)

        # Decode
        recon = self.decode(z_q)

        return recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor) -> Tensor:
        """
        Encode input ECG to discrete codebook indices.
        Used for training the prior model.
        
        Args:
            x: Input ECG signals [B, in_channels, L]
            
        Returns:
            indices: Codebook indices [B, L_latent]
        """
        z, _ = self.encode(x)
        _, _, indices = self.quantize(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """
        Decode from discrete codebook indices to ECG.
        Used for generation from the prior model.
        
        Args:
            indices: Codebook indices [B, L_latent]
            
        Returns:
            Reconstructed ECG [B, in_channels, L]
        """
        z_q = self.vq.get_codebook_entry(indices)
        recon = self.decode(z_q)
        return recon
