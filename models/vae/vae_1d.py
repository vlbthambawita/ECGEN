from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.components.encoders import Encoder1D
from models.components.decoders import Decoder1D


class VAE1D(nn.Module):
    """Variational Autoencoder for ECG signals"""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = Encoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

        self.decoder = Decoder1D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution"""
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to reconstruction"""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor) -> Tensor:
        """Encode to latent (for diffusion training)"""
        mean, _ = self.encode(x)
        return mean

    @torch.no_grad()
    def decode_from_latent(self, z: Tensor) -> Tensor:
        """Decode from latent (for diffusion sampling)"""
        return self.decode(z)
