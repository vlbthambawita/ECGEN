from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pytorch_lightning as pl


class ResidualBlock1D(nn.Module):
    """1D Residual block with group normalization"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual


class Encoder1D(nn.Module):
    """1D Encoder for ECG signals"""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch

            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)

        return mean, logvar


class Decoder1D(nn.Module):
    """1D Decoder for ECG signals"""

    def __init__(
        self,
        out_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels

        channel_multipliers = tuple(reversed(channel_multipliers))

        ch = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)

        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch

            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


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


def vae_loss(
    recon: Tensor,
    x: Tensor,
    mean: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.0001,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    VAE loss function

    Args:
        recon: Reconstructed signal
        x: Original signal
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / (x.size(0) * x.size(1) * x.size(2))

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


@dataclass
class VAEConfig:
    in_channels: int = 12
    base_channels: int = 64
    latent_channels: int = 8
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    lr: float = 1e-4
    kl_weight: float = 0.0001
    b1: float = 0.9
    b2: float = 0.999


class VAELightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VAE1D model.
    """

    def __init__(self, config: VAEConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VAEConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.vae = VAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
        )

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vae(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs: Tensor = batch["ecg_signals"]

        recon, mean, logvar = self.vae(ecgs)
        total_loss, recon_loss, kl_loss = vae_loss(recon, ecgs, mean, logvar, self.config.kl_weight)

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs: Tensor = batch["ecg_signals"]

        recon, mean, logvar = self.vae(ecgs)
        total_loss, recon_loss, kl_loss = vae_loss(recon, ecgs, mean, logvar, self.config.kl_weight)

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()

        return total_loss

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruct input ECG signals."""
        self.vae.eval()
        recon, _, _ = self.vae(x)
        return recon

    @torch.no_grad()
    def sample(self, n_samples: int = 16, seq_length: int = 5000) -> Tensor:
        """Sample from the latent space."""
        self.vae.eval()
        device = next(self.vae.parameters()).device

        latent_length = seq_length // (2 ** (len(self.config.channel_multipliers) - 1))
        z = torch.randn(
            n_samples,
            self.config.latent_channels,
            latent_length,
            device=device,
        )
        samples = self.vae.decode(z)
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer
