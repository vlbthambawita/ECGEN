from __future__ import annotations

from typing import Any

import torch
import pytorch_lightning as pl
from torch import Tensor

from models.vae.vae_1d import VAE1D
from models.vae.config import VAEConfig
from training.losses.vae_losses import vae_loss


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
