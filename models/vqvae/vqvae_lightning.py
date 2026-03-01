from __future__ import annotations

from typing import Any

import torch
import pytorch_lightning as pl
from torch import Tensor

from models.vqvae.vqvae_1d import VQVAE1D
from models.vqvae.config import VQVAEConfig
from training.losses.vqvae_losses import vqvae_loss


class VQVAELightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VQ-VAE model.
    
    Handles training, validation, and sampling for Stage 1 (VQ-VAE training).
    """

    def __init__(self, config: VQVAEConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = VQVAEConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.vqvae = VQVAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
        )

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vqvae(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            ecgs: Tensor = batch["ecg_signals"]
        else:
            ecgs, _ = batch  # MIMIC dataset returns (ecg, features)

        recon, vq_loss_val, indices = self.vqvae(ecgs)
        total_loss, recon_loss, vq_loss_component = vqvae_loss(recon, ecgs, vq_loss_val)

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/vq_loss", vq_loss_component, on_step=True, on_epoch=True, prog_bar=False)

        # Log codebook usage statistics
        if batch_idx % 100 == 0:
            unique_codes = torch.unique(indices).numel()
            self.log("train/unique_codes", float(unique_codes), on_step=True, on_epoch=False)
            self.log("train/codebook_usage", float(unique_codes) / self.config.num_embeddings, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            ecgs: Tensor = batch["ecg_signals"]
        else:
            ecgs, _ = batch  # MIMIC dataset returns (ecg, features)

        recon, vq_loss_val, indices = self.vqvae(ecgs)
        total_loss, recon_loss, vq_loss_component = vqvae_loss(recon, ecgs, vq_loss_val)

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/vq_loss", vq_loss_component, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log codebook usage
        unique_codes = torch.unique(indices).numel()
        self.log("val/unique_codes", float(unique_codes), on_step=False, on_epoch=True)
        self.log("val/codebook_usage", float(unique_codes) / self.config.num_embeddings, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()

        return total_loss

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruct input ECG signals."""
        self.vqvae.eval()
        recon, _, _ = self.vqvae(x)
        return recon

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor) -> Tensor:
        """
        Encode ECG to discrete codebook indices.
        Used for training the prior model.
        """
        self.vqvae.eval()
        return self.vqvae.encode_to_indices(x)

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """
        Decode from discrete codebook indices.
        Used for generation from the prior model.
        """
        self.vqvae.eval()
        return self.vqvae.decode_from_indices(indices)

    @torch.no_grad()
    def sample(self, n_samples: int = 16, seq_length: int = 5000) -> Tensor:
        """
        Sample by randomly selecting codebook indices.
        
        Note: This is not a good sampling method. For proper generation,
        use the trained prior model (Stage 2).
        """
        self.vqvae.eval()
        device = next(self.vqvae.parameters()).device

        # Calculate latent length based on downsampling
        latent_length = seq_length // (2 ** (len(self.config.channel_multipliers) - 1))
        
        # Random indices (not a good prior!)
        indices = torch.randint(
            0,
            self.config.num_embeddings,
            (n_samples, latent_length),
            device=device,
        )
        
        samples = self.vqvae.decode_from_indices(indices)
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.vqvae.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer
