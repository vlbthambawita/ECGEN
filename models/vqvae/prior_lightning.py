from __future__ import annotations

from typing import Any
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch import Tensor

from models.vqvae.pixelcnn_prior import PixelCNNPrior
from models.vqvae.vqvae_lightning import VQVAELightning
from models.vqvae.config import PriorConfig
from training.losses.vqvae_losses import prior_loss


class PriorLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for PixelCNN prior model.
    
    Handles training and sampling for Stage 2 (prior training).
    Requires a trained VQ-VAE checkpoint to extract discrete codes.
    """

    def __init__(self, config: PriorConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = PriorConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.prior = PixelCNNPrior(
            num_embeddings=config.num_embeddings,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )

        # Load frozen VQ-VAE for encoding
        self.vqvae: VQVAELightning | None = None
        if config.vqvae_checkpoint:
            self._load_vqvae(config.vqvae_checkpoint)

    def _load_vqvae(self, checkpoint_path: str) -> None:
        """Load frozen VQ-VAE model from checkpoint."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
        
        print(f"Loading VQ-VAE from: {checkpoint_path}")
        self.vqvae = VQVAELightning.load_from_checkpoint(checkpoint_path)
        self.vqvae.eval()
        self.vqvae.freeze()
        print("VQ-VAE loaded and frozen")

    def forward(self, indices: Tensor) -> Tensor:
        return self.prior(indices)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            ecgs: Tensor = batch["ecg_signals"]
        else:
            ecgs, _ = batch  # MIMIC dataset returns (ecg, features)

        # Extract discrete codes from frozen VQ-VAE
        with torch.no_grad():
            if self.vqvae is None:
                raise RuntimeError("VQ-VAE not loaded. Provide vqvae_checkpoint in config.")
            indices = self.vqvae.encode_to_indices(ecgs)  # [B, L]

        # Predict next codes autoregressively
        logits = self.prior(indices)  # [B, num_embeddings, L]

        # Compute loss: predict indices[i+1] from indices[:i]
        # Shift targets by 1 position
        loss = prior_loss(logits[:, :, :-1], indices[:, 1:])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            ecgs: Tensor = batch["ecg_signals"]
        else:
            ecgs, _ = batch  # MIMIC dataset returns (ecg, features)

        # Extract discrete codes
        with torch.no_grad():
            if self.vqvae is None:
                raise RuntimeError("VQ-VAE not loaded. Provide vqvae_checkpoint in config.")
            indices = self.vqvae.encode_to_indices(ecgs)

        # Predict next codes
        logits = self.prior(indices)

        # Compute loss
        loss = prior_loss(logits[:, :, :-1], indices[:, 1:])

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        seq_length: int = 5000,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate ECG samples by sampling from the prior and decoding.
        
        Args:
            n_samples: Number of samples to generate
            seq_length: Length of ECG sequence
            temperature: Sampling temperature
            
        Returns:
            Generated ECG samples [B, C, L]
        """
        if self.vqvae is None:
            raise RuntimeError("VQ-VAE not loaded. Cannot decode samples.")

        self.prior.eval()
        device = next(self.prior.parameters()).device

        # Calculate latent length from VQ-VAE architecture
        # Assumes 4 downsampling stages with stride 2 each -> 16x downsampling
        latent_length = seq_length // 16

        # Sample discrete codes from prior
        indices = self.prior.sample(
            batch_size=n_samples,
            latent_length=latent_length,
            temperature=temperature,
            device=device,
        )

        # Decode to ECG
        samples = self.vqvae.decode_from_indices(indices)

        return samples

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.prior.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer
