from __future__ import annotations

from typing import Any, Mapping, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ecgen.families.vqvae.pixelcnn_prior import PixelCNNPrior
from ecgen.families.vqvae.vqvae_lightning import VQVAELightning
from ecgen.families.vqvae.config import PriorConfig
from ecgen.families.vqvae.losses import prior_loss
from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.registry import FAMILIES
from ecgen.utils.config import resolve_config


@FAMILIES.register("vqvae_prior")
class PriorLightning(GenerativeModule):
    """
    PyTorch Lightning wrapper for PixelCNN prior model.

    Handles training and sampling for Stage 2 (prior training).
    Requires a trained VQ-VAE checkpoint to extract discrete codes.
    """

    family = "vqvae"
    supports = Capabilities(conditional=True, cfg=True, likelihood=True, stages=2)
    accepts = frozenset({"vector", "dense"})

    def __init__(
        self, config: PriorConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        config = resolve_config(PriorConfig, config, kwargs)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.init_conditioning(config.conditioning)
        cond_dim = dense_channels = 0
        if self.conditioner is not None:
            cond_dim = (
                self.conditioner.vector_dim if "vector" in self.conditioner.slots else 0
            )
            dense_channels = self.conditioner.dense_channels

        self.prior = PixelCNNPrior(
            num_embeddings=config.num_embeddings,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            kernel_size=config.kernel_size,
            dilation_cycle=config.dilation_cycle,
            cond_dim=cond_dim,
            dense_channels=dense_channels,
        )

        # Frozen VQ-VAE for encoding/decoding. When restoring a prior checkpoint,
        # vqvae_config is set and the VQ-VAE weights come from this checkpoint's
        # state_dict, so the original stage-1 file is not needed.
        self.vqvae: VQVAELightning | None = None
        if config.vqvae_config is not None:
            self.vqvae = VQVAELightning(config.vqvae_config)
            self.vqvae.eval()
            self.vqvae.freeze()
        elif config.vqvae_checkpoint:
            self._load_vqvae(config.vqvae_checkpoint)

    def _load_vqvae(self, checkpoint_path: str) -> None:
        """Load frozen VQ-VAE model from checkpoint."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")

        print(f"Loading VQ-VAE from: {checkpoint_path}")
        self.vqvae = VQVAELightning.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        self.vqvae.eval()
        self.vqvae.freeze()
        # Store the architecture so prior checkpoints are self-contained
        vqvae_config = dict(self.vqvae.config.__dict__)
        self.config.vqvae_config = vqvae_config
        self.hparams["vqvae_config"] = vqvae_config
        print("VQ-VAE loaded and frozen")

    def forward(self, indices: Tensor) -> Tensor:
        return self.prior(indices)

    def compute_loss(
        self,
        indices: Tensor,
        c: Optional[Tensor] = None,
        dense: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-entropy of each code given the codes before it (logits at t predict code t)."""
        logits = self.prior(indices, c, dense)  # [B, num_embeddings, L]
        return prior_loss(logits, indices)

    @staticmethod
    def _latent_dense(dense: Optional[Tensor], length: int) -> Optional[Tensor]:
        return None if dense is None else F.adaptive_avg_pool1d(dense, length)

    def _batch_loss(self, batch: Any) -> Tensor:
        ecgs = self.get_signal(batch)
        # Extract discrete codes from frozen VQ-VAE
        with torch.no_grad():
            if self.vqvae is None:
                raise RuntimeError(
                    "VQ-VAE not loaded. Provide vqvae_checkpoint in config."
                )
            indices = self.vqvae.encode_to_indices(ecgs)  # [B, L]
        cond = self.condition(batch)
        if cond is None:
            return self.compute_loss(indices)
        return self.compute_loss(
            indices, cond.vector, self._latent_dense(cond.dense, indices.shape[1])
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._batch_loss(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._batch_loss(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        seq_length: int = 5000,
        temperature: float = 1.0,
        cond: Optional[Mapping[str, Any]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        guidance: float = 1.0,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate ECG samples by sampling from the prior and decoding.

        Args:
            n_samples: Number of samples to generate
            seq_length: Length of ECG sequence
            temperature: Sampling temperature
            cond: Raw condition values (conditional priors)
            obs: Partially observed ECG (dense conditions)
            obs_mask: Mask of ``obs`` (1 = observed)
            guidance: Classifier-free guidance scale (1 = off)

        Returns:
            Generated ECG samples [B, C, L]
        """
        if self.vqvae is None:
            raise RuntimeError("VQ-VAE not loaded. Cannot decode samples.")

        self.prior.eval()
        device = next(self.prior.parameters()).device

        latent_length = self.vqvae.latent_length(seq_length)

        kwargs_c: dict[str, Any] = {}
        c = self.encode_condition(n_samples, cond, obs, obs_mask)
        if c is not None:
            kwargs_c = {
                "c": c.vector,
                "dense": self._latent_dense(c.dense, latent_length),
            }
            if guidance != 1.0:
                u = self.encode_condition(n_samples, cond, obs, obs_mask, drop="all")
                kwargs_c["c_uncond"] = u.vector
                kwargs_c["dense_uncond"] = self._latent_dense(u.dense, latent_length)
                kwargs_c["guidance"] = guidance

        # Sample discrete codes from prior
        indices = self.prior.sample(
            batch_size=n_samples,
            latent_length=latent_length,
            temperature=temperature,
            device=device,
            **kwargs_c,
        )

        # Decode to ECG
        samples = self.vqvae.decode_from_indices(indices)

        return samples

    def configure_optimizers(self) -> Any:
        params = list(self.prior.parameters())
        if self.conditioner is not None:
            params += list(self.conditioner.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer
