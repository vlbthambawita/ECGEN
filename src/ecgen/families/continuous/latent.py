"""
Latent diffusion / flow matching: the process runs in the latent space of a
frozen, pre-trained autoencoder (stage 1), as in latent diffusion models
(Rombach et al., 2022).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ecgen.families.continuous.module import (
    ContinuousConfig,
    ContinuousGenerativeModule,
)
from ecgen.registry import FAMILIES


@dataclass
class LatentConfig(ContinuousConfig):
    """
    :class:`ContinuousConfig` plus the frozen stage-1 autoencoder.

    Attributes:
        autoencoder_checkpoint: Stage-1 checkpoint (``vae`` with ``latent_type: spatial``,
            or ``vqvae``).
        autoencoder_family: ``"vae"`` or ``"vqvae"``.
        autoencoder_config: Filled in from the checkpoint, so latent checkpoints
            restore the autoencoder from their own weights (stage 1 is not needed again).
        latent_scale: Latents are multiplied by this before the process (``None`` =
            ``1 / std`` of the first training batch's latents, then stored).
    """

    autoencoder_checkpoint: str = ""
    autoencoder_family: str = "vae"
    autoencoder_config: Optional[Dict[str, Any]] = None
    latent_scale: Optional[float] = None
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "unet1d",
            "params": {"channel_mults": [1, 2, 2]},
        }
    )


class LatentContinuousModule(ContinuousGenerativeModule):
    """
    Diffusion / flow in a frozen autoencoder's latent space.

    - ``vae`` stage 1: the encoder's posterior mean is the latent; decode with the decoder.
    - ``vqvae`` stage 1: the quantized embeddings are the latent; sampled latents
      are snapped to the codebook before decoding.

    Dense conditions are average-pooled to the latent length; observation masks
    and training-free inpainting are signal-space features and are not used here
    (a ``masked_signal`` condition still works).
    """

    config_cls = LatentConfig

    def _model_channels(self, config: LatentConfig) -> int:
        if config.autoencoder_family not in ("vae", "vqvae"):
            raise ValueError(
                f"autoencoder_family must be 'vae' or 'vqvae', got {config.autoencoder_family!r}"
            )
        if config.loss_on != "all" or config.clean_observed:
            raise ValueError(
                "loss_on='unobserved' and clean_observed are signal-space options"
            )
        from ecgen.families import VAELightning, VQVAELightning

        cls = VAELightning if config.autoencoder_family == "vae" else VQVAELightning
        if config.autoencoder_config is not None:
            ae = cls(**config.autoencoder_config)
        else:
            if (
                not config.autoencoder_checkpoint
                or not Path(config.autoencoder_checkpoint).exists()
            ):
                raise FileNotFoundError(
                    f"Autoencoder checkpoint not found: {config.autoencoder_checkpoint!r}"
                )
            ae = cls.load_from_checkpoint(
                config.autoencoder_checkpoint, map_location="cpu"
            )
            config.autoencoder_config = dict(ae.config.__dict__)
            self.hparams["autoencoder_config"] = config.autoencoder_config
        if config.autoencoder_family == "vae":
            if ae.config.latent_type != "spatial" or ae.is_conditional:
                raise ValueError(
                    "Latent models need an unconditional VAE with latent_type='spatial'"
                )
            channels = ae.config.latent_channels
        else:
            channels = ae.config.latent_channels
        ae.eval()
        ae.freeze()
        self.autoencoder = ae
        self.downsample = 2 ** (len(ae.config.channel_multipliers) - 1)
        scale = config.latent_scale if config.latent_scale is not None else 0.0
        self.register_buffer("latent_scale", torch.tensor(float(scale)))
        return channels

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()  # stays frozen (no dropout / batch-stat changes)
        return self

    # ------------------------------------------------------------ latents

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Unscaled latent of a signal."""
        if self.config.autoencoder_family == "vae":
            mean, _ = self.autoencoder.vae.encode(x)
            return mean
        z, _ = self.autoencoder.vqvae.encoder(x)
        _, z_q, _ = self.autoencoder.vqvae.quantize(z)
        return z_q

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Signal from an unscaled latent."""
        if self.config.autoencoder_family == "vae":
            return self.autoencoder.vae.decode(z)
        _, z_q, _ = self.autoencoder.vqvae.quantize(z)
        return self.autoencoder.vqvae.decode(z_q)

    def to_model_space(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        if float(self.latent_scale) == 0.0:  # first batch: fix the scale
            self.latent_scale.fill_(1.0 / z.float().std().clamp_min(1e-6))
        return z * self.latent_scale

    def from_model_space(self, z: Tensor) -> Tensor:
        return self.decode(z / self.latent_scale)

    def model_space_shape(self, n: int, seq_length: int) -> tuple[int, ...]:
        if seq_length % self.downsample:
            raise ValueError(f"seq_length must be a multiple of {self.downsample}")
        return (n, self._latent_channels(), seq_length // self.downsample)

    def _latent_channels(self) -> int:
        return self.autoencoder.config.latent_channels

    def configure_optimizers(self) -> Any:
        # the frozen autoencoder has requires_grad=False, so the base filter skips it
        return super().configure_optimizers()


@dataclass
class LatentFlowConfig(LatentConfig):
    process: Dict[str, Any] = field(default_factory=lambda: {"name": "rectified_flow"})


@FAMILIES.register("latent_diffusion")
class LatentDiffusionModule(LatentContinuousModule):
    """Latent diffusion (default process ``ddpm``)."""

    family = "latent_diffusion"


@FAMILIES.register("latent_flow_matching")
class LatentFlowMatchingModule(LatentContinuousModule):
    """Latent flow matching (default process ``rectified_flow``)."""

    family = "latent_flow_matching"
    config_cls = LatentFlowConfig
