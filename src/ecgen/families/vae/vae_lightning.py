from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.families.vae.config import VAEConfig
from ecgen.families.vae.losses import cvae_loss, vae_loss
from ecgen.families.vae.vae_1d import VAE1D
from ecgen.registry import FAMILIES
from ecgen.types import ConditionOutput
from ecgen.utils.config import resolve_config


@FAMILIES.register("vae")
class VAELightning(GenerativeModule):
    """
    PyTorch Lightning wrapper for VAE1D model.

    With ``config.conditioning`` it is a conditional VAE: global conditions
    (``vector``) modulate the encoder and decoder with FiLM, time-aligned
    conditions (``dense``, e.g. a masked ECG) are concatenated to their inputs,
    and ``learned_prior`` replaces N(0, I) with ``p(z | c)``.
    """

    family = "vae"
    supports = Capabilities(conditional=True)
    accepts = frozenset({"vector", "dense"})

    def __init__(
        self, config: VAEConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        config = resolve_config(VAEConfig, config, kwargs)
        self.save_hyperparameters(config.__dict__)
        self.config = config
        if config.recon_on not in ("all", "unobserved"):
            raise ValueError(
                f"recon_on must be 'all' or 'unobserved', got {config.recon_on!r}"
            )

        self.init_conditioning(config.conditioning)
        cond_dim = dense_channels = 0
        if self.conditioner is not None:
            if "vector" in self.conditioner.slots:
                cond_dim = self.conditioner.vector_dim
            dense_channels = self.conditioner.dense_channels

        self.vae = VAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            cond_dim=cond_dim,
            dense_channels=dense_channels,
            learned_prior=config.learned_prior,
            latent_type=config.latent_type,
            global_latent_dim=config.global_latent_dim,
            seq_length=config.seq_length,
        )
        if config.learned_prior and self.conditioner is None:
            raise ValueError("learned_prior needs a conditioning config")

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vae(x)

    @property
    def fixed_seq_length(self) -> Optional[int]:
        return self.config.seq_length if self.config.latent_type == "global" else None

    def _step(self, batch: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ecgs = self.get_signal(batch)
        cond = self.condition(batch)
        if cond is None:
            recon, mean, logvar = self.vae(ecgs)
            total, rec, kl = vae_loss(recon, ecgs, mean, logvar, self.config.kl_weight)
            return total, rec, kl, ecgs, recon

        recon, mean, logvar = self.vae(ecgs, cond.vector, cond.dense)
        p_mean, p_logvar = self.vae.prior_params(
            ecgs.shape[-1] // self.vae.downsample_factor,
            cond.vector,
            cond.dense,
            ecgs.shape[0],
            ecgs.device,
        )
        mask = None
        if self.config.recon_on == "unobserved" and batch.obs_mask is not None:
            mask = 1.0 - batch.obs_mask
        total, rec, kl = cvae_loss(
            recon, ecgs, mean, logvar, p_mean, p_logvar, self.config.kl_weight, mask
        )
        return total, rec, kl, ecgs, recon

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        total_loss, recon_loss, kl_loss, _, _ = self._step(batch)

        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        total_loss, recon_loss, kl_loss, ecgs, recon = self._step(batch)

        self.log(
            "val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()

        return total_loss

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruct input ECG signals (unconditional models)."""
        self.vae.eval()
        recon, _, _ = self.vae(x)
        return recon

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        seq_length: int = 5000,
        cond: Optional[Mapping[str, Any]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        paste_observed: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        """
        Sample ECGs: ``z`` from the prior (``p(z | c)`` if learned), then decode.

        Args:
            n_samples: Number of ECGs.
            seq_length: Length (multiple of the downsampling factor).
            cond: Raw condition values (see :meth:`prepare_condition`).
            obs: Partially observed ECG ``[n, C, T]``.
            obs_mask: Mask of ``obs`` (1 = observed).
            paste_observed: Copy the observed part of ``obs`` into the output.
        """
        self.vae.eval()
        device = next(self.vae.parameters()).device
        if self.config.latent_type == "global":
            seq_length = self.config.seq_length
        latent_length = seq_length // self.vae.downsample_factor
        c: Optional[ConditionOutput] = self.encode_condition(
            n_samples, cond, obs, obs_mask
        )
        vector = c.vector if c is not None else None
        dense = c.dense if c is not None else None

        mean, logvar = self.vae.prior_params(
            latent_length, vector, dense, n_samples, device
        )
        noise = torch.randn(
            n_samples, *self.vae.latent_shape(seq_length), device=device
        )
        z = mean + noise * (0.5 * logvar).exp()
        samples = self.vae.decode(z, vector, dense)
        if paste_observed and obs is not None:
            m = torch.ones_like(obs) if obs_mask is None else obs_mask.to(obs)
            samples = obs.to(samples) * m.to(samples) + samples * (1 - m.to(samples))
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer
