from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.nn.encoders import Encoder1D
from ecgen.nn.decoders import Decoder1D


class GlobalPrior(nn.Module):
    """Learned ``p(z | c)`` for a global latent ``[B, D]``; zero-initialised (starts as N(0, I))."""

    def __init__(self, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * latent_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.latent_dim = latent_dim

    def forward(
        self, c: Optional[Tensor], batch_size: int, device
    ) -> tuple[Tensor, Tensor]:
        if c is None:
            zeros = torch.zeros(batch_size, self.latent_dim, device=device)
            return zeros, zeros
        mean, logvar = self.proj(c).chunk(2, dim=-1)
        return mean, logvar


class ConditionalPrior(nn.Module):
    """
    Learned prior ``p(z | c)``: a Gaussian per latent position.

    Global conditions give a per-channel mean/log-variance; dense conditions
    (at latent resolution) add a per-position term. Zero-initialised, so it
    starts as N(0, I).
    """

    def __init__(
        self, latent_channels: int, cond_dim: int = 0, dense_channels: int = 0
    ) -> None:
        super().__init__()
        self.vector = nn.Linear(cond_dim, 2 * latent_channels) if cond_dim else None
        self.dense = (
            nn.Conv1d(dense_channels, 2 * latent_channels, kernel_size=3, padding=1)
            if dense_channels
            else None
        )
        for m in (self.vector, self.dense):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.latent_channels = latent_channels

    def forward(
        self,
        length: int,
        c: Optional[Tensor],
        dense: Optional[Tensor],
        batch_size: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        params = torch.zeros(
            batch_size, 2 * self.latent_channels, length, device=device
        )
        if self.vector is not None and c is not None:
            params = params + self.vector(c)[..., None]
        if self.dense is not None and dense is not None:
            params = params + self.dense(dense)
        mean, logvar = params.chunk(2, dim=1)
        return mean, logvar


class VAE1D(nn.Module):
    """
    Variational Autoencoder for ECG signals; a conditional VAE when
    ``cond_dim`` (global conditions, via FiLM) or ``dense_channels``
    (time-aligned conditions, concatenated) is set.

    ``latent_type="spatial"`` (default) keeps a ``[latent_channels, T / f]``
    latent map with a per-position Gaussian prior — good reconstructions and a
    first stage for latent priors, but samples straight from the prior are not
    coherent ECGs. ``latent_type="global"`` uses one ``[global_latent_dim]``
    vector per ECG (flattened encoder output -> linear; linear -> decoder input
    map), whose prior samples decode to coherent ECGs; it needs ``seq_length``.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 0,
        dense_channels: int = 0,
        learned_prior: bool = False,
        latent_type: str = "spatial",
        global_latent_dim: int = 64,
        seq_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        if latent_type not in ("spatial", "global"):
            raise ValueError(
                f"latent_type must be 'spatial' or 'global', got {latent_type!r}"
            )
        self.latent_type = latent_type
        self.latent_channels = latent_channels

        self.encoder = Encoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            cond_dim=cond_dim,
            extra_in_channels=dense_channels,
        )

        self.decoder = Decoder1D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            cond_dim=cond_dim,
            extra_in_channels=dense_channels,
        )

        self.prior: Optional[nn.Module] = None
        if latent_type == "global":
            if not seq_length or seq_length % self.downsample_factor:
                raise ValueError(
                    "latent_type='global' needs seq_length divisible by the downsampling"
                )
            self.map_length = seq_length // self.downsample_factor
            flat = 2 * latent_channels * self.map_length
            self.to_global = nn.Linear(flat, 2 * global_latent_dim)
            self.from_global = nn.Linear(
                global_latent_dim, latent_channels * self.map_length
            )
            self.global_latent_dim = global_latent_dim
            if learned_prior:
                self.prior = (
                    GlobalPrior(global_latent_dim, cond_dim) if cond_dim else None
                )
                if dense_channels:
                    raise ValueError(
                        "learned_prior with latent_type='global' supports vector conditions only"
                    )
        elif learned_prior:
            self.prior = ConditionalPrior(latent_channels, cond_dim, dense_channels)

    def latent_dense(self, dense: Optional[Tensor], length: int) -> Optional[Tensor]:
        """Dense conditions pooled to the latent length."""
        return None if dense is None else F.adaptive_avg_pool1d(dense, length)

    def encode(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution"""
        mean, logvar = self.encoder(x, c, dense)
        if self.latent_type == "global":
            mean, logvar = self.to_global(
                torch.cat([mean, logvar], dim=1).flatten(1)
            ).chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(
        self, z: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> Tensor:
        """Decode latent to reconstruction (``dense`` at full resolution)."""
        if self.latent_type == "global":
            z = self.from_global(z).view(
                z.shape[0], self.latent_channels, self.map_length
            )
        return self.decoder(z, c, self.latent_dense(dense, z.shape[-1]))

    def prior_params(
        self,
        length: int,
        c: Optional[Tensor],
        dense: Optional[Tensor],
        batch_size: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """Mean and log-variance of ``p(z | c)`` (zeros for the standard normal prior).

        ``length`` is the latent map length (ignored for a global latent).
        """
        if self.latent_type == "global":
            if self.prior is None:
                zeros = torch.zeros(batch_size, self.global_latent_dim, device=device)
                return zeros, zeros
            return self.prior(c, batch_size, device)
        if self.prior is None:
            zeros = torch.zeros(
                batch_size, self.decoder.latent_channels, length, device=device
            )
            return zeros, zeros
        return self.prior(
            length, c, self.latent_dense(dense, length), batch_size, device
        )

    def latent_shape(self, seq_length: int) -> tuple[int, ...]:
        """Shape of one sample's latent."""
        if self.latent_type == "global":
            return (self.global_latent_dim,)
        return (self.latent_channels, seq_length // self.downsample_factor)

    def forward(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass"""
        mean, logvar = self.encode(x, c, dense)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z, c, dense)
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
