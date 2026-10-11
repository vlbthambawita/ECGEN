from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
    # Conditional VAE (all off = the plain VAE)
    conditioning: Optional[dict[str, Any]] = None  # see ecgen.conditioning
    learned_prior: bool = False  # p(z | c) instead of N(0, I)
    recon_on: str = "all"  # "all" or "unobserved" (with an obs_mask)
    # "spatial" latent map (default) or one "global" vector per ECG (coherent prior samples)
    latent_type: str = "spatial"
    global_latent_dim: int = 64
    seq_length: int = 5000  # needed for latent_type="global"
