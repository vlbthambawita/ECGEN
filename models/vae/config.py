from __future__ import annotations

from dataclasses import dataclass


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
