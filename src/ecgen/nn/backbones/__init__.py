"""
Denoiser / velocity backbones ``forward(x, t, cond)``; importing this package
registers them in ``ecgen.registry.BACKBONES``: ``unet1d``, ``dit1d``, ``ssm`` (S4D), ``mamba`` (needs ``mamba-ssm``).
"""

from ecgen.nn.backbones.dit1d import DiT1d
from ecgen.nn.backbones.ssm import S4D, MambaBackbone, SSMBackbone
from ecgen.nn.backbones.unet1d import UNet1d

__all__ = ["DiT1d", "MambaBackbone", "S4D", "SSMBackbone", "UNet1d"]
