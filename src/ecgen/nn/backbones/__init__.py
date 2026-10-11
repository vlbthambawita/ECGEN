"""
Denoiser / velocity backbones ``forward(x, t, cond)``; importing this package
registers them in ``ecgen.registry.BACKBONES`` (``unet1d``).
"""

from ecgen.nn.backbones.unet1d import UNet1d

__all__ = ["UNet1d"]
