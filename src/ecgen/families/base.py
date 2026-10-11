"""
Shared base class for generative-model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytorch_lightning as pl
from torch import Tensor

from ecgen.types import get_signal


@dataclass(frozen=True)
class Capabilities:
    """
    What a family supports; checked when a config is built.

    Attributes:
        conditional: Accepts condition fields.
        cfg: Supports classifier-free guidance at sampling time.
        inpainting: Can fill in unobserved parts of an ECG at sampling time.
        likelihood: Can evaluate (a bound on) log-likelihood.
        stages: Number of training stages (2 for VQ-VAE + prior).
    """

    conditional: bool = False
    cfg: bool = False
    inpainting: bool = False
    likelihood: bool = False
    stages: int = 1


class GenerativeModule(pl.LightningModule):
    """
    Base ``LightningModule`` for every family.

    Each family owns its own training logic (loss, optimizers, manual vs.
    automatic optimization), so this class only fixes the shared contract:

    - ``__init__(config=None, **kwargs)`` built with ``utils.config.resolve_config``,
      so ``load_from_checkpoint(path)`` works from the flat saved hyperparameters;
    - batches are read with :meth:`get_signal`, which accepts :class:`~ecgen.types.ECGBatch`
      and the legacy dict / tuple formats;
    - :meth:`sample` returns ``[n, C, T]`` generated ECGs.
    """

    family: ClassVar[str] = ""
    supports: ClassVar[Capabilities] = Capabilities()

    @staticmethod
    def get_signal(batch: Any) -> Tensor:
        return get_signal(batch)

    def sample(self, n_samples: int = 16, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement sample()")
