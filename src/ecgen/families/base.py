"""
Shared base class for generative-model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from ecgen.conditioning.composite import CompositeConditioner, Drop, build_conditioner
from ecgen.data.normalizers import FeatureNormalizer
from ecgen.types import ConditionOutput, ECGBatch, get_signal


@dataclass(frozen=True)
class Capabilities:
    """
    What a family supports; checked when a config is built.

    Attributes:
        conditional: Accepts condition fields (a ``conditioning`` config).
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
    - conditional families build :attr:`conditioner` from their ``conditioning``
      config with :meth:`init_conditioning`, and get conditions with
      :meth:`condition` (training) or :meth:`prepare_condition` (raw values at
      sampling time);
    - :attr:`data_info` (lead names, fs, normalizers of the training data) is set
      by the runner and stored in checkpoints;
    - :meth:`sample` returns ``[n, C, T]`` generated ECGs.

    Class attributes:
        family: Family name.
        supports: :class:`Capabilities`.
        accepts: ConditionOutput slots the network consumes
            (subset of ``{"vector", "tokens", "dense"}``).
    """

    family: ClassVar[str] = ""
    supports: ClassVar[Capabilities] = Capabilities()
    accepts: ClassVar[FrozenSet[str]] = frozenset()

    conditioner: Optional[CompositeConditioner]

    def __init__(self) -> None:
        super().__init__()
        self.conditioner = None
        self.data_info: Dict[str, Any] = {}

    # ----------------------------------------------------------------- batch

    @staticmethod
    def get_signal(batch: Any) -> Tensor:
        return get_signal(batch)

    # ---------------------------------------------------------- conditioning

    def init_conditioning(self, spec: Optional[Mapping[str, Any]]) -> None:
        """Build :attr:`conditioner` from a ``conditioning`` config (``None`` = unconditional)."""
        conditioner = build_conditioner(spec)
        if conditioner is not None:
            if not self.supports.conditional:
                raise ValueError(f"{type(self).__name__} does not support conditioning")
            extra = conditioner.slots - set(self.accepts)
            if extra:
                raise ValueError(
                    f"{type(self).__name__} accepts condition slots {sorted(self.accepts)}, "
                    f"but the conditioning config produces {sorted(extra)}"
                )
        self.conditioner = conditioner

    @property
    def is_conditional(self) -> bool:
        return self.conditioner is not None

    def condition(self, batch: Any, drop: Drop = None) -> Optional[ConditionOutput]:
        """Encode a training batch's conditions (``None`` for unconditional models)."""
        if self.conditioner is None:
            return None
        if not isinstance(batch, ECGBatch):
            raise TypeError(
                "Conditional models need ECGBatch batches (use ECGDataModule)"
            )
        obs = batch.signal if batch.obs_mask is not None else None
        return self.conditioner(
            batch.cond,
            batch.cond_missing,
            obs=obs,
            obs_mask=batch.obs_mask,
            batch_size=len(batch),
            drop=drop,
        )

    def prepare_condition(
        self, cond: Optional[Mapping[str, Any]], n: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Turn raw condition values into model inputs for ``n`` samples.

        Numeric fields with a stored normalizer are given in raw units, either as
        a list/array of length F (broadcast to all samples) or ``[n, F]``; NaN
        or ``None`` marks a missing value. Other fields are passed as tensors (or
        anything ``torch.as_tensor`` accepts), broadcast if they have no batch
        dimension. Fields not given are left out (the model uses its null
        embedding for them).

        Returns:
            ``(cond, cond_missing)`` on this module's device.
        """
        device = self.device
        out: Dict[str, Tensor] = {}
        missing: Dict[str, Tensor] = {}
        normalizers = self.data_info.get("normalizers") or {}
        for field, value in (cond or {}).items():
            if field in normalizers:
                norm = FeatureNormalizer.from_dict(normalizers[field])
                raw = np.array(value, dtype=np.float64).reshape(-1, len(norm.columns))
                raw = np.broadcast_to(raw, (n, len(norm.columns)))
                vals, miss = zip(*(norm.transform(row) for row in raw))
                out[field] = torch.as_tensor(np.stack(vals), device=device)
                missing[field] = torch.as_tensor(np.stack(miss), device=device)
            elif isinstance(value, str) or (
                isinstance(value, (list, tuple)) and value and isinstance(value[0], str)
            ):
                texts = [value] * n if isinstance(value, str) else list(value)
                if len(texts) != n:
                    raise ValueError(
                        f"{field}: got {len(texts)} strings for {n} samples"
                    )
                out[field] = texts
            else:
                t = torch.as_tensor(value, device=device)
                if t.dim() == 0 or (t.shape[0] != n):
                    t = t.unsqueeze(0).expand(n, *t.shape)
                out[field] = t
        return out, missing

    def encode_condition(
        self,
        n: int,
        cond: Optional[Mapping[str, Any]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        drop: Drop = None,
    ) -> Optional[ConditionOutput]:
        """Raw sampling-time conditions → :class:`ConditionOutput` (``None`` if unconditional)."""
        if self.conditioner is None:
            if cond or obs is not None:
                raise ValueError(
                    f"{type(self).__name__} was trained without conditioning"
                )
            return None
        values, missing = self.prepare_condition(cond, n)
        if obs is not None:
            obs = obs.to(self.device)
            obs_mask = (
                torch.ones_like(obs)
                if obs_mask is None
                else obs_mask.to(self.device, obs.dtype)
            )
        return self.conditioner(
            values, missing, obs=obs, obs_mask=obs_mask, batch_size=n, drop=drop
        )

    # ------------------------------------------------------------- data info

    def set_data_info(self, info: Mapping[str, Any]) -> None:
        """Record the training data contract (lead names, fs, seq_length, normalizers)."""
        self.data_info = dict(info)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        from ecgen.registry import FAMILIES

        checkpoint["ecgen_data_info"] = self.data_info
        checkpoint["ecgen_family"] = FAMILIES.name_of(self)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.data_info = dict(checkpoint.get("ecgen_data_info") or {})

    @property
    def fixed_seq_length(self) -> Optional[int]:
        """Signal length the network is built for, or ``None`` if any length works."""
        return None

    # -------------------------------------------------------------- sampling

    def sample(self, n_samples: int = 16, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement sample()")
