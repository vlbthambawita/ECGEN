"""
Data contracts shared by datasets, conditioners, and generative families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import Tensor


@dataclass
class ECGBatch:
    """
    A collated batch of ECGs and their condition fields.

    Attributes:
        signal: ``[B, C, T]`` ECG signals.
        lead_names: Names of the ``C`` leads, in channel order.
        fs: Sampling rate in Hz.
        cond: Raw condition fields by name (e.g. ``"measurements": [B, F]``).
            Empty for unconditional data.
        cond_missing: Per-field missing masks, ``True`` = missing, same keys as ``cond``.
        obs_mask: ``[B, C, T]`` mask, 1 = observed, for partial-ECG conditioning.
        meta: Record identifiers etc.; never fed to a model.
    """

    signal: Tensor
    lead_names: List[str]
    fs: int
    cond: Dict[str, Tensor] = field(default_factory=dict)
    cond_missing: Dict[str, Tensor] = field(default_factory=dict)
    obs_mask: Optional[Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.signal.shape[0]


@dataclass
class ConditionOutput:
    """
    Encoded conditions, in the slots a backbone can consume.

    Attributes:
        vector: ``[B, D]`` global embedding (labels, tabular, pooled text);
            injected with FiLM / AdaGN / AdaLN or added to the time embedding.
        tokens: ``[B, N, D]`` token sequence for cross-attention.
        tokens_mask: ``[B, N]`` bool, ``True`` = valid token.
        dense: ``[B, C', T]`` time-aligned conditions (masked ECG + mask, PPG, ...);
            concatenated to the input.
    """

    vector: Optional[Tensor] = None
    tokens: Optional[Tensor] = None
    tokens_mask: Optional[Tensor] = None
    dense: Optional[Tensor] = None


def get_signal(batch: Any) -> Tensor:
    """
    Return the ``[B, C, T]`` ECG tensor from any supported batch format.

    Accepts an :class:`ECGBatch`, a dict with ``"signal"`` or the legacy
    ``"ecg_signals"`` key, or a legacy ``(ecg, features)`` tuple.
    """
    if isinstance(batch, ECGBatch):
        return batch.signal
    if isinstance(batch, dict):
        if "signal" in batch:
            return batch["signal"]
        if "ecg_signals" in batch:
            return batch["ecg_signals"]
        raise KeyError(
            f"Batch dict has no 'signal' or 'ecg_signals' key: {sorted(batch)}"
        )
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, Tensor):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")
