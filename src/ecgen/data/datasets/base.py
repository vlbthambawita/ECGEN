"""
Dataset contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch.utils.data import Dataset

from ecgen.data.normalizers import FeatureNormalizer

# "zscore": per-record (x - mean) / std (legacy default); "none": physical units
# (mV); a number: (x - mean) / number
Scale = Union[str, float, int]


def scale_signal(ecg: np.ndarray, scale: Scale, eps: float = 1e-6) -> np.ndarray:
    """Apply the dataset-level amplitude scaling to a ``[C, T]`` record."""
    if scale == "none":
        return ecg.astype(np.float32)
    mean = ecg.mean()
    if scale == "zscore":
        divisor = max(float(np.std(ecg)), eps)
    elif isinstance(scale, (int, float)) and not isinstance(scale, bool):
        divisor = float(scale)
    else:
        raise ValueError(f"Unknown scale {scale!r}; use 'zscore', 'none', or a number")
    return ((ecg.astype(np.float32) - mean) / divisor).astype(np.float32)


class BaseECGDataset(Dataset, ABC):
    """
    Base class for ECG datasets.

    ``__getitem__`` returns a dict::

        {
            "signal": FloatTensor [C, T],
            "cond": {field_name: Tensor},          # may be empty
            "cond_missing": {field_name: BoolTensor},
            "meta": {...},                         # ids; not fed to models
        }

    and the dataset exposes :attr:`lead_names` and :attr:`fs`, which
    :class:`ecgen.data.collate.ECGCollate` attaches to every batch.
    """

    lead_names: List[str]
    fs: int
    seq_length: int
    feature_normalizer: Optional[FeatureNormalizer] = None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...

    @property
    def num_leads(self) -> int:
        return len(self.lead_names)
