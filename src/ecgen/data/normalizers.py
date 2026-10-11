"""
Normalization of numeric condition fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureNormalizer:
    """
    Per-column standardization, ``(x - mean) / std``, with missing values.

    Fit it on the training split only and reuse it for validation, test, and
    sampling; :meth:`to_dict` / :meth:`from_dict` make it storable in a checkpoint.
    Missing values (NaN) are ignored when fitting and come out of
    :meth:`transform` as 0 with ``missing=True``.
    """

    columns: List[str]
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    eps: float = 1e-6

    @classmethod
    def fit(
        cls, df: pd.DataFrame, columns: Sequence[str], eps: float = 1e-6
    ) -> "FeatureNormalizer":
        norm = cls(columns=list(columns), eps=eps)
        for name in norm.columns:
            vals = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                norm.mean[name], norm.std[name] = 0.0, 1.0
            else:
                norm.mean[name] = float(np.mean(vals))
                norm.std[name] = float(np.std(vals)) + eps
        return norm

    def transform(self, values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize one row of raw values (ordered as :attr:`columns`).

        Returns:
            ``(normalized float32 [F], missing bool [F])``
        """
        raw = np.asarray(values, dtype=np.float64)
        missing = np.isnan(raw)
        mean = np.array([self.mean[c] for c in self.columns])
        std = np.array([self.std[c] for c in self.columns])
        out = np.where(missing, 0.0, (raw - mean) / std)
        return out.astype(np.float32), missing

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.array([self.mean[c] for c in self.columns])
        std = np.array([self.std[c] for c in self.columns])
        return np.asarray(values) * std + mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "mean": dict(self.mean),
            "std": dict(self.std),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            columns=list(d["columns"]),
            mean=dict(d["mean"]),
            std=dict(d["std"]),
            eps=d.get("eps", 1e-6),
        )
