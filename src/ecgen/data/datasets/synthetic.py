"""
Synthetic ECG-like dataset for tests and smoke runs (no data download needed).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ecgen.data.datasets.base import BaseECGDataset, Scale, scale_signal
from ecgen.data.leads import LeadSpec, resolve_leads, STANDARD_12
from ecgen.registry import DATASETS

_SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


@DATASETS.register("synthetic")
class SyntheticECG(BaseECGDataset):
    """
    Deterministic periodic "beats" with per-record heart rate and amplitude.

    ``cond["measurements"]`` holds the normalized ``[heart_rate, amplitude]``
    that produced each record, so conditional models can be smoke-tested
    against a known ground truth.

    Args:
        split: ``"train"``, ``"val"``, or ``"test"`` (different records per split).
        num_samples: Records in this split.
        leads: Lead spec (see :func:`ecgen.data.leads.resolve_leads`); default all 12.
        seq_length: Samples per record.
        fs: Sampling rate in Hz.
        scale: As in :func:`ecgen.data.datasets.base.scale_signal`.
        seed: Base seed; record ``i`` of a split is always the same signal.
    """

    HR_RANGE = (45.0, 150.0)
    AMP_RANGE = (0.5, 2.0)

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 64,
        leads: LeadSpec = None,
        seq_length: int = 5000,
        fs: int = 500,
        scale: Scale = "zscore",
        seed: int = 0,
    ) -> None:
        if split not in _SPLIT_OFFSET:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.num_samples = num_samples
        self.lead_names = resolve_leads(leads, STANDARD_12)
        self.seq_length = seq_length
        self.fs = fs
        self.scale = scale
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + _SPLIT_OFFSET[self.split] + idx)
        hr = rng.uniform(*self.HR_RANGE)
        amp = rng.uniform(*self.AMP_RANGE)

        t = np.arange(self.seq_length) / self.fs
        phase = (t * hr / 60.0 + rng.uniform()) % 1.0
        beat = amp * np.exp(-((phase - 0.3) ** 2) / 2e-4) + 0.2 * amp * np.exp(
            -((phase - 0.6) ** 2) / 4e-3
        )
        gains = rng.uniform(-1.0, 1.0, size=(len(self.lead_names), 1))
        noise = 0.02 * rng.standard_normal((len(self.lead_names), self.seq_length))
        ecg = (gains * beat[None, :] + noise).astype(np.float32)

        hr_mid, hr_half = np.mean(self.HR_RANGE), np.ptp(self.HR_RANGE) / 2
        amp_mid, amp_half = np.mean(self.AMP_RANGE), np.ptp(self.AMP_RANGE) / 2
        features = np.array(
            [(hr - hr_mid) / hr_half, (amp - amp_mid) / amp_half], dtype=np.float32
        )

        return {
            "signal": torch.from_numpy(scale_signal(ecg, self.scale)),
            "cond": {"measurements": torch.from_numpy(features)},
            "cond_missing": {"measurements": torch.zeros(2, dtype=torch.bool)},
            "meta": {"index": idx},
        }
