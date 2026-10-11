"""
Synthetic ECG-like dataset for tests and smoke runs (no data download needed).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ecgen.data.datasets.base import BaseECGDataset, Scale, scale_signal
from ecgen.data.leads import LeadSpec, resolve_leads, STANDARD_12
from ecgen.data.normalizers import FeatureNormalizer
from ecgen.registry import DATASETS

_SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


@DATASETS.register("synthetic")
class SyntheticECG(BaseECGDataset):
    """
    Deterministic periodic "beats" with per-record heart rate and amplitude.

    Every record carries the ground truth that produced it, so conditional
    models can be tested against known targets:

    - ``cond["measurements"]``: ``[heart_rate (bpm), amplitude (mV)]``, normalized
      with :attr:`feature_normalizer` (fixed statistics of the sampling ranges);
    - ``cond["rhythm"]``: class 0 = brady (< 60 bpm), 1 = normal, 2 = tachy (> 100 bpm);
    - ``cond["labels"]``: multi-hot ``[brady, tachy, high_amplitude (> 1.25 mV)]``.

    Args:
        split: ``"train"``, ``"val"``, or ``"test"`` (different records per split).
        num_samples: Records in this split.
        leads: Lead spec (see :func:`ecgen.data.leads.resolve_leads`); default all 12.
        seq_length: Samples per record.
        fs: Sampling rate in Hz.
        scale: As in :func:`ecgen.data.datasets.base.scale_signal`; default ``"none"``
            keeps the amplitude, which ``"zscore"`` would normalize away.
        seed: Base seed; record ``i`` of a split is always the same signal.
    """

    HR_RANGE = (45.0, 150.0)
    AMP_RANGE = (0.5, 2.0)
    MEASUREMENTS = ("heart_rate", "amplitude")
    RHYTHMS = ("brady", "normal", "tachy")
    LABELS = ("brady", "tachy", "high_amplitude")

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 64,
        leads: LeadSpec = None,
        seq_length: int = 5000,
        fs: int = 500,
        scale: Scale = "none",
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
        ranges = (self.HR_RANGE, self.AMP_RANGE)
        self.feature_normalizer = FeatureNormalizer(
            columns=list(self.MEASUREMENTS),
            mean={c: float(np.mean(r)) for c, r in zip(self.MEASUREMENTS, ranges)},
            std={
                c: float(np.ptp(r) / np.sqrt(12))
                for c, r in zip(self.MEASUREMENTS, ranges)
            },
        )

    def __len__(self) -> int:
        return self.num_samples

    @classmethod
    def render(
        cls, hr: float, amp: float, num_leads: int, seq_length: int, fs: int, rng
    ) -> np.ndarray:
        """
        The ``[C, T]`` signal for a heart rate (bpm) and R amplitude (mV).

        Each beat has P, Q, R, S and T waves with fixed widths in seconds (the
        QT interval shortens with the RR interval, Bazett-style); every lead is
        the same beat train times a random gain and sign, plus white noise.
        """
        rr = 60.0 / hr
        t = np.arange(seq_length) / fs - rng.uniform(0, rr)

        def wave(center: float, width: float) -> np.ndarray:
            dt = (t - center + rr / 2) % rr - rr / 2  # time to the nearest wave
            return np.exp(-(dt**2) / (2 * width**2))

        beat = (
            0.12 * wave(-0.16, 0.025)  # P
            - 0.15 * wave(-0.03, 0.008)  # Q
            + wave(0.0, 0.012)  # R
            - 0.25 * wave(0.03, 0.010)  # S
            + 0.30 * wave(0.30 * np.sqrt(rr), 0.045)  # T
        )
        gains = rng.uniform(0.5, 1.0, size=(num_leads, 1)) * rng.choice(
            [-1.0, 1.0], size=(num_leads, 1)
        )
        noise = 0.02 * rng.standard_normal((num_leads, seq_length))
        return (amp * gains * beat[None, :] + noise).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + _SPLIT_OFFSET[self.split] + idx)
        hr = rng.uniform(*self.HR_RANGE)
        amp = rng.uniform(*self.AMP_RANGE)
        ecg = self.render(hr, amp, len(self.lead_names), self.seq_length, self.fs, rng)

        features, missing = self.feature_normalizer.transform([hr, amp])
        rhythm = 0 if hr < 60 else (2 if hr > 100 else 1)
        labels = np.array([hr < 60, hr > 100, amp > 1.25], dtype=np.float32)
        return {
            "signal": torch.from_numpy(scale_signal(ecg, self.scale)),
            "cond": {
                "measurements": torch.from_numpy(features),
                "rhythm": torch.tensor(rhythm, dtype=torch.long),
                "labels": torch.from_numpy(labels),
            },
            "cond_missing": {"measurements": torch.from_numpy(missing)},
            "meta": {"index": idx, "heart_rate": float(hr), "amplitude": float(amp)},
        }
