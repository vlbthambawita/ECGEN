"""
MIMIC-IV-ECG dataset (dict samples).

Layout under ``root``:

- ``files/p{XXXX}/p{subject_id}/s{study_id}/{study_id}.hea/.dat`` (WFDB, 12 leads, 500 Hz, 10 s)
- ``machine_measurements.csv`` (one row per study; interval/axis measurements and report lines)

Records are stored in the lead order ``I, II, III, aVR, aVF, aVL, V1..V6`` — note
aVF before aVL, unlike the standard clinical order.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split

from ecgen.data.datasets.base import BaseECGDataset, Scale, scale_signal
from ecgen.data.leads import LeadSpec, lead_indices, resolve_leads
from ecgen.data.normalizers import FeatureNormalizer
from ecgen.registry import DATASETS

MIMIC_LEAD_ORDER: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVF",
    "aVL",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "rr_interval",
    "p_onset",
    "p_end",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "p_axis",
    "qrs_axis",
    "t_axis",
)


@DATASETS.register("mimic_iv_ecg")
class MIMICIVECG(BaseECGDataset):
    """
    MIMIC-IV-ECG records with machine measurements as a numeric condition field.

    Samples are dicts (see :class:`~ecgen.data.datasets.base.BaseECGDataset`) with
    ``cond["measurements"]`` = the normalized ``measurements`` columns.

    The train/val/test split is by subject and seeded, and matches the legacy
    ``MIMICIVECGDataset`` when ``missing="drop"`` (the default). The measurement
    normalizer is always fit on the *train* split, whichever split is loaded.

    Args:
        root: MIMIC-IV-ECG directory (contains ``files/`` and ``machine_measurements.csv``).
        split: ``"train"``, ``"val"``, or ``"test"``.
        val_split: Fraction of subjects in the validation split.
        test_split: Fraction of subjects in the test split.
        seed: Split seed.
        max_samples: Keep only the first N rows of the split (for quick runs).
        skip_missing_check: Skip the check that each record's ``.hea`` exists.
        leads: ``None`` (file order, padded/truncated to ``num_leads``), an alias
            (``"standard12"``, ``"independent8"``), or a list of lead names.
        num_leads: Channel count when ``leads`` is ``None``.
        seq_length: Samples per record (padded/truncated).
        scale: ``"zscore"`` (per record, legacy), ``"none"`` (mV), or a number
            (subtract record mean, divide by it).
        measurements: Columns for ``cond["measurements"]``; ``[]`` for none.
        missing: ``"drop"`` rows with any missing measurement (legacy) or
            ``"mask"`` to keep them and report the gaps in ``cond_missing``.
            ``"mask"`` keeps more subjects, so its split differs from ``"drop"``.
        feature_normalizer: Use these statistics instead of fitting on train
            (e.g. those stored in a checkpoint).
        report: Add ``cond["report"]``: the machine report lines joined by ". ".
    """

    fs = 500

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
        skip_missing_check: bool = False,
        leads: LeadSpec = None,
        num_leads: int = 12,
        seq_length: int = 5000,
        scale: Scale = "zscore",
        ecg_norm_eps: float = 1e-6,
        measurements: Optional[Sequence[str]] = None,
        missing: str = "drop",
        feature_normalizer: Optional[FeatureNormalizer | Dict[str, Any]] = None,
        report: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if missing not in ("drop", "mask"):
            raise ValueError(f"missing must be 'drop' or 'mask', got {missing!r}")

        self.root = root
        self.split = split
        self.seed = seed
        self.seq_length = seq_length
        self.scale = scale
        self.ecg_norm_eps = ecg_norm_eps
        self.use_report = report
        self.measurement_columns: List[str] = list(
            MEASUREMENT_COLUMNS if measurements is None else measurements
        )

        self._leads_spec = leads
        if leads is None:
            self.lead_names = list(MIMIC_LEAD_ORDER[:num_leads]) + [
                f"pad{i}" for i in range(max(0, num_leads - len(MIMIC_LEAD_ORDER)))
            ]
        else:
            self.lead_names = resolve_leads(leads, MIMIC_LEAD_ORDER)

        table = self._load_measurements(missing)
        table["split"] = self._assign_splits(table, val_split, test_split)

        if feature_normalizer is None:
            self.feature_normalizer = FeatureNormalizer.fit(
                table[table["split"] == "train"], self.measurement_columns
            )
        elif isinstance(feature_normalizer, dict):
            self.feature_normalizer = FeatureNormalizer.from_dict(feature_normalizer)
        else:
            self.feature_normalizer = feature_normalizer

        self.records = table[table["split"] == split].reset_index(drop=True)

        if skip_missing_check:
            warnings.warn(
                "Skipping missing file check. Some samples may fail during loading.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._filter_missing_files()

        if max_samples is not None:
            self.records = self.records.head(max_samples).reset_index(drop=True)

    # ------------------------------------------------------------------ setup

    def _load_measurements(self, missing: str) -> pd.DataFrame:
        path = os.path.join(self.root, "machine_measurements.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"machine_measurements.csv not found at {path}. "
                "Download from https://physionet.org/content/mimic-iv-ecg/1.0/"
            )
        table = pd.read_csv(path, low_memory=False)

        # The legacy loader always required (and dropped NaNs in) the 9 standard
        # columns, which also decides which subjects enter the split.
        required = (
            ["subject_id", "study_id"]
            + list(MEASUREMENT_COLUMNS)
            + self.measurement_columns
        )
        missing_cols = sorted({c for c in required if c not in table.columns})
        if missing_cols:
            raise ValueError(
                f"machine_measurements.csv missing columns: {missing_cols}"
            )

        if missing == "drop":
            subset = list(
                dict.fromkeys(list(MEASUREMENT_COLUMNS) + self.measurement_columns)
            )
            table = table.dropna(subset=subset)
        return table.reset_index(drop=True)

    def _assign_splits(
        self, table: pd.DataFrame, val_split: float, test_split: float
    ) -> pd.Series:
        subjects = table["subject_id"].unique()
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_split, random_state=self.seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=val_split / (1 - test_split),
            random_state=self.seed,
        )
        split = pd.Series("train", index=table.index)
        split[table["subject_id"].isin(set(val_subjects))] = "val"
        split[table["subject_id"].isin(set(test_subjects))] = "test"
        return split

    def record_path(self, subject_id: int, study_id: int) -> str:
        sub = str(subject_id)
        return os.path.join(
            self.root, "files", f"p{sub[:4]}", f"p{sub}", f"s{study_id}", str(study_id)
        )

    def _filter_missing_files(self) -> None:
        if not os.path.isdir(os.path.join(self.root, "files")):
            return
        exists = [
            os.path.isfile(self.record_path(s, t) + ".hea")
            for s, t in zip(self.records["subject_id"], self.records["study_id"])
        ]
        self.records = self.records[np.asarray(exists, dtype=bool)].reset_index(
            drop=True
        )

    # ---------------------------------------------------------------- samples

    def load_ecg(self, idx: int) -> np.ndarray:
        """Raw ``[C, T]`` signal in mV, with the configured leads and length."""
        row = self.records.iloc[idx]
        record = wfdb.rdrecord(self.record_path(row["subject_id"], row["study_id"]))
        signal = record.p_signal.T.astype(np.float32)  # (leads, time)

        if self._leads_spec is None:
            n = len(self.lead_names)
            if signal.shape[0] < n:
                signal = np.vstack(
                    [
                        signal,
                        np.zeros(
                            (n - signal.shape[0], signal.shape[1]), dtype=np.float32
                        ),
                    ]
                )
            signal = signal[:n]
        else:
            signal = signal[lead_indices(self.lead_names, record.sig_name)]

        if signal.shape[1] < self.seq_length:
            pad = np.zeros(
                (signal.shape[0], self.seq_length - signal.shape[1]), dtype=np.float32
            )
            signal = np.hstack([signal, pad])
        return signal[:, : self.seq_length]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ecg = scale_signal(self.load_ecg(idx), self.scale, self.ecg_norm_eps)
        row = self.records.iloc[idx]
        sample: Dict[str, Any] = {
            "signal": torch.from_numpy(ecg),
            "cond": {},
            "cond_missing": {},
            "meta": {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
            },
        }
        if self.use_report:
            lines = [row.get(f"report_{i}") for i in range(18)]
            sample["cond"]["report"] = ". ".join(
                str(x).strip() for x in lines if isinstance(x, str) and x.strip()
            )
        if self.measurement_columns:
            raw = pd.to_numeric(
                row[self.measurement_columns], errors="coerce"
            ).to_numpy(dtype=np.float64)
            values, missing = self.feature_normalizer.transform(raw)
            sample["cond"]["measurements"] = torch.from_numpy(values)
            sample["cond_missing"]["measurements"] = torch.from_numpy(missing)
        return sample
