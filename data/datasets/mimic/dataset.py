"""
Native MIMIC-IV-ECG dataset implementation.

Dataset structure:
- ECG waveforms: files/p{XXXX}/p{subject_id}/s{study_id}/{study_id}.hea/.dat (WFDB format)
- Machine measurements: machine_measurements.csv

Expected columns in machine_measurements.csv:
- subject_id, study_id
- rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end
- p_axis, qrs_axis, t_axis
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MIMICIVECGDataset(Dataset):
    """
    Dataset for MIMIC-IV-ECG signals with machine measurements conditioning.

    Returns (ecg, features) where:
    - ecg: (num_leads, seq_length) normalized ECG signal, float32
    - features: (9,) normalized machine measurements, float32
    """

    FEATURE_NAMES = [
        "rr_interval",
        "p_onset",
        "p_end",
        "qrs_onset",
        "qrs_end",
        "t_end",
        "p_axis",
        "qrs_axis",
        "t_axis",
    ]

    def __init__(
        self,
        mimic_path: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        skip_missing_check: bool = False,
        ecg_norm_eps: float = 1e-6,
        ecg_norm_factor: Optional[float] = None,
        num_leads: int = 12,
        seq_length: int = 5000,
    ) -> None:
        self.mimic_path = mimic_path
        self.ecg_norm_eps = ecg_norm_eps
        self.ecg_norm_factor = ecg_norm_factor
        self.split = split
        self.seed = seed
        self.skip_missing_check = skip_missing_check
        self.num_leads = num_leads
        self.seq_length = seq_length

        self.load_measurements()
        self.create_splits(val_split, test_split)
        self.filter_by_split()

        if not skip_missing_check:
            self.filter_missing_files()
        else:
            import warnings
            warnings.warn(
                "Skipping missing file check. Some samples may fail during loading.",
                UserWarning,
                stacklevel=2,
            )

        if max_samples is not None:
            self.measurements = self.measurements.head(max_samples).reset_index(drop=True)

        self.compute_feature_stats()

    def load_measurements(self) -> None:
        path = os.path.join(self.mimic_path, "machine_measurements.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"machine_measurements.csv not found at {path}. "
                "Download from https://physionet.org/content/mimic-iv-ecg/1.0/"
            )
        self.measurements = pd.read_csv(path)

        required = ["subject_id", "study_id"] + self.FEATURE_NAMES
        missing = [c for c in required if c not in self.measurements.columns]
        if missing:
            raise ValueError(f"machine_measurements.csv missing columns: {missing}")

        self.measurements = self.measurements.dropna(subset=self.FEATURE_NAMES).reset_index(drop=True)

    def create_splits(self, val_split: float, test_split: float) -> None:
        subjects = self.measurements["subject_id"].unique()

        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_split, random_state=self.seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=val_split / (1 - test_split),
            random_state=self.seed,
        )

        def assign_split(row: pd.Series) -> str:
            sid = row["subject_id"]
            if sid in val_subjects:
                return "val"
            if sid in test_subjects:
                return "test"
            return "train"

        self.measurements["split"] = self.measurements.apply(assign_split, axis=1)

    def filter_by_split(self) -> None:
        self.measurements = self.measurements[
            self.measurements["split"] == self.split
        ].reset_index(drop=True)

    def filter_missing_files(self) -> None:
        files_dir = os.path.join(self.mimic_path, "files")
        if not os.path.isdir(files_dir):
            return

        valid = []
        for idx in range(len(self.measurements)):
            row = self.measurements.iloc[idx]
            rec_path = self._ecg_record_path(row["subject_id"], row["study_id"])
            if os.path.isfile(rec_path + ".hea"):
                valid.append(idx)

        self.measurements = self.measurements.iloc[valid].reset_index(drop=True)

    def _ecg_record_path(self, subject_id: int, study_id: int) -> str:
        sub_str = str(subject_id)
        prefix = sub_str[:4]
        return os.path.join(
            self.mimic_path,
            "files",
            f"p{prefix}",
            f"p{subject_id}",
            f"s{study_id}",
            str(study_id),
        )

    def compute_feature_stats(self) -> None:
        self.feature_stats = {}
        for name in self.FEATURE_NAMES:
            vals = self.measurements[name].values
            self.feature_stats[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) + 1e-6,
            }

    def load_ecg(self, idx: int) -> np.ndarray:
        row = self.measurements.iloc[idx]
        rec_path = self._ecg_record_path(row["subject_id"], row["study_id"])

        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal  # (time, leads)
        signal = signal.T.astype(np.float32)  # (leads, time)

        if signal.shape[0] < self.num_leads:
            pad = np.zeros((self.num_leads - signal.shape[0], signal.shape[1]), dtype=np.float32)
            signal = np.vstack([signal, pad])
        elif signal.shape[0] > self.num_leads:
            signal = signal[: self.num_leads]

        if signal.shape[1] < self.seq_length:
            pad = np.zeros((signal.shape[0], self.seq_length - signal.shape[1]), dtype=np.float32)
            signal = np.hstack([signal, pad])
        elif signal.shape[1] > self.seq_length:
            signal = signal[:, : self.seq_length]

        return signal

    def _get_features(self, idx: int) -> np.ndarray:
        row = self.measurements.iloc[idx]
        out = []
        for name in self.FEATURE_NAMES:
            val = row[name]
            s = self.feature_stats[name]
            out.append((float(val) - s["mean"]) / s["std"])
        return np.array(out, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ecg = self.load_ecg(idx)

        ecg_mean = ecg.mean()
        if self.ecg_norm_factor is not None:
            scale = self.ecg_norm_factor
        else:
            scale = max(float(np.std(ecg)), self.ecg_norm_eps)
        ecg = (ecg.astype(np.float32) - ecg_mean) / scale

        features = self._get_features(idx)

        return torch.from_numpy(ecg), torch.from_numpy(features)
