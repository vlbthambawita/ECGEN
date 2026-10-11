"""
PTB-XL (Wagner et al., 2020; PhysioNet ``ptb-xl`` 1.0.x), optionally with
PTB-XL+ features (Strodthoff et al., 2023).

Layout under ``root``: ``ptbxl_database.csv``, ``scp_statements.csv``,
``records500/…/{id}_hr`` (500 Hz, 10 s) and ``records100/…/{id}_lr`` (100 Hz).
Leads are stored in the standard order ``I, II, III, aVR, aVL, aVF, V1–V6``
(the files write ``AVR``/``AVL``/``AVF``).
"""

from __future__ import annotations

import ast
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import wfdb

from ecgen.data.datasets.base import BaseECGDataset, Scale, scale_signal
from ecgen.data.leads import STANDARD_12, LeadSpec, lead_indices, resolve_leads
from ecgen.data.normalizers import FeatureNormalizer
from ecgen.registry import DATASETS

#: The recommended split of the 10 stratified folds.
DEFAULT_FOLDS = {"train": list(range(1, 9)), "val": [9], "test": [10]}

#: PTB-XL+ 12SL global measurements used by default for ``cond["features"]``.
PTBXL_PLUS_FEATURES = (
    "HR_Ventr_Global",
    "PR_Int_Global",
    "QRS_Dur_Global",
    "QT_Int_Global",
    "QT_IntBazett_Global",
    "P_AxisFront_Global",
    "R_AxisFrontal_Global",
    "T_AxisFront_Global",
)

DEMOGRAPHICS = ("age", "height", "weight")


def _lead_name(raw: str) -> str:
    return {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(raw, raw)


@DATASETS.register("ptbxl")
class PTBXL(BaseECGDataset):
    """
    PTB-XL records with diagnostic statements, demographics, and reports.

    Condition fields (each can be switched off):

    - ``scp``: multi-hot statements; ``labels`` selects the vocabulary:
      ``"all"`` (71 statements, as in SSSD-ECG), ``"superclass"`` (NORM, MI, STTC,
      CD, HYP), ``"subclass"`` (23), ``"diagnostic"``, ``"form"``, ``"rhythm"``.
      :attr:`label_names` lists the order.
    - ``demographics``: normalized ``[age, height, weight]`` with a missing mask
      (height and weight are often missing).
    - ``sex``: class 0 = male, 1 = female (PTB-XL coding).
    - ``report``: the free-text report (mostly German) as a string.
    - ``features`` (with ``ptbxl_plus_root``): PTB-XL+ 12SL measurements
      (default :data:`PTBXL_PLUS_FEATURES`), normalized with a missing mask.

    Normalizers are fit on the training folds. Args mirror
    :class:`~ecgen.data.datasets.mimic.MIMICIVECG` where they overlap.

    Args:
        root: PTB-XL directory.
        split: ``"train"`` / ``"val"`` / ``"test"``.
        folds: Map split -> list of ``strat_fold`` values (default 1–8 / 9 / 10).
        sampling_rate: 500 or 100 Hz.
        labels: Statement vocabulary for ``scp`` (``None`` = no ``scp`` field).
        min_likelihood: Keep a statement only if its likelihood is at least this
            (PTB-XL stores 0–100; 0 keeps every listed statement).
        demographics, sex, report: Include those fields.
        ptbxl_plus_root: PTB-XL+ directory (adds ``features``).
        feature_columns: PTB-XL+ columns for ``features``.
        validated_only: Keep only records with ``validated_by_human``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[Dict[str, Sequence[int]]] = None,
        sampling_rate: int = 500,
        max_samples: Optional[int] = None,
        leads: LeadSpec = None,
        seq_length: Optional[int] = None,
        scale: Scale = "zscore",
        ecg_norm_eps: float = 1e-6,
        labels: Optional[str] = "all",
        min_likelihood: float = 0.0,
        demographics: bool = True,
        sex: bool = True,
        report: bool = True,
        ptbxl_plus_root: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
        validated_only: bool = False,
        field_normalizers: Optional[Dict[str, Any]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        self.root = root
        self.split = split
        self.fs = sampling_rate
        self.seq_length = seq_length or 10 * sampling_rate
        self.scale = scale
        self.ecg_norm_eps = ecg_norm_eps
        self.lead_names = resolve_leads(leads, STANDARD_12)
        self.use_demographics, self.use_sex, self.use_report = demographics, sex, report

        db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
        if validated_only:
            db = db[db["validated_by_human"]]
        folds = folds or DEFAULT_FOLDS

        # ---- statement vocabulary
        self.label_names: List[str] = []
        self._scp_map: Dict[str, List[str]] = {}
        if labels is not None:
            self._build_vocabulary(labels)
        self.min_likelihood = min_likelihood

        # ---- PTB-XL+ features
        self.feature_columns: List[str] = []
        if ptbxl_plus_root:
            self.feature_columns = list(feature_columns or PTBXL_PLUS_FEATURES)
            feats = pd.read_csv(
                os.path.join(ptbxl_plus_root, "features", "12sl_features.csv"),
                usecols=["ecg_id", *self.feature_columns],
                index_col="ecg_id",
            )
            db = db.join(feats, how="left")

        # ---- normalizers (training folds)
        train_rows = db[db["strat_fold"].isin(folds["train"])]
        stored = field_normalizers or {}
        self._normalizers: Dict[str, FeatureNormalizer] = {}
        if demographics:
            self._normalizers["demographics"] = _norm(
                stored.get("demographics"), train_rows, DEMOGRAPHICS
            )
        if self.feature_columns:
            self._normalizers["features"] = _norm(
                stored.get("features"), train_rows, self.feature_columns
            )

        self.records = db[db["strat_fold"].isin(folds[split])].reset_index()
        if max_samples is not None:
            self.records = self.records.head(max_samples).reset_index(drop=True)
        self._file_col = "filename_hr" if sampling_rate == 500 else "filename_lr"

    def _build_vocabulary(self, labels: str) -> None:
        stmts = pd.read_csv(os.path.join(self.root, "scp_statements.csv"), index_col=0)
        if labels == "all":
            names = list(stmts.index)
            self._scp_map = {s: [s] for s in names}
        elif labels in ("superclass", "subclass"):
            col = (
                "diagnostic_class" if labels == "superclass" else "diagnostic_subclass"
            )
            diag = stmts[stmts["diagnostic"] == 1]
            names = sorted(diag[col].dropna().unique())
            self._scp_map = {
                s: [diag.loc[s, col]] for s in diag.index if pd.notna(diag.loc[s, col])
            }
        elif labels in ("diagnostic", "form", "rhythm"):
            names = list(stmts[stmts[labels] == 1].index)
            self._scp_map = {s: [s] for s in names}
        else:
            raise ValueError(
                f"labels must be all, superclass, subclass, diagnostic, form, rhythm or None; got {labels!r}"
            )
        self.label_names = list(names)
        self._label_index = {n: i for i, n in enumerate(self.label_names)}

    @property
    def field_normalizers(self) -> Dict[str, FeatureNormalizer]:
        return dict(self._normalizers)

    def __len__(self) -> int:
        return len(self.records)

    def load_ecg(self, idx: int) -> np.ndarray:
        row = self.records.iloc[idx]
        record = wfdb.rdrecord(os.path.join(self.root, row[self._file_col]))
        names = [_lead_name(n) for n in record.sig_name]
        signal = record.p_signal.T.astype(np.float32)[
            lead_indices(self.lead_names, names)
        ]
        if signal.shape[1] < self.seq_length:
            signal = np.pad(signal, ((0, 0), (0, self.seq_length - signal.shape[1])))
        signal = signal[:, : self.seq_length]
        if np.isnan(signal).any():
            warnings.warn(
                f"NaN in PTB-XL record {row['ecg_id']}; set to 0",
                RuntimeWarning,
                stacklevel=2,
            )
            signal = np.nan_to_num(signal)
        return signal

    def multi_hot(self, scp_codes: Dict[str, float]) -> np.ndarray:
        y = np.zeros(len(self.label_names), dtype=np.float32)
        for code, likelihood in scp_codes.items():
            if likelihood < self.min_likelihood:
                continue
            for name in self._scp_map.get(code, []):
                y[self._label_index[name]] = 1.0
        return y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        ecg = scale_signal(self.load_ecg(idx), self.scale, self.ecg_norm_eps)
        cond: Dict[str, Any] = {}
        missing: Dict[str, Any] = {}
        if self.label_names:
            cond["scp"] = torch.from_numpy(self.multi_hot(row["scp_codes"]))
        for field, columns in (
            ("demographics", DEMOGRAPHICS),
            ("features", self.feature_columns),
        ):
            if field in self._normalizers:
                raw = pd.to_numeric(row[list(columns)], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                values, miss = self._normalizers[field].transform(raw)
                cond[field] = torch.from_numpy(values)
                missing[field] = torch.from_numpy(miss)
        if self.use_sex:
            cond["sex"] = torch.tensor(int(row["sex"]), dtype=torch.long)
        if self.use_report:
            cond["report"] = str(row["report"]) if pd.notna(row["report"]) else ""
        return {
            "signal": torch.from_numpy(ecg),
            "cond": cond,
            "cond_missing": missing,
            "meta": {
                "ecg_id": int(row["ecg_id"]),
                "patient_id": int(row["patient_id"]),
                "strat_fold": int(row["strat_fold"]),
            },
        }


def _norm(
    stored: Optional[Dict[str, Any]], train_rows: pd.DataFrame, columns: Sequence[str]
) -> FeatureNormalizer:
    if stored is not None:
        return (
            FeatureNormalizer.from_dict(stored) if isinstance(stored, dict) else stored
        )
    return FeatureNormalizer.fit(train_rows, list(columns))
