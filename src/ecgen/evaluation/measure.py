"""
Measurements on (generated) ECGs, e.g. to check that a model follows the
heart rate it was asked for.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def detect_r_peaks(
    signal: ArrayLike,
    fs: int,
    lead: Optional[int] = None,
    method: str = "simple",
) -> np.ndarray:
    """
    R-peak sample indices of one ECG ``[C, T]`` (or ``[T]``).

    Args:
        signal: One ECG.
        fs: Sampling rate in Hz.
        lead: Lead to use; default = the lead with the largest QRS energy.
        method: ``"simple"`` (band-pass + peak picking, no extra dependency) or
            ``"neurokit"`` (``neurokit2.ecg_peaks``; needs ``ecgen[eval]``).
    """
    x = _as_numpy(signal)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] < int(0.5 * fs):
        return np.array([], dtype=int)

    nyq = fs / 2
    b, a = butter(2, [5 / nyq, min(30 / nyq, 0.99)], btype="band")
    filtered = filtfilt(b, a, x, axis=-1)
    if lead is None:
        lead = int(np.argmax(np.percentile(np.abs(filtered), 99, axis=-1)))
    sig = filtered[lead]

    if method == "neurokit":
        import neurokit2 as nk

        _, info = nk.ecg_peaks(x[lead], sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    if method != "simple":
        raise ValueError(f"Unknown method {method!r}")

    # QRS polarity varies by lead: use the dominant sign
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    height = 0.4 * np.percentile(sig, 99.5)
    peaks, _ = find_peaks(sig, height=height, distance=int(0.25 * fs))
    return peaks


def heart_rate(
    signals: ArrayLike,
    fs: int,
    method: str = "simple",
    lead: Optional[int] = None,
) -> np.ndarray:
    """
    Heart rate (bpm) per ECG from the median RR interval; NaN when fewer than
    two beats are found.

    Args:
        signals: ``[N, C, T]``, ``[C, T]``, or ``[T]``.
        fs: Sampling rate in Hz.
    """
    x = _as_numpy(signals)
    if x.ndim == 1:
        x = x[None, None]
    elif x.ndim == 2:
        x = x[None]
    rates: List[float] = []
    for ecg in x:
        peaks = detect_r_peaks(ecg, fs, lead=lead, method=method)
        if len(peaks) < 2:
            rates.append(np.nan)
        else:
            rates.append(60.0 * fs / float(np.median(np.diff(peaks))))
    return np.asarray(rates)


def adherence(requested: Sequence[float], measured: Sequence[float]) -> dict:
    """
    How well measured values follow requested ones (NaNs ignored).

    Returns ``{"mae", "pearson_r", "n_valid", "n"}``.
    """
    req = np.asarray(requested, dtype=np.float64)
    mea = np.asarray(measured, dtype=np.float64)
    ok = ~(np.isnan(req) | np.isnan(mea))
    out = {
        "n": int(len(req)),
        "n_valid": int(ok.sum()),
        "mae": float("nan"),
        "pearson_r": float("nan"),
    }
    if ok.sum() >= 2:
        out["mae"] = float(np.mean(np.abs(req[ok] - mea[ok])))
        if np.std(req[ok]) > 0 and np.std(mea[ok]) > 0:
            out["pearson_r"] = float(np.corrcoef(req[ok], mea[ok])[0, 1])
    return out
