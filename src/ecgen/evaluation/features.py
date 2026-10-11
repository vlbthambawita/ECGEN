"""
Feature extractors for distribution metrics (Fréchet distance, MMD,
precision/recall, nearest neighbours).

There is no single pretrained ECG encoder that every user has, so the default
is a set of interpretable hand-crafted features; any ``torch.nn.Module`` that
maps ``[B, C, T]`` to ``[B, D]`` (e.g. the penultimate layer of an ECG
classifier, see :mod:`ecgen.evaluation.classifier`) can be used instead.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
from scipy.signal import welch

from ecgen.evaluation.measure import _as_numpy, heart_rate

ArrayLike = Union[np.ndarray, torch.Tensor]
Extractor = Callable[[ArrayLike], np.ndarray]

BANDS = ((0.5, 5.0), (5.0, 15.0), (15.0, 40.0), (40.0, 100.0))


def handcrafted_features(x: ArrayLike, fs: int = 500) -> np.ndarray:
    """
    Per-ECG feature vector ``[N, D]``:

    - heart rate (bpm, NaN → batch median);
    - per lead: standard deviation, 1st/99th percentile, and the relative power
      in the bands 0.5–5, 5–15, 15–40, 40–100 Hz;
    - the upper triangle of the inter-lead correlation matrix.
    """
    x = _as_numpy(x)
    if x.ndim == 2:
        x = x[None]
    n, c, t = x.shape
    hr = heart_rate(x, fs)
    hr = np.where(np.isnan(hr), np.nanmedian(hr) if np.isfinite(hr).any() else 0.0, hr)

    std = x.std(axis=-1)
    p1, p99 = np.percentile(x, 1, axis=-1), np.percentile(x, 99, axis=-1)
    freqs, psd = welch(x, fs=fs, nperseg=min(t, 2 * fs), axis=-1)
    total = psd.sum(-1, keepdims=True) + 1e-12
    bands = (
        np.stack(
            [
                psd[..., (freqs >= lo) & (freqs < min(hi, fs / 2))].sum(-1)
                for lo, hi in BANDS
            ],
            axis=-1,
        )
        / total
    )
    iu = np.triu_indices(c, k=1)
    corr = (
        np.stack([np.nan_to_num(np.corrcoef(e))[iu] for e in x])
        if c > 1
        else np.zeros((n, 0))
    )
    feats = [hr[:, None], std, p1, p99, bands.reshape(n, -1), corr]
    return np.concatenate(feats, axis=1).astype(np.float64)


def module_features(
    model: torch.nn.Module, batch_size: int = 64, device: Optional[str] = None
) -> Extractor:
    """Wrap a ``[B, C, T] -> [B, D]`` network as an extractor."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def extract(x: ArrayLike) -> np.ndarray:
        xt = torch.as_tensor(_as_numpy(x), dtype=torch.float32)
        model.eval().to(device)
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), batch_size):
                out.append(
                    model(xt[i : i + batch_size].to(device)).float().cpu().numpy()
                )
        return np.concatenate(out)

    return extract
