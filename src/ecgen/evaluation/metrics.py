"""
Metrics for generated ECGs (plan §8): distribution fidelity and diversity,
condition adherence, partial-ECG reconstruction, lead consistency, and
memorisation.

Arrays are ``[N, C, T]`` (NumPy or torch). Distribution metrics work on feature
vectors from an extractor (:mod:`ecgen.evaluation.features`).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import linalg

from ecgen.evaluation.features import ArrayLike, handcrafted_features
from ecgen.evaluation.measure import _as_numpy

# ------------------------------------------------------------ distributions


def frechet_distance(
    feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6
) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets (FID-style)."""
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    sa, sb = (
        np.atleast_2d(np.cov(feats_a, rowvar=False)),
        np.atleast_2d(np.cov(feats_b, rowvar=False)),
    )
    covmean, _ = linalg.sqrtm(sa @ sb, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sa.shape[0]) * eps
        covmean = linalg.sqrtm((sa + off) @ (sb + off))
    covmean = covmean.real
    return float(
        ((mu_a - mu_b) ** 2).sum() + np.trace(sa) + np.trace(sb) - 2 * np.trace(covmean)
    )


def mmd_rbf(
    feats_a: np.ndarray, feats_b: np.ndarray, bandwidth: Optional[float] = None
) -> float:
    """Unbiased squared MMD with an RBF kernel (bandwidth = median heuristic)."""
    x, y = np.asarray(feats_a, float), np.asarray(feats_b, float)
    z = np.concatenate([x, y])
    d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    if bandwidth is None:
        bandwidth = np.sqrt(0.5 * np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
    k = np.exp(-d2 / (2 * bandwidth**2))
    n, m = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2 * kxy.mean()
    )


def precision_recall(
    real: np.ndarray, fake: np.ndarray, k: int = 3
) -> Dict[str, float]:
    """
    k-NN manifold precision / recall (Kynkäänniemi et al., 2019): precision = share
    of generated samples inside the real manifold, recall = share of real samples
    inside the generated manifold.
    """

    def radii(a: np.ndarray) -> np.ndarray:
        d = np.sqrt(((a[:, None] - a[None]) ** 2).sum(-1))
        return np.sort(d, axis=1)[:, min(k, len(a) - 1)]

    def inside(points: np.ndarray, ref: np.ndarray, r: np.ndarray) -> float:
        d = np.sqrt(((points[:, None] - ref[None]) ** 2).sum(-1))
        return float((d <= r[None]).any(axis=1).mean())

    return {
        "precision": inside(fake, real, radii(real)),
        "recall": inside(real, fake, radii(fake)),
    }


def _standardize(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = a.mean(0), a.std(0) + 1e-8
    return (a - mu) / sd, (b - mu) / sd


def distribution_metrics(
    real: ArrayLike,
    fake: ArrayLike,
    fs: int = 500,
    extractor: Optional[Callable[[ArrayLike], np.ndarray]] = None,
    k: int = 3,
) -> Dict[str, float]:
    """
    Fréchet distance, MMD and precision/recall between real and generated ECGs on
    (standardized) features; default features: :func:`handcrafted_features`.
    """
    extract = extractor or (lambda x: handcrafted_features(x, fs))
    fr, ff = extract(real), extract(fake)
    fr, ff = _standardize(fr, ff)
    out = {"frechet_distance": frechet_distance(fr, ff), "mmd": mmd_rbf(fr, ff)}
    out.update(precision_recall(fr, ff, k))
    return out


# ------------------------------------------------------- lead consistency


def inter_lead_correlation_error(real: ArrayLike, fake: ArrayLike) -> float:
    """Mean |corr_real − corr_fake| over lead pairs, using mean correlation matrices (MIDT-ECG)."""

    def mean_corr(x: np.ndarray) -> np.ndarray:
        return np.nanmean(np.stack([np.corrcoef(e) for e in x]), axis=0)

    r, f = mean_corr(_as_numpy(real)), mean_corr(_as_numpy(fake))
    iu = np.triu_indices(r.shape[0], k=1)
    return float(np.abs(r[iu] - f[iu]).mean())


def einthoven_residual(x: ArrayLike, lead_names: Sequence[str]) -> Dict[str, float]:
    """
    Relative residuals of the limb-lead identities (III = II − I,
    aVR = −(I + II)/2, aVL = I − II/2, aVF = II − I/2) after per-lead centering;
    ~0 for real recordings, larger when a model generates all 12 leads independently.
    """
    x = _as_numpy(x)
    x = x - x.mean(-1, keepdims=True)
    idx = {n: i for i, n in enumerate(lead_names)}
    need = ("I", "II", "III", "aVR", "aVL", "aVF")
    if not all(n in idx for n in need):
        raise ValueError(
            f"einthoven_residual needs leads {need}; got {list(lead_names)}"
        )
    i, ii = x[:, idx["I"]], x[:, idx["II"]]
    expected = {
        "III": ii - i,
        "aVR": -(i + ii) / 2,
        "aVL": i - ii / 2,
        "aVF": ii - i / 2,
    }
    out = {}
    for name, exp in expected.items():
        actual = x[:, idx[name]]
        out[name] = float(
            np.linalg.norm(actual - exp) / (np.linalg.norm(actual) + 1e-12)
        )
    out["mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------- reconstruction


def reconstruction_metrics(
    pred: ArrayLike, true: ArrayLike, unobserved: ArrayLike
) -> Dict[str, float]:
    """
    MAE, RMSE and mean Pearson r on the unobserved entries (``unobserved``:
    ``[N, C, T]`` or ``[C]`` / ``[N, C]`` bool, True = to evaluate).
    """
    p, t = _as_numpy(pred), _as_numpy(true)
    m = np.asarray(_as_numpy(unobserved), dtype=bool)
    if m.ndim == 1:
        m = np.broadcast_to(m[None, :, None], p.shape)
    elif m.ndim == 2:
        m = np.broadcast_to(m[:, :, None], p.shape)
    err = (p - t)[m]
    rs = []
    for n in range(p.shape[0]):
        for c in range(p.shape[1]):
            sel = m[n, c]
            if sel.sum() > 2 and p[n, c][sel].std() > 0 and t[n, c][sel].std() > 0:
                rs.append(np.corrcoef(p[n, c][sel], t[n, c][sel])[0, 1])
    return {
        "mae": float(np.abs(err).mean()) if err.size else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())) if err.size else float("nan"),
        "pearson_r": float(np.mean(rs)) if rs else float("nan"),
    }


# --------------------------------------------------------- memorisation


def nearest_neighbour_distances(
    query: ArrayLike, reference: ArrayLike, batch: int = 256
) -> np.ndarray:
    """L2 distance from each query ECG to its nearest reference ECG (flattened signals)."""
    q = _as_numpy(query).reshape(len(_as_numpy(query)), -1)
    r = _as_numpy(reference).reshape(len(_as_numpy(reference)), -1)
    r2 = (r**2).sum(1)
    out = []
    for i in range(0, len(q), batch):
        qb = q[i : i + batch]
        d2 = (qb**2).sum(1)[:, None] + r2[None] - 2 * qb @ r.T
        out.append(np.sqrt(np.clip(d2.min(1), 0, None)))
    return np.concatenate(out)


def memorization_ratio(
    fake: ArrayLike, train: ArrayLike, real_holdout: ArrayLike
) -> Dict[str, float]:
    """
    Median nearest-neighbour distance of generated ECGs to the training set,
    divided by that of held-out real ECGs. Values well below 1 suggest copying.
    """
    d_fake = nearest_neighbour_distances(fake, train)
    d_real = nearest_neighbour_distances(real_holdout, train)
    return {
        "nn_fake_median": float(np.median(d_fake)),
        "nn_real_median": float(np.median(d_real)),
        "ratio": float(np.median(d_fake) / (np.median(d_real) + 1e-12)),
    }
