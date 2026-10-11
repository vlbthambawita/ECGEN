"""
Public sampling API: ``ecgen.sample(checkpoint, n, ...)`` for any checkpoint
trained with ``ecgen train``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch

from ecgen.registry import FAMILIES, load_builtins

PathLike = Union[str, Path]


def load_model(
    checkpoint: PathLike, family: Optional[str] = None, device: Optional[str] = None
):
    """
    Load a generative model from an ``ecgen train`` checkpoint.

    The family is read from the checkpoint (``ecgen_family``); pass ``family``
    for checkpoints written before it was recorded. ``hf://<repo>/<path>``
    checkpoints are downloaded (see :func:`ecgen.generate`).
    """
    load_builtins()
    from ecgen.generate import _resolve_model_path

    path = _resolve_model_path(checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if family is None:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        family = meta.get("ecgen_family")
        if family is None:
            raise ValueError(
                f"{path} does not record its family; pass family= (one of {', '.join(FAMILIES)})"
            )
    model = FAMILIES.get(family).load_from_checkpoint(str(path), map_location=device)
    return model.to(device).eval()


def _per_sample(value: Any, n: int) -> bool:
    """
    Whether a condition value holds one entry per sample: a list of ``n``
    strings, an array whose first of >= 2 dims is ``n``, or a 1-D integer array
    of length ``n`` (class indices). Anything else (a numeric vector, one string)
    is broadcast to all samples.
    """
    if isinstance(value, str):
        return False
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, str) for v in value)
    ):
        return len(value) == n
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
    else:
        return False
    if arr.ndim >= 2:
        return arr.shape[0] == n
    return arr.ndim == 1 and arr.dtype.kind in "iu" and arr.shape[0] == n


def _slice(value: Any, start: int, stop: int, n: int) -> Any:
    if not _per_sample(value, n):
        return value
    if isinstance(value, (list, tuple)):
        return list(value[start:stop])
    return value[start:stop]


def sample(
    checkpoint: Union[PathLike, Any],
    n: int = 16,
    *,
    family: Optional[str] = None,
    cond: Optional[Mapping[str, Any]] = None,
    obs: Optional[Any] = None,
    obs_mask: Optional[Any] = None,
    guidance: Optional[float] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    derive_12: bool = False,
    output_dir: Optional[PathLike] = None,
    format: Optional[str] = "npy",
    plot: int = 0,
    **sample_kwargs: Any,
) -> np.ndarray:
    """
    Generate ``n`` ECGs from a checkpoint (or an already loaded model).

    Args:
        checkpoint: ``.ckpt`` path, ``hf://`` path, or a loaded ``GenerativeModule``.
        n: Number of ECGs.
        family: Registry name, for checkpoints that do not record it.
        cond: Raw condition values (numeric fields in raw units; NaN = unknown;
            strings for text fields). A value with a leading dimension of ``n``
            (or a list of ``n`` strings) gives one condition per sample.
        obs: Partially observed ECGs ``[n, C, T]``.
        obs_mask: Masks of ``obs`` (1 = observed).
        guidance: Classifier-free guidance scale (conditional diffusion / flow / prior).
        steps: Sampling steps (diffusion / flow).
        seed: Seed for reproducible sampling.
        batch_size: ECGs generated per forward pass.
        device: Default: CUDA if available.
        derive_12: Complete 8 independent leads (I, II, V1–V6) to the standard 12.
        output_dir: Write the samples here (``format``: ``npy``, ``csv``, ``wfdb``, or
            ``None`` to skip) plus ``samples.json`` metadata.
        plot: Render the first ``plot`` samples as interactive HTML (``ecgen.plot_ecg``).
        sample_kwargs: Passed to the model's ``sample`` (e.g. ``eta``, ``solver``,
            ``inpaint``, ``temperature``, ``seq_length``).

    Returns:
        ``[n, C, T]`` float32 array.
    """
    model = (
        checkpoint
        if hasattr(checkpoint, "sample") and not isinstance(checkpoint, (str, Path))
        else load_model(checkpoint, family, device)
    )
    info: Dict[str, Any] = dict(getattr(model, "data_info", {}) or {})
    params = inspect.signature(model.sample).parameters
    extra: Dict[str, Any] = dict(sample_kwargs)
    if guidance is not None:
        extra["guidance"] = guidance
    if steps is not None:
        extra["steps"] = steps
    if "seq_length" in params and "seq_length" not in extra and info.get("seq_length"):
        extra["seq_length"] = info["seq_length"]
    if seed is not None:
        torch.manual_seed(seed)

    out: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        kw = dict(extra)
        if cond:
            kw["cond"] = {k: _slice(v, start, stop, n) for k, v in cond.items()}
        if obs is not None:
            kw["obs"] = torch.as_tensor(
                np.asarray(obs) if not isinstance(obs, torch.Tensor) else obs
            )[start:stop]
            if obs_mask is not None:
                m = (
                    obs_mask
                    if isinstance(obs_mask, torch.Tensor)
                    else torch.as_tensor(np.asarray(obs_mask))
                )
                kw["obs_mask"] = m[start:stop]
        if seed is not None and "seed" in params:
            kw["seed"] = seed + start
        with torch.no_grad():
            x = model.sample(stop - start, **kw)
        out.append(x.detach().float().cpu().numpy())
    samples = np.concatenate(out).astype(np.float32)

    lead_names = list(
        info.get("lead_names") or [f"lead{i}" for i in range(samples.shape[1])]
    )
    if derive_12:
        from ecgen.data.leads import STANDARD_12, derive_standard12

        samples = derive_standard12(samples, lead_names).astype(np.float32)
        lead_names = list(STANDARD_12)

    if output_dir is not None:
        write_samples(
            samples,
            output_dir,
            lead_names,
            int(info.get("fs") or 500),
            format,
            scale=info.get("scale"),
            cond=cond,
            checkpoint=checkpoint,
            plot=plot,
        )
    return samples


def write_samples(
    samples: np.ndarray,
    output_dir: PathLike,
    lead_names: List[str],
    fs: int,
    format: Optional[str] = "npy",
    scale: Any = None,
    cond: Optional[Mapping[str, Any]] = None,
    checkpoint: Any = None,
    plot: int = 0,
) -> List[Path]:
    """Write ``[n, C, T]`` samples as ``npy`` / ``csv`` / ``wfdb`` (+ ``samples.json``, plots)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    digits = len(str(len(samples)))
    units = "mV" if scale == "none" else "NU"  # z-scored data has no physical unit
    if format == "npy":
        created.append(out / "samples.npy")
        np.save(created[-1], samples)
    elif format == "csv":
        for i, ecg in enumerate(samples):
            path = out / f"sample_{i + 1:0{digits}d}.csv"
            np.savetxt(
                path,
                ecg.T,
                delimiter=",",
                header=",".join(lead_names),
                comments="",
                fmt="%.6f",
            )
            created.append(path)
    elif format == "wfdb":
        import wfdb

        for i, ecg in enumerate(samples):
            name = f"sample_{i + 1:0{digits}d}"
            wfdb.wrsamp(
                name,
                fs=fs,
                units=[units] * len(lead_names),
                sig_name=list(lead_names),
                p_signal=ecg.T.astype(np.float64),
                fmt=["16"] * len(lead_names),
                write_dir=str(out),
            )
            created.append(out / f"{name}.hea")
    elif format is not None:
        raise ValueError(f"format must be npy, csv, wfdb or None; got {format!r}")

    if plot:
        from ecgen.plot import plot_ecg

        for i in range(min(plot, len(samples))):
            path = out / f"sample_{i + 1:0{digits}d}.html"
            plot_ecg(
                samples[i],
                sample_rate=fs,
                lead_names=lead_names,
                title=f"Generated ECG {i + 1}",
                output_path=path,
            )
            created.append(path)

    meta = {
        "n": int(len(samples)),
        "shape": list(samples.shape),
        "lead_names": list(lead_names),
        "fs": fs,
        "units": units,
        "format": format,
        "checkpoint": str(checkpoint) if isinstance(checkpoint, (str, Path)) else None,
        "cond": {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in (cond or {}).items()
        },
    }
    (out / "samples.json").write_text(json.dumps(meta, indent=2, default=str))
    return created
