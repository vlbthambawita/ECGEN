"""
ecgen.generate — Generate synthetic ECG signals from a trained Pulse2Pulse checkpoint.

Model paths can be local files or HuggingFace Hub paths (``hf://`` prefix):

    from ecgen import generate

    # Local checkpoint
    paths = generate(
        model_path="runs/pulse2pulse/ptbxl/best.pt",
        n_samples=10,
        output_dir="outputs/",
    )

    # HuggingFace Hub — downloaded and cached automatically
    paths = generate(
        model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
        n_samples=10,
        output_dir="outputs/",
        ecgplot=True,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

# Lead names produced by the Pulse2Pulse PTB-XL model.
# PTB-XL 12-lead order: I, II, III, aVR, aVL, aVF, V1–V6.
# The model uses indices [0, 1, 6, 7, 8, 9, 10, 11] → I, II, V1–V6.
LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Keys renamed between the original Pulse2Pulse repo and the ECGEN model
_KEY_REMAP = {
    "Conv1dTrans": "conv1d_transpose",
    "ppfilter1":   "post_proc_filter",
}


def _remap_state_dict(state_dict: dict) -> dict:
    """Translate checkpoint keys saved by the original Pulse2Pulse repo to ECGEN names."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in _KEY_REMAP.items():
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def _resolve_model_path(model_path: Union[str, Path]) -> Path:
    """Return a local filesystem path to the checkpoint.

    If ``model_path`` starts with ``hf://``, the file is downloaded from
    HuggingFace Hub and cached in ``~/.cache/huggingface/hub/``.
    Subsequent calls with the same path return the cached copy instantly.

    HF path format::

        hf://<repo_id>/<path_in_repo>
        hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt

    A HF_TOKEN in the environment (or a ``.env`` file in CWD) is used automatically
    when present, so private repos are supported without extra configuration.
    """
    path_str = str(model_path)

    if not path_str.startswith("hf://"):
        local = Path(path_str)
        if not local.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local}")
        return local

    # --- HuggingFace path ---
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models from HF. "
            "Install with: pip install 'ecgen[hf]'"
        ) from exc

    # Strip scheme: "hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/file.pt"
    #           → "vlbthambawita/ECGEN/pulse2pulse/ptbxl/file.pt"
    remainder = path_str[len("hf://"):]

    # First two segments are the repo_id, the rest is the filename in the repo
    parts = remainder.split("/", 2)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid hf:// path '{path_str}'. "
            "Expected format: hf://<owner>/<repo>/<path_in_repo>"
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]

    # Load HF_TOKEN from .env if present (mirrors upload_hf behaviour)
    token: Optional[str] = os.environ.get("HF_TOKEN")
    if token is None:
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=False)
            except ImportError:
                _parse_dotenv(env_file)
            token = os.environ.get("HF_TOKEN")

    print(f"Downloading from HF: {repo_id}/{filename}")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    print(f"Cached at: {local_path}")
    return Path(local_path)


def _parse_dotenv(path: Path) -> None:
    """Minimal .env parser (fallback when python-dotenv is not installed)."""
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def _load_generator(
    checkpoint_path: Union[str, Path],
    model_size: int,
    device: torch.device,
) -> torch.nn.Module:
    from ecgen.models.pulse2pulse import WaveGANGenerator

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Checkpoints may store the full training state or just the generator weights
    if "netG_state_dict" in ckpt:
        state_dict = ckpt["netG_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # bare state dict

    state_dict = _remap_state_dict(state_dict)

    # Detect whether the checkpoint includes the post-processing filter
    has_ppfilter = any("post_proc_filter" in k for k in state_dict)
    post_proc_filt_len = 512 if has_ppfilter else 0

    netG = WaveGANGenerator(
        model_size=model_size,
        num_channels=8,
        post_proc_filt_len=post_proc_filt_len,
        upsample=True,
    )
    netG.load_state_dict(state_dict, strict=True)
    netG.to(device)
    netG.eval()
    return netG


def generate(
    model_path: Union[str, Path],
    n_samples: int,
    output_dir: Union[str, Path],
    *,
    format: str = "csv",
    header: bool = True,
    ecgplot: bool = False,
    model_size: int = 50,
    batch_size: int = 32,
    device: Optional[str] = None,
    seq_length: int = 5000,
) -> List[Path]:
    """Generate synthetic ECG signals from a trained Pulse2Pulse checkpoint.

    Parameters
    ----------
    model_path:
        Local path to a ``.pt`` checkpoint **or** a HuggingFace Hub path with
        the ``hf://`` prefix::

            "hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt"

        HF files are downloaded on first use and cached in
        ``~/.cache/huggingface/hub/``; subsequent calls are instant.
        Set ``HF_TOKEN`` (or add it to ``.env``) to access private repos.
    n_samples:
        Number of ECG samples to generate.
    output_dir:
        Directory where output files will be written (created if it doesn't exist).
    format:
        Output format. Currently supports ``"csv"``.
    header:
        When ``format="csv"``, include a header row with lead names
        (``I, II, V1, V2, V3, V4, V5, V6``). Default ``True``.
    ecgplot:
        If ``True``, also save an ECG plot image (``.png``) for each sample
        alongside the CSV. Requires ``pip install ecgplot``.
    model_size:
        Generator ``model_size`` hyperparameter (default 50, matching PTB-XL checkpoint).
    batch_size:
        Number of samples to generate per forward pass. Reduce if running out of GPU memory.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect, prefers GPU).
    seq_length:
        Length of each generated ECG signal in samples (default 5000).

    Returns
    -------
    List of ``Path`` objects pointing to the files that were created.
    """
    if format not in ("csv",):
        raise ValueError(f"Unsupported format '{format}'. Supported: 'csv'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    print(f"Using device: {_device}")

    # Resolve local path or download from HuggingFace Hub
    local_path = _resolve_model_path(model_path)
    print(f"Loading generator from: {local_path}")
    netG = _load_generator(local_path, model_size=model_size, device=_device)

    # Generate in batches
    n_digits = len(str(n_samples))
    all_samples: List[np.ndarray] = []

    with torch.no_grad():
        remaining = n_samples
        while remaining > 0:
            bs = min(batch_size, remaining)
            noise = torch.FloatTensor(bs, 8, seq_length).uniform_(-1, 1).to(_device)
            fake = netG(noise)                      # (bs, 8, 5000)
            all_samples.append(fake.cpu().numpy())  # keep on CPU
            remaining -= bs

    ecg_array = np.concatenate(all_samples, axis=0)  # (n_samples, 8, 5000)
    print(f"Generated {len(ecg_array)} ECG sample(s).")

    # Save outputs
    created: List[Path] = []

    for idx, ecg in enumerate(ecg_array):
        # ecg shape: (8, 5000)  — channels-first
        sample_id = str(idx + 1).zfill(n_digits)

        if format == "csv":
            csv_path = output_dir / f"sample_{sample_id}.csv"
            _save_csv(ecg, csv_path, header=header)
            created.append(csv_path)

        if ecgplot:
            png_path = output_dir / f"sample_{sample_id}.png"
            _save_ecgplot(ecg, png_path)
            created.append(png_path)

    print(f"Saved {len(created)} file(s) to: {output_dir}")
    return created


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(ecg: np.ndarray, path: Path, header: bool) -> None:
    """Save a single ECG as a CSV file.

    Rows = time steps (5000), columns = leads (I, II, V1–V6).
    """
    import csv

    # ecg shape: (8, 5000) → transpose to (5000, 8)
    data = ecg.T

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(LEAD_NAMES)
        for row in data:
            writer.writerow([f"{v:.6f}" for v in row])


def _save_ecgplot(ecg: np.ndarray, path: Path) -> None:
    """Save an ECG plot image using the ecgplot library."""
    try:
        import ecgplot
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "ecgplot is required for plot output. Install with: pip install ecgplot"
        ) from e

    # ecgplot.plot expects shape (n_leads, n_samples)
    fig, ax = plt.subplots(figsize=(20, 10))
    ecgplot.plot(ecg, sample_rate=500, title=path.stem, columns=2, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
