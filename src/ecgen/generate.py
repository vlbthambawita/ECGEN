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

    # The original Pulse2Pulse repo defined ppfilter1 in __init__ but never
    # called it in forward(), so its weights are random kaiming initialisation
    # and were never updated by backprop.  Applying them at inference corrupts
    # the output.  Drop the keys so the filter is not used.
    state_dict = {k: v for k, v in state_dict.items() if "post_proc_filter" not in k}

    netG = WaveGANGenerator(
        model_size=model_size,
        num_channels=8,
        post_proc_filt_len=0,
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
    plot_format: str = "png",
    num_plot_samples: int = 4,
    model_size: int = 50,
    batch_size: int = 32,
    device: Optional[str] = None,
    seq_length: int = 5000,
    denorm: float = 1.0,
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
        Output format for signal data.  ``"csv"`` saves one file per sample;
        ``"npy"`` saves all samples as a single ``generated_ecgs.npy`` array of
        shape ``(n_samples, 8, 5000)``.
    header:
        When ``format="csv"``, include a header row with lead names
        (``I, II, V1, V2, V3, V4, V5, V6``). Default ``True``.
    ecgplot:
        If ``True``, save ECG plots into a ``plots/`` sub-directory.

        * ``plot_format="png"`` (default) — matplotlib PNG: one PNG per sample
          plus a batch overview PNG.
        * ``plot_format="html"`` — interactive D3 HTML, one file per sample.
        * ``plot_format="svg"`` — static SVG, one file per sample.
        * ``plot_format="pdf"`` — PDF via cairosvg, one file per sample
          (requires ``pip install 'ecgen[plot]'``).
    num_plot_samples:
        When ``ecgplot=True`` and ``plot_format="png"``, number of individual
        sample PNGs and the batch-overview plot (default 4).
    model_size:
        Generator ``model_size`` hyperparameter (default 50, matching PTB-XL checkpoint).
    batch_size:
        Number of samples to generate per forward pass. Reduce if running out of GPU memory.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect, prefers GPU).
    seq_length:
        Length of each generated ECG signal in samples (default 5000).
    denorm:
        Multiply raw generator output by this factor before saving.
        Default ``1.0`` (no scaling) — correct for the PTB-XL checkpoint, which
        was trained with ``wfdb`` physical mV values (``norm_num=1.0``).
        Use ``denorm=6000.0`` for models trained on ``ECGDataSimple``, which
        normalises signals by dividing by 6000 before training.

    Returns
    -------
    List of ``Path`` objects pointing to the files that were created.
    """
    if format not in ("csv", "npy"):
        raise ValueError(f"Unsupported format '{format}'. Supported: 'csv', 'npy'")
    if ecgplot and plot_format not in ("png", "html", "svg", "pdf"):
        raise ValueError(
            f"Unsupported plot_format '{plot_format}'. Choose 'png', 'html', 'svg', or 'pdf'."
        )

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
    if denorm != 1.0:
        ecg_array = ecg_array * denorm
    print(f"Generated {len(ecg_array)} ECG sample(s).")

    # Save signal outputs
    created: List[Path] = []

    if format == "npy":
        npy_path = output_dir / "generated_ecgs.npy"
        np.save(npy_path, ecg_array)
        created.append(npy_path)
        print(f"Saved {ecg_array.shape[0]} ECGs to: {npy_path}")
    else:  # csv
        n_digits = len(str(n_samples))
        for idx, ecg in enumerate(ecg_array):
            sample_id = str(idx + 1).zfill(n_digits)
            csv_path = output_dir / f"sample_{sample_id}.csv"
            _save_csv(ecg, csv_path, header=header)
            created.append(csv_path)

    # Save plots
    if ecgplot:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if plot_format == "png":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from ecgen.plot import plot_generated_single, plot_generated_batch

            n_plot = min(num_plot_samples, ecg_array.shape[0])
            for idx in range(n_plot):
                fig = plot_generated_single(
                    ecg_array[idx],
                    sampling_rate=seq_length // 10,  # 5000 samples → 500 Hz
                    title=f"Generated sample {idx + 1}",
                )
                png_path = plots_dir / f"generated_sample_{idx:04d}.png"
                fig.savefig(png_path, dpi=150)
                plt.close(fig)
                created.append(png_path)

            fig_batch = plot_generated_batch(
                ecg_array,
                num_of_plots=n_plot,
                sampling_rate=seq_length // 10,
            )
            batch_path = plots_dir / "generated_batch_overview.png"
            fig_batch.savefig(batch_path, dpi=150)
            plt.close(fig_batch)
            created.append(batch_path)
        else:
            # D3/SVG/PDF — one file per sample
            from ecgen.plot import plot_ecg
            _ext = {"html": ".html", "svg": ".svg", "pdf": ".pdf"}[plot_format]
            n_digits = len(str(n_samples))
            for idx, ecg in enumerate(ecg_array):
                sample_id = str(idx + 1).zfill(n_digits)
                plot_path = plots_dir / f"sample_{sample_id}{_ext}"
                plot_ecg(
                    ecg,
                    sample_rate=seq_length // 10,
                    title=f"Sample {sample_id}",
                    output_path=plot_path,
                    format=plot_format,
                )
                created.append(plot_path)

        print(f"Saved plots to: {plots_dir}")

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


