# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ECGEN is an ECG (electrocardiogram) generation framework built on PyTorch Lightning. It implements two generative models:
- **VAE** — variational autoencoder for unsupervised latent representation of 12-lead ECG signals
- **Pulse2Pulse** — conditional WaveGAN (WGAN-GP) for ECG generation

Primary dataset: **MIMIC-IV-ECG** (12-lead, 5000 samples per signal). PTB-XL is also supported via configs.

## Development setup

```bash
pip install -e .
# or for scripts that don't install:
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Optional extras (see `pyproject.toml`):
- `pip install 'ecgen[hf]'` — `huggingface_hub` + `python-dotenv` (required for HF uploads and `hf://` model paths)
- `pip install 'ecgen[plot]'` — `cairosvg` (only required to write **PDF** plots; HTML and SVG work out of the box)
- `pip install 'ecgen[all]'` — both

## Running tests

```bash
python tests/test_vae.py                          # all VAE tests
python tests/test_models.py
python tests/test_data.py
python tests/test_upload_hf.py
python -c "from tests.test_vae import test_vae_forward; test_vae_forward()"  # single test
```

There is no pytest config; each test file defines plain functions and calls them at the bottom.

## Training

```bash
# VAE — config-driven (recommended)
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml

# VAE — CLI args (useful for quick experiments)
python scripts/train_vae_mimic.py \
  --data-dir /path/to/mimic-iv-ecg \
  --max-samples 1000 --max-epochs 10 --batch-size 16

# Pulse2Pulse
python -m ecgen.training.train --config configs/experiments/pulse2pulse_mimic.yaml

# Resume from checkpoint
python scripts/train_vae_mimic.py \
  --config configs/experiments/vae_mimic.yaml \
  --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
```

Run outputs land in `runs/{exp_name}/seed_{seed}/` with `checkpoints/`, `samples/`, and `tb/` subdirectories.

## Generating ECG signals

```python
from ecgen import generate

# Local checkpoint
paths = generate(
    model_path="checkpoints/pulse2pulse_ptbxl_epoch900.pt",
    n_samples=10,
    output_dir="outputs/",
    format="csv",        # only "csv" supported currently
    header=True,         # include I,II,V1–V6 header row
    ecgplot=False,       # True → also save a plot per sample
    plot_format="html",  # "html" | "svg" | "pdf"  (pdf needs ecgen[plot])
    model_size=50,       # must match training config
    batch_size=32,
    device=None,         # None → auto-detect cuda/cpu
    denorm=1.0,          # use 6000.0 for ECGDataSimple-trained models
)

# Or load directly from HuggingFace Hub (cached after first call)
paths = generate(
    model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
    n_samples=10,
    output_dir="outputs/",
)
```

Each CSV: 5000 rows × 8 columns (`I, II, V1, V2, V3, V4, V5, V6`), one file per sample.

`ecgen.generate` lives in `src/ecgen/generate.py`. Two non-obvious details:
1. Checkpoints from the original Pulse2Pulse repo use `Conv1dTrans` and `ppfilter1` as state-dict keys; `_remap_state_dict()` rewrites these to `conv1d_transpose` and `post_proc_filter` before loading.
2. `hf://<repo_id>/<path>` paths are resolved through `_resolve_model_path()`, which uses an `HF_TOKEN` from the environment or a `.env` in CWD when present (so private repos work without extra config).

## Standalone ECG plots

`plot_ecg` (exported from the top-level `ecgen` package) renders any ECG array on a real ECG grid sheet:

```python
from ecgen import plot_ecg
import numpy as np

ecg = np.load("my_ecg.npy")             # (n_leads, n_samples) or transposed
plot_ecg(ecg, output_path="ecg.html")   # interactive D3 HTML (default)
plot_ecg(ecg, output_path="ecg.svg", format="svg")
plot_ecg(ecg, output_path="ecg.pdf", format="pdf")  # needs ecgen[plot]
```

HTML and SVG are pure-Python (no extras). PDF goes through `cairosvg`.

## Code architecture

```
src/ecgen/
├── generate.py    — ecgen.generate(): load checkpoint (local or hf://) → CSV + plots
├── plot.py        — ecgen.plot_ecg(): D3 HTML / SVG / PDF ECG grid renderer
├── models/
│   ├── vae.py          — ResidualBlock1D → Encoder1D/Decoder1D → VAE1D → VAELightning
│   └── pulse2pulse.py  — WaveGANGenerator, WaveGANDiscriminator, Pulse2PulseGAN
├── data/
│   ├── mimic_dataset.py       — MIMICIVECGDataset: returns (ecg[12,5000], features[9])
│   ├── pulse2pulse_mimic.py   — Pulse2PulseMIMICDataModule + ECGDatasetAdapter (8 leads)
│   ├── datamodule.py          — generic LightningDataModule
│   └── transforms.py
├── training/
│   ├── train.py        — generic config-driven training entry point
│   ├── validate.py     — validation entry point
│   ├── test.py         — test entry point
│   ├── losses.py       — loss functions (WGAN-GP gradient penalty)
│   ├── metrics.py
│   └── callbacks.py    — ECGVisualizationCallback, GeneratedSamplesCallback
├── f/
│   └── upload_hf.py    — upload_hf(), upload_hf_single()
└── utils/
    ├── io.py           — read_yaml / write_yaml / write_json
    ├── seed.py         — set_global_seed() (Python + NumPy + PyTorch)
    ├── logging.py
    └── metadata.py
```

All models are `pytorch_lightning.LightningModule` subclasses. Training is fully config-driven via YAML.

## Config system

YAML configs use a `target` + `params` pattern for dynamic instantiation:

```yaml
model:
  target: ecgen.models.vae.VAELightning
  params:
    config:
      in_channels: 12
      latent_channels: 8
      kl_weight: 0.0001

data:
  target: ecgen.data.mimic_dataset.MIMICIVECGDataset
  params:
    mimic_path: /path/to/mimic-iv-ecg
    max_samples: null   # null = full dataset
```

`training/train.py` imports and instantiates these at runtime via `importlib`.

Config layout:
- `configs/experiments/` — full self-contained experiment configs (`vae_mimic.yaml`, `pulse2pulse_mimic.yaml`)
- `configs/model/`, `configs/dataset/`, `configs/trainer/` — modular building blocks (e.g. `ptbxl.yaml`, `default.yaml`)
- `configs/upload_checkpoints.yaml` — HF upload manifest

## Key data conventions

- ECG tensors are always `(batch, channels, seq_length)` — e.g. `(B, 12, 5000)`
- VAE uses all 12 leads; Pulse2Pulse uses 8 leads (first 8, via `ECGDatasetAdapter`)
- Dataset splits default to 0.8 / 0.1 / 0.1 (train/val/test), set at init time
- VAE loss: `recon_loss + kl_weight × kl_divergence`

## Releasing to PyPI

Versioning is fully driven by git tags via `setuptools-scm` — there is no `__version__` literal to bump. Just tag and push:

```bash
git tag v1.1.0
git push origin v1.1.0
```

`pip install -e .` and any `python -m build` derive the version from the latest reachable tag. The generated file `src/ecgen/_version.py` is gitignored. Between tags, editable installs report a dev version like `1.0.1.dev3+g<sha>`.

The workflow (`.github/workflows/publish.yml`) checks out with `fetch-depth: 0` so tags are visible to setuptools-scm, then publishes to PyPI using OIDC trusted publishing (no token needed, but the publisher must be registered once at pypi.org/manage/account/publishing/).

## Uploading checkpoints to HuggingFace

Install with HF support: `pip install 'ecgen[hf]'`

**From Python (programmatic):**

```python
from ecgen.f import upload_hf, upload_hf_single

upload_hf("configs/upload_checkpoints.yaml")
upload_hf("configs/upload_checkpoints.yaml", dry_run=True)   # preview

upload_hf_single(
    local_path="runs/vae/ptbxl/best.ckpt",
    repo_path="vae/ptbxl/best.ckpt",
    repo_id="username/ECGEN",
)

# Plain dict (notebooks / CI)
upload_hf({
    "hf_repo_id": "username/ECGEN",
    "checkpoints": [
        {"local_path": "runs/vae/ptbxl/best.ckpt", "repo_path": "vae/ptbxl/best.ckpt"},
    ],
})
```

**From CLI:**

```bash
huggingface-cli login
python scripts/upload_checkpoints_to_hf.py --dry-run   # preview
python scripts/upload_checkpoints_to_hf.py             # upload
```

`upload_hf` raises `RuntimeError` on any failed upload (instead of calling `sys.exit`), so it's safe to use inside larger pipelines. The CLI script wraps it and converts exceptions to exit codes.

The `ecgen.f` sub-package (`src/ecgen/f/`) is the home for all such functional utilities. `huggingface_hub` is an optional dependency; the import error is surfaced as `ImportError` with an install hint.
