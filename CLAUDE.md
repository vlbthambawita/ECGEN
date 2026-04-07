# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

ECGEN is an ECG (electrocardiogram) generation framework built on PyTorch Lightning. It implements two generative models:
- **VAE** — variational autoencoder for unsupervised latent representation of 12-lead ECG signals
- **Pulse2Pulse** — conditional WaveGAN (WGAN-GP) for ECG generation

Primary dataset: **MIMIC-IV-ECG** (12-lead, 5000 samples per signal).

## Development setup

```bash
pip install -e .
# or for scripts that don't install:
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Running tests

```bash
python tests/test_vae.py                          # all VAE tests
python -c "from tests.test_vae import test_vae_forward; test_vae_forward()"  # single test
```

No pytest configuration exists; test files use plain functions called at the bottom of each file.

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

## Code architecture

```
src/ecgen/
├── models/
│   ├── vae.py          — ResidualBlock1D → Encoder1D/Decoder1D → VAE1D → VAELightning
│   └── pulse2pulse.py  — WaveGANGenerator, WaveGANDiscriminator, Pulse2PulseGAN
├── data/
│   ├── mimic_dataset.py       — MIMICIVECGDataset: returns (ecg[12,5000], features[9])
│   └── pulse2pulse_mimic.py   — Pulse2PulseMIMICDataModule + ECGDatasetAdapter (8 leads)
├── training/
│   ├── train.py        — generic config-driven training entry point
│   ├── losses.py       — loss functions (WGAN-GP gradient penalty)
│   ├── metrics.py      — evaluation metrics
│   └── callbacks.py    — ECGVisualizationCallback, GeneratedSamplesCallback
└── utils/
    ├── io.py           — read_yaml / write_yaml / write_json
    └── seed.py         — set_global_seed() (Python + NumPy + PyTorch)
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

`training/train.py` imports and instantiates these at runtime via `importlib`. Config templates live in `configs/experiments/`.

## Key data conventions

- ECG tensors are always `(batch, channels, seq_length)` — e.g. `(B, 12, 5000)`
- VAE uses all 12 leads; Pulse2Pulse uses 8 leads (first 8, via `ECGDatasetAdapter`)
- Dataset splits default to 0.8 / 0.1 / 0.1 (train/val/test), set at init time
- VAE loss: `recon_loss + kl_weight × kl_divergence`

## Releasing to PyPI

Tag a commit to trigger automatic publish via GitHub Actions:

```bash
# bump __version__ in src/ecgen/__init__.py, commit, then:
git tag v0.2.0
git push origin v0.2.0
```

The workflow (`.github/workflows/publish.yml`) overwrites `__version__` with the tag and publishes to PyPI using OIDC trusted publishing (no token needed, but the publisher must be registered once at pypi.org/manage/account/publishing/).

## Uploading checkpoints to HuggingFace

Install with HF support: `pip install 'ecgen[hf]'`

**From Python (programmatic):**

```python
from ecgen.f import upload_hf, upload_hf_single

# Upload everything defined in a YAML config
upload_hf("configs/upload_checkpoints.yaml")
upload_hf("configs/upload_checkpoints.yaml", dry_run=True)   # preview

# Upload a single checkpoint
upload_hf_single(
    local_path="runs/vae/ptbxl/best.ckpt",
    repo_path="vae/ptbxl/best.ckpt",
    repo_id="username/ECGEN",
)

# Upload from a plain dict (useful in notebooks / CI)
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

`upload_hf` raises `RuntimeError` on any failed upload (instead of calling `sys.exit`), so it is safe to use inside larger pipelines. The CLI script wraps it and converts exceptions to exit codes.

The `ecgen.f` sub-package (`src/ecgen/f/`) is the home for all such functional utilities. `huggingface_hub` is an optional dependency; the import error is surfaced as `ImportError` with an install hint.
