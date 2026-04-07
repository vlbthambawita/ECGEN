# ECGEN

**ECG Generation Framework** — PyTorch Lightning implementations of generative models for 12-lead electrocardiogram (ECG) signal synthesis and latent representation learning.

## Overview

ECGEN provides two deep generative models for ECG signals, both trained primarily on the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset (12-lead, 5000 samples/signal):

| Model | Type | Leads | Purpose |
|---|---|---|---|
| **VAE** | Variational Autoencoder | 12 | Unsupervised latent representation & reconstruction |
| **Pulse2Pulse** | WaveGAN (WGAN-GP) | 8 | Conditional ECG signal generation |

---

## Installation

```bash
# From source
git clone https://github.com/vlbthambawita/ECGEN.git
cd ECGEN
pip install -e .

# With HuggingFace Hub support (for checkpoint uploads)
pip install 'ecgen[hf]'
```

**Requirements:** Python ≥ 3.8, PyTorch, PyTorch Lightning

---

## Models

### VAE — Variational Autoencoder

1D convolutional VAE with residual blocks for learning compact ECG representations. The encoder maps a 12-lead signal to a Gaussian latent space; the decoder reconstructs the signal from sampled latents.

**Architecture:** `ResidualBlock1D → Encoder1D → latent (μ, σ) → Decoder1D`

**Loss:** `reconstruction_loss + kl_weight × KL_divergence`

```python
from ecgen.models import VAELightning, VAEConfig

config = VAEConfig(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
    channel_multipliers=(1, 2, 4, 4),
    num_res_blocks=2,
    lr=1e-4,
    kl_weight=1e-4,
)
model = VAELightning(config)

# Generate new ECG signals
samples = model.sample(n_samples=16, seq_length=5000)  # (16, 12, 5000)
```

### Pulse2Pulse — WaveGAN

Wasserstein GAN with gradient penalty for conditional ECG generation. Uses the first 8 leads of MIMIC-IV-ECG signals.

```python
from ecgen.models import Pulse2PulseGAN, Pulse2PulseConfig

config = Pulse2PulseConfig(
    model_size=50,
    num_channels=8,
    seq_length=5000,
    lr=1e-4,
    lmbda=10.0,   # gradient penalty weight
    n_critic=5,   # discriminator steps per generator step
)
model = Pulse2PulseGAN(config)
```

---

## Generating ECG Signals

Generate synthetic 8-lead ECG signals from any trained Pulse2Pulse checkpoint — local or directly from HuggingFace Hub:

```python
from ecgen import generate

# From HuggingFace Hub — downloaded once, cached forever
paths = generate(
    model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
    n_samples=10,
    output_dir="outputs/generated",
)
# → outputs/generated/sample_01.csv … sample_10.csv

# Local checkpoint
paths = generate(
    model_path="checkpoints/pulse2pulse_ptbxl_epoch900.pt",
    n_samples=10,
    output_dir="outputs/generated",
)

# Without headers
generate(..., header=False)

# With ECG plot images (requires: pip install 'ecgen[plot]')
generate(..., ecgplot=True)
# → also saves sample_01.png … sample_10.png
```

Each CSV has **5000 rows × 8 columns** (one row per time step, columns = `I, II, V1, V2, V3, V4, V5, V6`).

**HF path format:** `hf://<owner>/<repo>/<path_in_repo>`  
Files are cached in `~/.cache/huggingface/hub/` after the first download. Set `HF_TOKEN` in `.env` for private repos.

| Parameter | Default | Description |
|---|---|---|
| `model_path` | — | Local `.pt` path **or** `hf://owner/repo/path` |
| `n_samples` | — | Number of ECGs to generate |
| `output_dir` | — | Output directory (created if absent) |
| `format` | `"csv"` | Output format (`"csv"`) |
| `header` | `True` | Include lead-name header row in CSV |
| `ecgplot` | `False` | Also save `.png` ECG plot per sample |
| `model_size` | `50` | Generator `model_size` (must match training) |
| `batch_size` | `32` | Samples per GPU forward pass |
| `device` | auto | `"cuda"`, `"cpu"`, or `None` (auto) |

---

## Training

### VAE

```bash
# Config-driven (recommended)
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml

# Quick test with a small subset
python scripts/train_vae_mimic.py \
  --data-dir /path/to/mimic-iv-ecg \
  --max-samples 1000 --max-epochs 10 --batch-size 16

# Resume from checkpoint
python scripts/train_vae_mimic.py \
  --config configs/experiments/vae_mimic.yaml \
  --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
```

### Pulse2Pulse

```bash
python -m ecgen.training.train --config configs/experiments/pulse2pulse_mimic.yaml
```

Run outputs are saved to `runs/{experiment_name}/seed_{seed}/` containing:
- `checkpoints/` — best and last model weights
- `samples/` — generated ECG batches
- `tb/` — TensorBoard logs

### Config format

All experiments use a YAML config with `target` + `params` for dynamic instantiation:

```yaml
experiment:
  name: vae_mimic
  seed: 42

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
    batch_size: 32
    max_samples: null   # null = full dataset

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: [0]
```

---

## Uploading Checkpoints to HuggingFace

```bash
pip install 'ecgen[hf]'
huggingface-cli login
```

**From Python:**

```python
from ecgen.f import upload_hf, upload_hf_single

# Upload using a YAML config (supports subcategories like vae/ptbxl/)
upload_hf("configs/upload_checkpoints.yaml")

# Upload a single checkpoint
upload_hf_single(
    local_path="runs/vae/ptbxl/best.ckpt",
    repo_path="vae/ptbxl/best.ckpt",
    repo_id="your_username/ECGEN",
)
```

**From CLI:**

```bash
python scripts/upload_checkpoints_to_hf.py --dry-run   # preview
python scripts/upload_checkpoints_to_hf.py             # upload
```

Edit `configs/upload_checkpoints.yaml` to define which checkpoints map to which paths in the HF repo.

---

## Package Structure

```
src/ecgen/
├── models/        — VAE and Pulse2Pulse model definitions
├── data/          — MIMIC-IV-ECG dataset loaders and data modules
├── training/      — Training loop, losses, metrics, callbacks
├── f/             — Functional utilities (upload_hf, upload_hf_single)
└── utils/         — Seeding, I/O, logging helpers
```

---

## Releasing to PyPI

The package is published as [`ecgen`](https://pypi.org/project/ecgen/) automatically on tagged commits:

```bash
# Bump __version__ in src/ecgen/__init__.py, commit, then:
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions builds and publishes to PyPI via OIDC trusted publishing (no API token required).

---

## License

See [LICENSE](LICENSE).
