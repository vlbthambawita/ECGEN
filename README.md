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

# With interactive D3 HTML plots alongside each CSV (default)
generate(..., ecgplot=True)
# → sample_01.csv + sample_01.html …

# Static SVG plots
generate(..., ecgplot=True, plot_format="svg")

# PDF plots (requires: pip install 'ecgen[plot]')
generate(..., ecgplot=True, plot_format="pdf")
```

| Parameter | Default | Description |
|---|---|---|
| `model_path` | — | Local `.pt` path **or** `hf://owner/repo/path` |
| `n_samples` | — | Number of ECGs to generate |
| `output_dir` | — | Output directory (created if absent) |
| `format` | `"csv"` | Signal output format |
| `header` | `True` | Lead-name header row in CSV (`I,II,V1…V6`) |
| `ecgplot` | `False` | Also save an ECG plot per sample |
| `plot_format` | `"html"` | `"html"` · `"svg"` · `"pdf"` |
| `model_size` | `50` | Generator `model_size` (must match training) |
| `batch_size` | `32` | Samples per GPU forward pass |
| `device` | auto | `"cuda"`, `"cpu"`, or `None` |
| `denorm` | `1.0` | Multiply raw output by this factor. Use `6000.0` for models trained on `ECGDataSimple` (÷6000 normalisation). Default `1.0` is correct for the PTB-XL checkpoint (physical mV, no normalisation). |

---

## ECG Plots

`plot_ecg` renders any ECG array as a **real ECG graph-sheet** — pink/red grid paper, labelled leads, 25 mm/s paper speed, 10 mm/mV scale:

```python
from ecgen import plot_ecg
import numpy as np

ecg = np.load("my_ecg.npy")  # shape (8, 5000) or (5000, 8)

# Interactive HTML — pan/zoom with mouse/touch, save-SVG and Print buttons
plot_ecg(ecg, output_path="ecg.html", title="My ECG")

# Static SVG — embed in documents, no JS required
plot_ecg(ecg, output_path="ecg.svg", format="svg")

# PDF (requires: pip install 'ecgen[plot]')
plot_ecg(ecg, output_path="ecg.pdf", format="pdf")

# Return content string instead of writing a file
html_str = plot_ecg(ecg)
svg_str  = plot_ecg(ecg, format="svg")
```

The interactive HTML uses **D3 v7** and is fully self-contained (single file). Features:
- Scroll / pinch to zoom (x-axis); drag to pan
- ECG grid scales with zoom so the mm² boxes always represent the same time/amplitude
- **⤓ SVG** button exports the current view as a clean SVG file
- **🖨 Print / PDF** button opens the browser print dialog (set destination to "Save as PDF")

| Parameter | Default | Description |
|---|---|---|
| `ecg` | — | `(n_leads, n_samples)` or `(n_samples, n_leads)` |
| `sample_rate` | `500` | Hz |
| `lead_names` | auto | Defaults to `["I","II","V1"…"V6"]` |
| `title` | `"ECG"` | Shown in toolbar and file header |
| `output_path` | `None` | Write to file; `None` returns the content string |
| `format` | `"html"` | `"html"` · `"svg"` · `"pdf"` |
| `paper_speed` | `25.0` | mm/s |
| `amplitude_scale` | `10.0` | mm/mV |

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
