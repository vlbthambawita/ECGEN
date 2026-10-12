# ECGEN

**ECG Generation Framework** — PyTorch Lightning implementations of generative models for 12-lead electrocardiogram (ECG) signal synthesis and latent representation learning.

## Overview

ECGEN trains and samples generative models of 12-lead ECGs through one library
and one command (`ecgen train <config.yaml>`), on MIMIC-IV-ECG, PTB-XL, or
your own data. Training methods, networks, and conditions are independent,
swappable parts (see [Concepts](https://vajira.info/ECGEN/concepts/)):

| Family (`model.name`) | Method |
|---|---|
| `vae` | VAE / conditional VAE (spatial or global latent, learned prior) |
| `pulse2pulse` | Pulse2Pulse WaveGAN (WGAN-GP); conditional and paired lead-reconstruction modes |
| `vqvae`, `vqvae_prior` | VQ-VAE + PixelCNN prior (conditional, classifier-free guidance) |
| `diffusion` | DDPM (ε / v / x₀, cosine / linear, min-SNR) with DDIM / ancestral sampling |
| `flow_matching` | Rectified flow / conditional flow matching with Euler / Heun sampling, optional OT coupling |
| `latent_diffusion`, `latent_flow_matching` | The same in the latent space of a frozen VAE or VQ-VAE |

Backbones for diffusion / flow: `unet1d`, `dit1d` (transformer, AdaLN-Zero),
`ssm` (SSSD-style S4D), `mamba`. Conditions: class labels and multi-label
diagnostic statements, tabular / demographic / measurement fields (missing-aware),
report text (hashed or a frozen Hugging Face encoder), and partial ECGs (any
lead subset, time spans, forecasting) — with condition dropout, classifier-free
guidance, and training-free inpainting (replace / RePaint).

```bash
ecgen train configs/experiments/diffusion_mimic_conditional.yaml
ecgen sample runs/diffusion_mimic_conditional/seed_42/checkpoints/last.ckpt -n 16 \
    --cond measurements=800,40,128,170,258,520,60,30,40 --guidance 2 --format csv -o samples/
ecgen evaluate samples/samples.npy configs/experiments/diffusion_mimic_conditional.yaml
```

The original Pulse2Pulse generator API (`ecgen.generate`) and the plotting
helpers below are unchanged.

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
from ecgen.families.vae import VAELightning, VAEConfig

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
from ecgen.families.gan import Pulse2PulseGAN, Pulse2PulseConfig

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

Every model family trains through one runner, driven by an experiment YAML in
`configs/experiments/`. `key=value` overrides are applied after the YAML.

```bash
ecgen train configs/experiments/vae_mimic.yaml          # VAE
ecgen train configs/experiments/pulse2pulse_mimic.yaml  # Pulse2Pulse GAN
ecgen train configs/experiments/vqvae_mimic.yaml        # VQ-VAE stage 1
ecgen train configs/experiments/prior_mimic.yaml \
  model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt  # stage 2

# Quick test with a small subset
ecgen train configs/experiments/vae_mimic.yaml \
  data.dataset.params.max_samples=1000 trainer.max_epochs=10 data.batch_size=16

# Smoke run without MIMIC
ecgen train configs/experiments/vae_mimic.yaml \
  data.dataset.name=synthetic "data.dataset.params={num_samples: 64}" trainer.accelerator=cpu

# Resume, inspect the resolved config, list registered models/datasets/callbacks
ecgen train configs/experiments/vae_mimic.yaml --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
ecgen train configs/experiments/vae_mimic.yaml --print-config
ecgen list
```

`python -m ecgen train ...` works without the console script. The shell wrappers in
`scripts/shell/` call the same runner.

Run outputs are saved to `runs/{experiment_name}/seed_{seed}/` containing:
- `checkpoints/` — best and last model weights
- `samples/` — generated ECG batches
- `tb/` — TensorBoard logs

### Config format

Components are built from a registry `name` (`ecgen list`) or an import-path `target`, with
`params` passed to the constructor:

```yaml
experiment:
  name: vae_mimic
  seed: 42

model:
  name: vae                 # or target: ecgen.families.vae.VAELightning
  params:
    in_channels: 12
    latent_channels: 8
    kl_weight: 0.0001

data:
  dataset:
    name: mimic_iv_ecg
    params:
      root: /path/to/mimic-iv-ecg
      max_samples: null     # null = full dataset
  batch_size: 32

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
├── api.py          — ecgen.sample / ecgen.load_model
├── cli.py          — ecgen train | sample | evaluate | list
├── registry.py     — FAMILIES, DATASETS, CONDITION_ENCODERS, MASKS, BACKBONES, PROCESSES, CALLBACKS
├── types.py        — ECGBatch, ConditionOutput
├── data/           — datasets (MIMIC-IV-ECG, PTB-XL, synthetic), masks, normalizers, ECGDataModule
├── conditioning/   — field encoders (labels, numeric, text, masked signal), CompositeConditioner
├── families/       — vae, gan (Pulse2Pulse), vqvae (+ prior), continuous (diffusion, flow, latent)
├── processes/      — ddpm, rectified_flow
├── nn/             — blocks, injection layers (FiLM, AdaGN, AdaLN-Zero, cross-attention), backbones
├── training/       — runner, callbacks, EMA
├── evaluation/     — measurements, distribution / adherence / reconstruction metrics, classifier
├── generate.py     — legacy Pulse2Pulse generator; plot.py — ECG plots; f/ — HF upload
└── models/         — deprecated alias of families/ (warns; removed in 2.0)
```

---

## Research framework

Besides the published `ecgen` package, the repository root holds the research code used for new experiments. These packages are **not** part of the PyPI wheel; run scripts and tests from the repo root:

```
models/         VAE, VQ-VAE + PixelCNN prior, Pulse2Pulse GAN (diffusion/SSM placeholders)
data/           Dataset implementations (MIMIC-IV-ECG)
training/       Losses, callbacks, trainers
evaluation/     Evaluation metrics and benchmarking
visualization/  ECG, latent-space and training-curve plots
utils/          Config helpers
configs/        Experiment YAMLs (configs/experiments/*.yaml)
scripts/        train/, generate/, evaluate/, shell/ wrappers
single_file/    Self-contained VQ-VAE (both stages) in one script
```

```bash
pip install -e ".[dev]"
pytest tests/

# VQ-VAE stage 1, then the PixelCNN prior (stage 2)
./scripts/shell/run_train_vqvae_mimic.sh
./scripts/shell/run_train_vqvae_prior_mimic.sh runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
```

Full documentation: **https://vajira.info/ECGEN/** (source in `docs/`, built with MkDocs Material).

---

## Releasing to PyPI

The package is published as [`ecgen`](https://pypi.org/project/ecgen/) automatically on tagged commits:

```bash
# The version comes from the tag (setuptools-scm); no file to bump
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions builds and publishes to PyPI via OIDC trusted publishing (no API token required).

---

## License

See [LICENSE](LICENSE).
