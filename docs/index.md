---
hide:
  - navigation
---

# ECGEN

**ECGEN** is a research framework for generative modeling of **12-lead electrocardiograms**, built on [PyTorch Lightning](https://lightning.ai/). It provides model implementations, dataset loaders, one YAML-driven training command (`ecgen train`), evaluation metrics and visualization tools.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started**

    ---

    Install the package and train your first model on MIMIC-IV-ECG.

    [:octicons-arrow-right-24: Installation](getting_started/installation.md) ·
    [Quick Start](getting_started/quickstart.md)

-   :material-brain:{ .lg .middle } **Models**

    ---

    VAE, VQ-VAE with a PixelCNN prior, and the Pulse2Pulse GAN.

    [:octicons-arrow-right-24: Architecture overview](models/architecture.md)

-   :material-cog:{ .lg .middle } **Training**

    ---

    One command for every model: `ecgen train <config.yaml> [key=value ...]`.

    [:octicons-arrow-right-24: Training with `ecgen train`](training/runner.md)

-   :material-api:{ .lg .middle } **API reference**

    ---

    Generated from the source docstrings.

    [:octicons-arrow-right-24: API reference](api/models.md)

</div>

## Model families

| Family | Package | `model.name` | Status | Docs |
|---|---|---|---|---|
| Variational Autoencoder | `ecgen.families.vae` | `vae` | Available | [VAE](models/vae.md) |
| VQ-VAE + PixelCNN prior | `ecgen.families.vqvae` | `vqvae`, `vqvae_prior` | Available | [VQ-VAE](models/vqvae.md) |
| Pulse2Pulse GAN | `ecgen.families.gan` | `pulse2pulse` | Available | [GAN](models/gan.md) |
| Diffusion | — | — | Planned | — |
| State Space Models | — | — | Planned | — |

## Datasets

| Dataset | Status | Docs |
|---|---|---|
| MIMIC-IV-ECG (`mimic_iv_ecg`) | Available | [MIMIC-IV-ECG](datasets/mimic.md) |
| Synthetic ECG-like signals (`synthetic`) | Available, for smoke tests | [Smoke runs](training/runner.md#smoke-runs-without-data) |
| PTB-XL | Planned (config only) | — |
| Your own data | — | [Adding a dataset](datasets/custom.md) |

## At a glance

```bash
git clone https://github.com/vlbthambawita/ECGEN.git
cd ECGEN
pip install -e .

# Train a VQ-VAE (stage 1); overrides follow the config
ecgen train configs/experiments/vqvae_mimic.yaml data.dataset.params.root=/path/to/mimic-iv-ecg

# then its prior (stage 2)
ecgen train configs/experiments/prior_mimic.yaml \
    model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
```

```python
from ecgen.families.vae import VAELightning

model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt")
samples = model.sample(n_samples=16, seq_length=5000)
```

!!! note
    See [Training with `ecgen train`](training/runner.md) for the CLI, the config schema and the run directory layout.
    The Lightning wrappers rebuild their config from the flat hyperparameters stored in the checkpoint, so `load_from_checkpoint(path)` needs no extra arguments. See [VAE](models/vae.md) for details.

## Repository layout

```text
ECGEN/
├── src/ecgen/       # installed package: families/, nn/, data/, training/ (runner, callbacks), cli.py
├── models/          # compatibility shims; diffusion/, ssm/ placeholders
├── data/            # compatibility shims for the old dataset paths
├── training/        # metrics/, losses/common.py, trainers/ (+ shims)
├── evaluation/      # metrics/, benchmark.py, visualize.py
├── visualization/   # ECG plots, latent space, training curves
├── utils/           # compatibility shims to ecgen.utils
├── scripts/         # shell/ (run_train_*.sh), generate/, evaluate/, data/
├── configs/         # dataset/, model/, training/, experiments/
├── single_file/     # self-contained VQ-VAE implementation
└── docs/            # this documentation
```

See [Project Structure](project_structure.md) for details.

## Building these docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://127.0.0.1:8000
```

Pushes to `main` or `simple` that touch the docs or the source packages rebuild the site and publish it to GitHub Pages.
