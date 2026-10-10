---
hide:
  - navigation
---

# ECGEN

**ECGEN** is a research framework for generative modeling of **12-lead electrocardiograms**, built on [PyTorch Lightning](https://lightning.ai/). It provides model implementations, dataset loaders, training scripts, evaluation metrics and visualization tools under one config-driven workflow.

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

    YAML-configured experiments, callbacks, TensorBoard and W&B logging.

    [:octicons-arrow-right-24: Training guide](training/guide.md)

-   :material-api:{ .lg .middle } **API reference**

    ---

    Generated from the source docstrings.

    [:octicons-arrow-right-24: API reference](api/models.md)

</div>

## Model families

| Family | Package | Status | Docs |
|---|---|---|---|
| Variational Autoencoder | `models.vae` | Available | [VAE](models/vae.md) |
| VQ-VAE + PixelCNN prior | `models.vqvae` | Available | [VQ-VAE](models/vqvae.md) |
| Pulse2Pulse GAN | `models.gan` | Available | [GAN](models/gan.md) |
| Diffusion | `models.diffusion` | Planned | — |
| State Space Models | `models.ssm` | Planned | — |

## Datasets

| Dataset | Status | Docs |
|---|---|---|
| MIMIC-IV-ECG | Available | [MIMIC-IV-ECG](datasets/mimic.md) |
| PTB-XL | Planned (config only) | — |
| Your own data | — | [Adding a dataset](datasets/custom.md) |

## At a glance

```bash
git clone https://github.com/vlbthambawita/ECGEN.git
cd ECGEN
pip install -e .

# Train a VQ-VAE (stage 1) with configs/experiments/vqvae_mimic.yaml
./scripts/shell/run_train_vqvae_mimic.sh
```

```python
from models.vae import VAELightning

model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt")
samples = model.sample(n_samples=16, seq_length=5000)
```

!!! note
    See the [training guide](training/guide.md) for every model's entry point and its known caveats.
    The Lightning wrappers rebuild their config from the flat hyperparameters stored in the checkpoint, so `load_from_checkpoint(path)` needs no extra arguments. See [VAE](models/vae.md) for details.

## Repository layout

```text
ECGEN/
├── models/          # vae/, vqvae/, gan/, diffusion/, ssm/, components/
├── data/            # datasets/ (base class + MIMIC-IV-ECG)
├── training/        # callbacks/, losses/, metrics/, trainers/
├── evaluation/      # metrics/, benchmark.py, visualize.py
├── visualization/   # ECG plots, latent space, training curves
├── utils/           # seeding, I/O, logging, run metadata
├── scripts/         # train/, generate/, evaluate/, data/, shell/
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
