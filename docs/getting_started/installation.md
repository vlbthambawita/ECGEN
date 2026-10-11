# Installation

## Requirements

`pyproject.toml` declares:

- Python >= 3.8
- `torch>=1.10`, `pytorch-lightning>=1.5`
- `pyyaml`, `pandas`, `scikit-learn`, `wfdb`, `matplotlib`, `tensorboard`
- extras: `[hf]` (`huggingface_hub`, `python-dotenv`), `[plot]` (`cairosvg`), `[all]`, `[dev]`
  (pytest, black, flake8, mypy)

!!! note "Python version"
    Several modules (for example `visualization/ecg_plots.py`) use `X | Y` type hints at runtime
    without `from __future__ import annotations`, so in practice Python 3.10+ is required to import
    every package.

Optional: `wandb` is imported lazily by the runner and callbacks only when Weights & Biases
logging is enabled. It is not a declared dependency; install it yourself if you need it.

## Installation steps

### 1. Clone the repository

```bash
git clone https://github.com/vlbthambawita/ECGEN.git
cd ECGEN
```

### 2. Install the package

```bash
pip install -e .            # the ecgen package + the `ecgen` command
pip install -e ".[dev]"     # + pytest, black, flake8, mypy
```

Only the `ecgen` package in `src/ecgen/` is installed. It contains the model families
(`ecgen.families.*`), datasets (`ecgen.data.*`), the training runner (`ecgen.training.runner`) and
the `ecgen` console script. The research packages at the repository root (`models`, `data`,
`training`, `evaluation`, `utils`, `visualization`) are **not** installed: run code that imports
them from the repo root (pytest adds the root via `pythonpath = ["."]`). Their moved modules
(`models.vae`, `models.vqvae`, `models.gan.pulse2pulse`, `models.components`, `training.callbacks`,
`training.losses.*_losses`, `utils.*`) are re-export shims of the `ecgen` modules, kept for one
release.

### 3. Verify the installation

```bash
ecgen list
# family: pulse2pulse, vae, vqvae, vqvae_prior
# dataset: mimic_iv_ecg, synthetic
# callback: early_stopping, ecg_visualization, generated_samples, lr_monitor, model_checkpoint, vae_visualization

python -c "from ecgen.families import VAELightning, VQVAELightning, PriorLightning, Pulse2PulseGAN; print('ok')"
```

## Running the tests

```bash
pytest tests/
```

## Dataset setup

See [MIMIC-IV-ECG Dataset](../datasets/mimic.md) for dataset setup instructions.
