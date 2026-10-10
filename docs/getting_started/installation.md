# Installation

## Requirements

`pyproject.toml` declares:

- Python >= 3.8
- `torch>=1.10`, `pytorch-lightning>=1.5`
- `pyyaml`, `pandas`, `scikit-learn`, `wfdb`, `matplotlib`, `scipy`, `numpy`

!!! note "Python version"
    Several modules (for example `visualization/ecg_plots.py`) use `X | Y` type hints at runtime
    without `from __future__ import annotations`, so in practice Python 3.10+ is required to import
    every package.

Optional: `wandb` is imported lazily by the training scripts and callbacks only when Weights & Biases
logging is enabled. It is not a declared dependency; install it yourself if you need it.

## Installation steps

### 1. Clone the repository

```bash
git clone https://github.com/vlbthambawita/ECGEN.git
cd ECGEN
```

### 2. Install the package

```bash
pip install -e .            # installs the top-level packages
pip install -e ".[dev]"     # + pytest, black, flake8, mypy
```

The distribution is named `ecgen`, but it installs six **top-level** packages (see
`[tool.setuptools] packages` in
[`pyproject.toml`](https://github.com/vlbthambawita/ECGEN/blob/main/pyproject.toml)):
`models`, `data`, `training`, `evaluation`, `utils`, `visualization`. Imports therefore look like
`from models.vae import VAELightning`, not `from ecgen.models...`.

### 3. Verify the installation

```bash
python -c "import models, data, training, evaluation, utils, visualization; print('ok')"
python -c "from models.vae import VAELightning; from models.vqvae import VQVAELightning; print('ok')"
```

!!! warning "Legacy `ecgen` package"
    The old `src/ecgen/` package is **not** installed by `pip install -e .`, but some code still
    imports `ecgen.*`: `scripts/train/train_vae_mimic.py` (inside its `VAEDataWrapper`),
    `scripts/train/train_pulse2pulse.py`,
    `scripts/utils/test_models_only.py`, `training/trainers/gan_trainer.py`,
    `data/datasets/mimic/pulse2pulse_datamodule.py` and `tests/test_models/test_vae.py`.
    To run those, put `src/` on the path:

    ```bash
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    ```

## Running the tests

```bash
pytest tests/
```

## Dataset setup

See [MIMIC-IV-ECG Dataset](../datasets/mimic.md) for dataset setup instructions.
