# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ECGEN is a research framework for generative modeling of 12-lead ECGs (MIMIC-IV-ECG; PTB-XL planned), built on PyTorch Lightning. Implemented model families: VAE, VQ-VAE + PixelCNN prior, and Pulse2Pulse GAN. `models/diffusion/` and `models/ssm/` are placeholders.

## Commands

```bash
pip install -e .            # installs top-level packages: models, data, training, evaluation, utils, visualization
pip install -e ".[dev]"     # + pytest, black, flake8, mypy

pytest tests/                                          # all tests
pytest tests/test_models/test_vae.py::<test_name>      # single test
black . && flake8 .

# Training: shell wrappers cd to repo root and call scripts/train/*.py --config <yaml>
./scripts/shell/run_train_vae_mimic_config.sh          # configs/experiments/vae_mimic.yaml
./scripts/shell/run_train_vqvae_mimic.sh [config]      # VQ-VAE stage 1
./scripts/shell/run_train_vqvae_prior_mimic.sh         # VQ-VAE stage 2 (PixelCNN prior)
./scripts/shell/run_train_pulse2pulse_mimic.sh
python scripts/train/train_vqvae_mimic.py --config configs/experiments/vqvae_mimic.yaml
```

Training runs on GPU against MIMIC-IV-ECG at a hard-coded cluster path (`data.params.mimic_path` in the experiment YAMLs). Set `data.params.max_samples` (e.g. 1000) for quick smoke runs.

## Architecture

**Two code layouts coexist — be careful which one you touch.**
- **New layout (current):** top-level packages `models/`, `data/`, `training/`, `evaluation/`, `visualization/`, `utils/`. Imports look like `from models.vqvae import VQVAELightning`, `from data.datasets.mimic.dataset import MIMICIVECGDataset`. See `MIGRATION_GUIDE.md` for the old→new import mapping.
- **Legacy layout:** `src/ecgen/` (imported as `ecgen.*`). It is *not* in `pyproject.toml`'s package list, yet some code still imports it: `tests/test_models/test_vae.py`, `scripts/train/train_pulse2pulse.py`, `scripts/generate/generate_pulse2pulse.py`, `scripts/utils/test_models_only.py`, `scripts/train/train_vae_mimic.py` (its `VAEDataWrapper`), `training/trainers/gan_trainer.py`, and `data/datasets/mimic/pulse2pulse_datamodule.py`. Those paths only work with `PYTHONPATH=src` (the scripts' own `sys.path` hack resolves to the nonexistent `scripts/src`).
- **`single_file/`:** a self-contained VQ-VAE (both stages) in `train_vqvae_standalone.py` with its own model/data/training code and `run_train_vqvae.sh`. Independent of the packages above; changes to `models/vqvae/` are not reflected there and vice versa.

**Model package pattern** (`models/vae/`, `models/vqvae/`): a dataclass config (`config.py`), a plain `nn.Module` (`*_1d.py`), and a `LightningModule` wrapper (`*_lightning.py`) that owns loss, optimizers, and `sample()`. Shared conv blocks live in `models/components/`; losses in `training/losses/` (`vae_losses.py`, `vqvae_losses.py`, `gan_losses.py`).

**VQ-VAE is two-stage:** `VQVAELightning` (encoder → `VectorQuantizer` → decoder) is trained first; `PriorLightning` then loads a frozen stage-1 checkpoint via `vqvae_checkpoint` in `configs/experiments/prior_mimic.yaml` and trains a `PixelCNNPrior` over the discrete code indices. Sampling = prior samples codes → VQ-VAE decodes.

**Configs:** `configs/experiments/*.yaml` are the entry points, with sections `experiment` (name, seed, runs_root), `wandb`, `model` (`target` + `params.config`), `data` (`target` + `params`), `callbacks`, `trainer`, `visualization`. Despite the `target:` fields, the train scripts do not instantiate generically — each script reads specific keys out of the YAML and maps them onto its argparse args, so a new config key needs matching handling in the script. `configs/{model,dataset,training}/` hold reusable fragments; `scripts/train/train_template.py` is the starting point for new models.

**Data:** `data/datasets/base.py` defines a `BaseECGDataset` (dict-returning) contract, but `MIMICIVECGDataset` is a plain `torch.utils.data.Dataset` that does *not* subclass it and returns a tuple `(ecg, features)`. Batch format is inconsistent across consumers (`VAEVisualizationCallback` expects tuples; `VAELightning` and Pulse2Pulse expect dicts) — check before wiring a dataset to a model. The dataset reads WFDB records, does a seeded subject-level train/val/test split internally (selected by `split=`), and yields `(num_leads, seq_length)` tensors (default 12 × 5000). Train scripts define their own `LightningDataModule` inline around it.

**Outputs:** runs go to `runs/<exp_name>/seed_<seed>/` (checkpoints named `epoch###-step######.ckpt` plus `last.ckpt` — there is no `best.ckpt`; TensorBoard logs under `tb/`). W&B logging is optional and enabled via the `wandb` config section. `training/callbacks/visualization.py` plots reconstructions/samples during training.

**Known pitfalls:** Lightning wrappers call `save_hyperparameters(config.__dict__)`, so plain `X.load_from_checkpoint(path)` fails with a missing-`config` TypeError — pass `config=<Config>(**ckpt["hyper_parameters"])`. The VQ-VAE/VAE downsample by `2 ** (len(channel_multipliers) - 1)` (8× by default), not 16×. In the train scripts, YAML values override CLI flags, and the YAML `callbacks:` section is ignored.

## Documentation

MkDocs Material site in `docs/` (config: `mkdocs.yml`; API pages in `docs/api/` are rendered by mkdocstrings from Google-style docstrings). Preview with `pip install -r requirements-docs.txt && mkdocs serve`; CI (`.github/workflows/docs.yml`) runs `mkdocs build --strict` and publishes to the `gh-pages` branch on pushes to `main`/`simple`. New pages must be added to `nav` in `mkdocs.yml`.
