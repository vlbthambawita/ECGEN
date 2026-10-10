# Project Structure

ECGEN is installed as the `ecgen` distribution (`pip install -e .`), which provides six **top-level**
packages: `models`, `data`, `training`, `evaluation`, `utils`, `visualization`. A legacy package under
`src/ecgen/` still exists and is used by a few scripts (see [below](#legacy-srcecgen)).

## Directory layout

```
ECGEN/
├── models/                         # Model architectures
│   ├── components/                 # Shared blocks
│   │   ├── blocks.py               #   ResidualBlock1D
│   │   ├── encoders.py             #   Encoder1D
│   │   └── decoders.py             #   Decoder1D
│   ├── vae/                        # VAE
│   │   ├── config.py               #   VAEConfig
│   │   ├── vae_1d.py               #   VAE1D
│   │   └── vae_lightning.py        #   VAELightning
│   ├── vqvae/                      # VQ-VAE + PixelCNN prior
│   │   ├── config.py               #   VQVAEConfig, PriorConfig
│   │   ├── vector_quantizer.py     #   VectorQuantizer
│   │   ├── vqvae_1d.py             #   VQVAE1D
│   │   ├── vqvae_lightning.py      #   VQVAELightning (stage 1)
│   │   ├── pixelcnn_prior.py       #   CausalGatedBlock, PixelCNNPrior
│   │   └── prior_lightning.py      #   PriorLightning (stage 2)
│   ├── gan/
│   │   └── pulse2pulse.py          #   WaveGANGenerator, WaveGANDiscriminator, Pulse2PulseConfig, Pulse2PulseGAN
│   ├── diffusion/                  # placeholder
│   └── ssm/                        # placeholder
│
├── data/
│   └── datasets/
│       ├── base.py                 # BaseECGDataset (abstract)
│       └── mimic/
│           ├── dataset.py          # MIMICIVECGDataset
│           ├── datamodule.py       # ECGDataModule (minimal dataclass, not a LightningDataModule)
│           └── pulse2pulse_datamodule.py   # imports the legacy ecgen dataset
│
├── training/
│   ├── callbacks/
│   │   ├── visualization.py        # ECGVisualizationCallback, GeneratedSamplesCallback, VAEVisualizationCallback
│   │   ├── checkpointing.py        # placeholder
│   │   └── logging.py              # placeholder
│   ├── losses/                     # vae_losses.py, vqvae_losses.py, gan_losses.py, common.py
│   ├── metrics/                    # reconstruction.py, generation.py, clinical.py
│   └── trainers/                   # gan_trainer.py (generic config trainer), base/vae placeholders
│
├── evaluation/
│   ├── metrics/                    # fidelity.py, diversity.py, signal_quality.py
│   ├── benchmark.py                # ModelBenchmark
│   └── visualize.py                # plot_evaluation_results, plot_comparison_matrix
│
├── visualization/                  # ecg_plots.py, latent_space.py, training_curves.py
├── utils/                          # seed.py, io.py, logging.py, metadata.py
│
├── configs/
│   ├── experiments/                # vae_mimic, vqvae_mimic, prior_mimic, pulse2pulse_mimic (.yaml)
│   ├── model/                      # vae/, vqvae/, gan/, imn.yaml  (reference fragments, not loaded)
│   ├── dataset/                    # mimic.yaml, ptbxl.yaml        (reference fragments, not loaded)
│   └── training/                   # default.yaml (empty), fast_dev.yaml, production.yaml
│
├── scripts/
│   ├── train/                      # train_vae_mimic.py, train_vqvae_mimic.py, train_prior_mimic.py,
│   │                               # train_pulse2pulse.py, train_template.py
│   ├── shell/                      # run_train_*.sh wrappers, run_sweep.sh
│   ├── generate/                   # generate_pulse2pulse.py, generate_vae.py (stub)
│   ├── evaluate/                   # evaluate_model.py, benchmark_models.py
│   ├── data/                       # prepare_mimic.py, validate_dataset.py (stubs)
│   └── utils/                      # test_models_only.py, test_pulse2pulse_setup.py
│
├── single_file/                    # self-contained VQ-VAE: train_vqvae_standalone.py, run_train_vqvae.sh
├── src/ecgen/                      # legacy package (not installed)
├── tests/                          # test_models/, test_data/, test_training/
├── docs/                           # this documentation (MkDocs)
├── runs/                           # training outputs (git-ignored)
└── pyproject.toml
```

## Module responsibilities

| Package | Contents |
|---------|----------|
| `models` | `nn.Module`s and their `LightningModule` wrappers; one sub-package per family. See [Architecture](models/architecture.md). |
| `data` | Datasets. `MIMICIVECGDataset` reads WFDB records, does a seeded subject-level train/val/test split, and returns `(ecg, features)` where `ecg` is `(num_leads, seq_length)` (default 12 × 5000) and `features` are 9 normalised machine measurements. See [MIMIC-IV-ECG](datasets/mimic.md) and [Custom datasets](datasets/custom.md). |
| `training` | Callbacks, losses, training-time metrics, trainers. See [Training guide](training/guide.md). |
| `evaluation` | Metrics and benchmarking for generated ECGs. See [Evaluation metrics](evaluation/metrics.md). |
| `visualization` | Matplotlib helpers for ECGs, latents and curves. See [Visualization](visualization/guide.md). |
| `utils` | `set_global_seed`, YAML/JSON I/O (`read_yaml`, `write_yaml`, `write_json`), `get_logger`, `collect_run_metadata`. |

## Import patterns

```python
from models.vae import VAE1D, VAELightning, VAEConfig
from models.vqvae import VQVAELightning, PriorLightning, VQVAEConfig, PriorConfig
from models.gan.pulse2pulse import Pulse2PulseGAN, Pulse2PulseConfig   # models.gan exports nothing
from models.components import ResidualBlock1D, Encoder1D, Decoder1D

from data.datasets.mimic.dataset import MIMICIVECGDataset   # or: from data.datasets.mimic import MIMICIVECGDataset
from data.datasets.base import BaseECGDataset

from training.callbacks import VAEVisualizationCallback, ECGVisualizationCallback, GeneratedSamplesCallback
from training.losses import vae_loss, vqvae_loss, prior_loss

from visualization import plot_ecg_leads, plot_latent_space_2d
from utils.seed import set_global_seed
```

The old `ecgen.*` imports (`from ecgen.models.vae import ...`, `from ecgen.training.callbacks import ...`)
refer to the legacy package. See
[`MIGRATION_GUIDE.md`](https://github.com/vlbthambawita/ECGEN/blob/main/MIGRATION_GUIDE.md) for the
old → new mapping.

## Legacy `src/ecgen/`

`src/ecgen/` is not listed in `pyproject.toml`, so it is not installed, but these files still import
it and need `PYTHONPATH=src`:

- `scripts/train/train_vae_mimic.py` (its `VAEDataWrapper`)
- `scripts/train/train_pulse2pulse.py`
- `scripts/utils/test_models_only.py`
- `training/trainers/gan_trainer.py`
- `data/datasets/mimic/pulse2pulse_datamodule.py`
- `tests/test_models/test_vae.py`
- the `target:` strings in `configs/experiments/vae_mimic.yaml` and `pulse2pulse_mimic.yaml`

## Run outputs

```
runs/<experiment.name>/seed_<seed>/
├── checkpoints/          # epochNNN-stepNNNNNN.ckpt + last.ckpt
├── samples/              # visualization PNGs
├── tb/                   # TensorBoard logs
├── config_resolved.yaml  # gan_trainer.py only
└── metadata.json         # gan_trainer.py only (also creates metrics/ and preds/)
```

## Quick reference

### Where is ...?

| Component | Location |
|-----------|----------|
| Shared conv blocks | `models/components/` |
| VAE | `models/vae/` |
| VQ-VAE and prior | `models/vqvae/` |
| Pulse2Pulse | `models/gan/pulse2pulse.py` |
| MIMIC dataset | `data/datasets/mimic/dataset.py` |
| Callbacks | `training/callbacks/visualization.py` |
| Losses | `training/losses/` |
| Training scripts | `scripts/train/` (wrappers in `scripts/shell/`) |
| Experiment configs | `configs/experiments/` |
| Standalone VQ-VAE | `single_file/` |

### How do I ...?

| Task | Command |
|------|---------|
| Train the VAE | `./scripts/shell/run_train_vae_mimic_config.sh [config]` |
| Train the VQ-VAE (stage 1) | `./scripts/shell/run_train_vqvae_mimic.sh [config]` |
| Train the prior (stage 2) | `./scripts/shell/run_train_vqvae_prior_mimic.sh <vqvae.ckpt> [config]` |
| Train Pulse2Pulse | `./scripts/shell/run_train_pulse2pulse_standalone.sh` (see [Pulse2Pulse training](training/pulse2pulse_training.md)) |
| View TensorBoard | `tensorboard --logdir runs/<exp>/seed_<seed>/tb` |
| Check reconstruction plots | `ls runs/<exp>/seed_<seed>/samples/` |
| Run tests | `pytest tests/` |

## Adding new components

- **Model**: new sub-package under `models/` (config dataclass + `nn.Module` + LightningModule),
  exported from its `__init__.py`. See [Architecture](models/architecture.md#adding-a-new-model).
- **Dataset**: subclass `BaseECGDataset` in `data/datasets/<name>/`. See
  [Custom datasets](datasets/custom.md).
- **Callback**: add it to `training/callbacks/` and export it from `training/callbacks/__init__.py`.
- **Training script**: copy `scripts/train/train_template.py` or a `train_*_mimic.py` script, and add a
  wrapper under `scripts/shell/` and a config under `configs/experiments/`.
