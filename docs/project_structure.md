# Project Structure

`pip install -e .` installs one package, **`ecgen`** (`src/ecgen/`), plus the `ecgen` command. It
holds the model families, datasets, the training runner and the published generation/plotting
helpers. The research packages at the repository root (`models`, `data`, `training`, `evaluation`,
`utils`, `visualization`) are **not** installed; run code that uses them from the repo root. Several
of their modules are now one-line re-export shims of `ecgen` modules (see
[below](#compatibility-shims)).

## Directory layout

```
ECGEN/
├── src/ecgen/                      # the installed package
│   ├── cli.py, __main__.py         # `ecgen train ...`, `ecgen list`, `python -m ecgen`
│   ├── registry.py                 # Registry, FAMILIES, DATASETS, CALLBACKS, load_builtins
│   ├── types.py                    # ECGBatch, ConditionOutput, get_signal
│   ├── families/                   # one sub-package per model family
│   │   ├── base.py                 #   GenerativeModule, Capabilities
│   │   ├── vae/                    #   config.py, vae_1d.py, vae_lightning.py, losses.py
│   │   ├── vqvae/                  #   config.py, vector_quantizer.py, vqvae_1d.py, vqvae_lightning.py,
│   │   │                           #   pixelcnn_prior.py, prior_lightning.py, losses.py
│   │   └── gan/                    #   pulse2pulse.py, losses.py
│   ├── nn/                         # blocks.py (ResidualBlock1D), encoders.py, decoders.py
│   ├── data/
│   │   ├── datasets/               #   base.py (BaseECGDataset), mimic.py (MIMICIVECG), synthetic.py
│   │   ├── datamodule.py           #   ECGDataModule (generic LightningDataModule)
│   │   ├── collate.py              #   ECGCollate -> ECGBatch
│   │   ├── leads.py                #   STANDARD_12, INDEPENDENT_8, resolve_leads
│   │   ├── normalizers.py          #   FeatureNormalizer
│   │   └── mimic_dataset.py, pulse2pulse_mimic.py   # legacy loaders (tuple / GAN datamodule)
│   ├── training/
│   │   ├── runner.py               #   YAML-driven training for every family
│   │   ├── callbacks/              #   visualization.py, checkpointing.py, logging.py (placeholders)
│   │   └── train.py                #   deprecated `--config` wrapper around the runner
│   ├── utils/                      # config.py, seed.py, io.py, logging.py, metadata.py
│   ├── generate.py, plot.py        # ecgen.generate, ecgen.plot_ecg
│   ├── f/                          # Hugging Face upload helpers
│   └── models/                     # deprecated shims to ecgen.families
│
├── models/                         # shims (vae/, vqvae/, gan/, components/); diffusion/, ssm/ placeholders
├── data/datasets/                  # shims: base.py, mimic/dataset.py, mimic/pulse2pulse_datamodule.py
├── training/
│   ├── callbacks/                  # shims to ecgen.training.callbacks
│   ├── losses/                     # common.py; *_losses.py are shims to ecgen.families.*.losses
│   ├── metrics/                    # reconstruction.py, generation.py, clinical.py
│   └── trainers/                   # gan_trainer.py (deprecated wrapper), base/vae placeholders
├── evaluation/
│   ├── metrics/                    # fidelity.py, diversity.py, signal_quality.py
│   ├── benchmark.py                # ModelBenchmark
│   └── visualize.py                # plot_evaluation_results, plot_comparison_matrix
├── visualization/                  # ecg_plots.py, latent_space.py, training_curves.py
├── utils/                          # shims to ecgen.utils
│
├── configs/
│   ├── experiments/                # vae_mimic, vqvae_mimic, prior_mimic, pulse2pulse_mimic (.yaml)
│   ├── model/                      # vae/, vqvae/, gan/, imn.yaml  (reference fragments, not loaded)
│   ├── dataset/                    # mimic.yaml, ptbxl.yaml        (reference fragments, not loaded)
│   └── training/                   # default.yaml (empty), fast_dev.yaml, production.yaml
│
├── scripts/
│   ├── train/                      # train_template.py (points at the runner)
│   ├── shell/                      # run_train_*.sh wrappers around `python -m ecgen train`, run_sweep.sh
│   ├── generate/                   # generate_pulse2pulse.py, generate_vae.py (stub)
│   ├── evaluate/                   # evaluate_model.py, benchmark_models.py
│   ├── data/                       # prepare_mimic.py, validate_dataset.py (stubs)
│   └── utils/                      # test_models_only.py, test_pulse2pulse_setup.py
│
├── single_file/                    # self-contained VQ-VAE: train_vqvae_standalone.py, run_train_vqvae.sh
├── tests/                          # test_models/, test_data/, test_training/
├── docs/                           # this documentation (MkDocs)
├── runs/                           # training outputs (git-ignored)
└── pyproject.toml
```

## Module responsibilities

| Module | Contents |
|---------|----------|
| `ecgen.families` | `nn.Module`s and their `GenerativeModule` (Lightning) wrappers, configs and losses; one sub-package per family. See [Architecture](models/architecture.md). |
| `ecgen.data` | `BaseECGDataset` dict contract, `MIMICIVECG` and `SyntheticECG`, `ECGDataModule`, lead handling and normalizers. See [MIMIC-IV-ECG](datasets/mimic.md) and [Adding a dataset](datasets/custom.md). |
| `ecgen.training` | The runner (`ecgen train`) and callbacks. See [Training with `ecgen train`](training/runner.md). |
| `ecgen.registry` | Name → class registries used to build components from YAML. |
| `training` (root) | Training-time metrics, `losses/common.py`, placeholder trainers. |
| `evaluation` | Metrics and benchmarking for generated ECGs. See [Evaluation metrics](evaluation/metrics.md). |
| `visualization` | Matplotlib helpers for ECGs, latents and curves. See [Visualization](visualization/guide.md). |
| `ecgen.utils` | `set_global_seed`, YAML/JSON I/O (`read_yaml`, `write_yaml`, `write_json`), `get_logger`, `collect_run_metadata`, `resolve_config`. |

## Import patterns

```python
from ecgen.families.vae import VAE1D, VAELightning, VAEConfig, vae_loss
from ecgen.families.vqvae import VQVAELightning, PriorLightning, VQVAEConfig, PriorConfig
from ecgen.families.gan import Pulse2PulseGAN, Pulse2PulseConfig
from ecgen.nn import ResidualBlock1D, Encoder1D, Decoder1D

from ecgen.data.datasets import BaseECGDataset, MIMICIVECG, SyntheticECG
from ecgen.data.datamodule import ECGDataModule
from ecgen.types import ECGBatch, get_signal

from ecgen.training.callbacks import VAEVisualizationCallback, ECGVisualizationCallback, GeneratedSamplesCallback
from ecgen.training.runner import train_from_file
from ecgen.registry import FAMILIES, DATASETS, CALLBACKS
from ecgen.utils.seed import set_global_seed

from visualization import plot_ecg_leads, plot_latent_space_2d   # from the repo root
```

## Compatibility shims

These old paths re-export the moved modules and will be removed after one release:

| Old import | New import |
|---|---|
| `models.vae`, `models.vqvae` | `ecgen.families.vae`, `ecgen.families.vqvae` |
| `models.gan.pulse2pulse`, `ecgen.models.pulse2pulse` | `ecgen.families.gan.pulse2pulse` |
| `ecgen.models.vae` | `ecgen.families.vae` |
| `models.components` | `ecgen.nn` |
| `training.callbacks` | `ecgen.training.callbacks` |
| `training.losses.{vae,vqvae,gan}_losses` | `ecgen.families.{vae,vqvae,gan}.losses` |
| `utils.{config,io,logging,metadata,seed}` | `ecgen.utils.*` |
| `data.datasets.base` | `ecgen.data.datasets.base` |
| `data.datasets.mimic.dataset`, `data.datasets.mimic.pulse2pulse_datamodule` | legacy `ecgen.data.mimic_dataset`, `ecgen.data.pulse2pulse_mimic` |

`training/trainers/gan_trainer.py` and `ecgen.training.train` are deprecated wrappers around the
runner.

## Run outputs

```
runs/<experiment.name>/seed_<seed>/
├── config_resolved.yaml
├── metadata.json
├── checkpoints/          # epochNNN-stepNNNNNN.ckpt + last.ckpt
├── samples/              # visualization output
├── metrics/              # train_val_metrics.json
└── tb/                   # TensorBoard logs
```

## Quick reference

### Where is ...?

| Component | Location |
|-----------|----------|
| Training runner / CLI | `src/ecgen/training/runner.py`, `src/ecgen/cli.py` |
| Shared conv blocks | `src/ecgen/nn/` |
| VAE | `src/ecgen/families/vae/` |
| VQ-VAE and prior | `src/ecgen/families/vqvae/` |
| Pulse2Pulse | `src/ecgen/families/gan/pulse2pulse.py` |
| MIMIC dataset | `src/ecgen/data/datasets/mimic.py` |
| Callbacks | `src/ecgen/training/callbacks/visualization.py` |
| Losses | `src/ecgen/families/*/losses.py` |
| Experiment configs | `configs/experiments/` |
| Standalone VQ-VAE | `single_file/` |

### How do I ...?

| Task | Command |
|------|---------|
| Train the VAE | `ecgen train configs/experiments/vae_mimic.yaml` |
| Train the VQ-VAE (stage 1) | `ecgen train configs/experiments/vqvae_mimic.yaml` |
| Train the prior (stage 2) | `ecgen train configs/experiments/prior_mimic.yaml model.params.vqvae_checkpoint=<vqvae.ckpt>` |
| Train Pulse2Pulse | `ecgen train configs/experiments/pulse2pulse_mimic.yaml` |
| Smoke-test without data | add `data.dataset.name=synthetic 'data.dataset.params={num_samples: 64}'` |
| List registered components | `ecgen list` |
| View TensorBoard | `tensorboard --logdir runs/<exp>/seed_<seed>/tb` |
| Check reconstruction plots | `ls runs/<exp>/seed_<seed>/samples/` |
| Run tests | `pytest tests/` |

## Adding new components

- **Model family**: new sub-package under `src/ecgen/families/` (config dataclass + `nn.Module` +
  `GenerativeModule` wrapper + `losses.py`), registered with `@FAMILIES.register("<name>")`. See
  [Architecture](models/architecture.md#adding-a-new-model).
- **Dataset**: subclass `BaseECGDataset` and register it with `@DATASETS.register("<name>")`, or
  reference it with `target:`. See [Adding a dataset](datasets/custom.md).
- **Callback**: add it to `src/ecgen/training/callbacks/` and register a name in its `__init__.py`,
  or reference it with `target:`. See [Callbacks](training/callbacks.md#custom-callbacks).
- **Experiment**: a YAML under `configs/experiments/`; no training script is needed. See
  [Training with `ecgen train`](training/runner.md).
