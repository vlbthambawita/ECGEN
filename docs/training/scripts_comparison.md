# Training Scripts Compared

All training entry points that exist in the repository, and how they differ.

## Overview

| Shell wrapper (`scripts/shell/`) | Python entry point | Model | Config? | Package layout |
|---|---|---|---|---|
| `run_train_vae_mimic_config.sh [config]` | `scripts/train/train_vae_mimic.py` | VAE | `--config` (default `configs/experiments/vae_mimic.yaml`) | New (`models.vae`), but its `VAEDataWrapper` imports legacy `ecgen.models.vae` |
| `run_train_vqvae_mimic.sh [config]` | `scripts/train/train_vqvae_mimic.py` | VQ-VAE stage 1 | `--config` (default `configs/experiments/vqvae_mimic.yaml`) | New |
| `run_train_vqvae_prior_mimic.sh <vqvae_ckpt> [config]` | `scripts/train/train_prior_mimic.py` | PixelCNN prior (stage 2) | `--config` (default `configs/experiments/prior_mimic.yaml`) + `--vqvae-checkpoint` | New |
| `run_train_pulse2pulse_mimic.sh [config]` | `scripts/train/train_pulse2pulse.py` | Pulse2Pulse GAN | passes `--config`, which this script does **not** accept (see below) | Legacy (`ecgen.*`) |
| `run_train_pulse2pulse_standalone.sh` | `scripts/train/train_pulse2pulse.py` | Pulse2Pulse GAN | Flags only, set via env vars `DATA_DIR`, `EXP_NAME`, `BATCH_SIZE`, `MAX_EPOCHS`, `GPU_ID`, `NUM_WORKERS` | Legacy (`ecgen.*`) |
| `run_sweep.sh` | loops over `configs/experiments/*.yaml` | — | Routes `*vae*` → `train_vae_mimic.py`, `*pulse2pulse*` → `train_pulse2pulse.py`, skips others | — |
| — | `training/trainers/gan_trainer.py --config <yaml>` | any (`target`-based) | Instantiates `model.target`, `data.target`, `callbacks[].target` | Imports legacy `ecgen.utils.*` |
| `single_file/run_train_vqvae.sh` | `single_file/train_vqvae_standalone.py --stage {1,2}` | VQ-VAE (both stages) | Flags only, set via env vars | Self-contained, no package imports |
| — | `scripts/train/train_template.py` | — | Skeleton for new models; prints a message only | — |

All wrappers in `scripts/shell/` first `cd` to the repository root, so they can be run from anywhere.

## Config-driven scripts (VAE, VQ-VAE, prior)

`train_vae_mimic.py`, `train_vqvae_mimic.py` and `train_prior_mimic.py` share one design:

- argparse flags with defaults, plus an optional `--config`
- if `--config` is given, specific YAML keys overwrite the corresponding flags
  (**YAML wins over CLI** for any key present in the file)
- a script-local `LightningDataModule` around `MIMICIVECGDataset`
- `ModelCheckpoint` (`epoch{epoch:03d}-step{step:06d}`, top-k on `val_loss`, `last.ckpt`),
  `EarlyStopping` on `val_loss`, `LearningRateMonitor`
- TensorBoard always, W&B optional
- outputs in `runs/<exp_name>/seed_<seed>/{checkpoints,samples,tb}`

Differences:

| | VAE | VQ-VAE | Prior |
|---|---|---|---|
| Module trained | `VAEDataWrapper` (script-local) | `VQVAELightning` | `PriorLightning` |
| Reconstruction plots | `VAEVisualizationCallback` | `VAEVisualizationCallback` | none |
| Reads `visualization:` | yes | yes | no |
| Extra required input | — | — | `--vqvae-checkpoint` (overridden by `model.params.config.vqvae_checkpoint` in the YAML) |
| Default W&B project | `ecg-vae` | `ecg-vqvae` | `ecg-vqvae-prior` |

Details: [VAE training](vae_training.md), [VQ-VAE training](vqvae_training.md),
[Configuration](../getting_started/configuration.md).

## Pulse2Pulse

`train_pulse2pulse.py` takes flags only (`--data-dir` is required) and imports
`Pulse2PulseGAN`, its datamodule, the callbacks and `set_global_seed` from the legacy `ecgen`
package. `run_train_pulse2pulse_standalone.sh` is therefore the wrapper that matches its
interface. `configs/experiments/pulse2pulse_mimic.yaml` is in the format expected by
`training/trainers/gan_trainer.py`. See [Pulse2Pulse training](pulse2pulse_training.md).

## Standalone VQ-VAE

`single_file/train_vqvae_standalone.py` re-implements the VQ-VAE, prior, dataset and training loop
in one file and is independent of `models/vqvae/`. See
[Standalone VQ-VAE](../standalone/vqvae_standalone.md).

## Which one to use

- **VAE**: `run_train_vae_mimic_config.sh` with an edited copy of `vae_mimic.yaml`.
- **VQ-VAE**: `run_train_vqvae_mimic.sh`, then `run_train_vqvae_prior_mimic.sh <checkpoint>`
  (set `vqvae_checkpoint` in `prior_mimic.yaml` to the same path, since the YAML value wins).
- **Pulse2Pulse**: `run_train_pulse2pulse_standalone.sh` (with `src/` on `PYTHONPATH`).
- **New model**: start from `train_template.py` or copy one of the config-driven scripts.
