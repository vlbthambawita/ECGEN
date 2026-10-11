# Training Entry Points

All families are trained by one runner, `ecgen train <config.yaml>` (see
[Training with `ecgen train`](runner.md)). The per-model scripts
`scripts/train/train_{vae_mimic,vqvae_mimic,prior_mimic,pulse2pulse}.py` and their script-local
DataModules were removed.

| Entry point | What it runs |
|---|---|
| `ecgen train <config> [key=value ...]` / `python -m ecgen train ...` | The runner, for any family |
| `scripts/shell/run_train_vae_mimic_config.sh [config] [overrides]` | VAE (`vae_mimic.yaml`) |
| `scripts/shell/run_train_vqvae_mimic.sh [config] [overrides]` | VQ-VAE stage 1 (`vqvae_mimic.yaml`) |
| `scripts/shell/run_train_vqvae_prior_mimic.sh <vqvae_ckpt> [config] [overrides]` | PixelCNN prior (`prior_mimic.yaml`, `model.params.vqvae_checkpoint=<vqvae_ckpt>`) |
| `scripts/shell/run_train_pulse2pulse_mimic.sh [config] [overrides]` | Pulse2Pulse (`pulse2pulse_mimic.yaml`) |
| `scripts/shell/run_train_pulse2pulse_standalone.sh` | Pulse2Pulse, settings from env vars `DATA_DIR`, `EXP_NAME`, `BATCH_SIZE`, `MAX_EPOCHS`, `GPU_ID`, `NUM_WORKERS` |
| `scripts/shell/run_sweep.sh [overrides]` | Every `configs/experiments/*.yaml` except the prior, then the prior on `runs/vqvae_mimic/seed_42/checkpoints/last.ckpt` if it exists |
| `training/trainers/gan_trainer.py --config <yaml>`, `python -m ecgen.training.train --config <yaml>` | Deprecated wrappers that call the runner |
| `single_file/run_train_vqvae.sh` → `single_file/train_vqvae_standalone.py --stage {1,2}` | Self-contained VQ-VAE (both stages), flags/env vars only, no package imports |

All wrappers in `scripts/shell/` first `cd` to the repository root, so they can be run from anywhere,
and pass extra arguments through to `ecgen train` as overrides. Overrides are applied after the YAML,
so the command line always wins.

`scripts/train/train_template.py` now only points at the runner: to add a model, register a family
and write a config instead of a new script (see
[Extending](runner.md#extending-new-families-datasets-and-callbacks)).

## Standalone VQ-VAE

`single_file/train_vqvae_standalone.py` re-implements the VQ-VAE, prior, dataset and training loop
in one file and is independent of `ecgen.families.vqvae`. See
[Standalone VQ-VAE](../standalone/vqvae_standalone.md).
