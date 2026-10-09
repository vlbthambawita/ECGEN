# VAE Training

The VAE is trained on MIMIC-IV-ECG with
[`scripts/train/train_vae_mimic.py`](https://github.com/vlbthambawita/ECGEN/blob/main/scripts/train/train_vae_mimic.py),
either from a YAML config or from command-line flags.

## Prerequisites

- `pip install -e .` (see [Installation](../getting_started/installation.md))
- The legacy `ecgen` package on the path — the script's `VAEDataWrapper` does
  `from ecgen.models.vae import VAE1D, vae_loss`:

    ```bash
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    ```

- MIMIC-IV-ECG on disk, laid out as described in [MIMIC-IV-ECG](../datasets/mimic.md)
  (`machine_measurements.csv` plus `files/p<XXXX>/p<subject_id>/s<study_id>/<study_id>.hea/.dat`)

## Quick start

=== "Shell wrapper (config)"

    ```bash
    ./scripts/shell/run_train_vae_mimic_config.sh                         # configs/experiments/vae_mimic.yaml
    ./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
    ```

    The wrapper `cd`s to the repo root and runs `python scripts/train/train_vae_mimic.py --config <yaml>`.
    It accepts only the config path; extra flags are not forwarded.

=== "Python (config)"

    ```bash
    python scripts/train/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml
    ```

=== "Python (flags only)"

    ```bash
    python scripts/train/train_vae_mimic.py \
        --data-dir /path/to/mimic-iv-ecg \
        --exp-name vae_smoke \
        --max-samples 1000 \
        --max-epochs 10
    ```

For a smoke run with a config, set `data.params.max_samples` (e.g. `1000`) and a small
`trainer.max_epochs` in the YAML. Remember that YAML values override CLI flags (see
[Configuration](../getting_started/configuration.md#how-configs-are-applied)), so editing the YAML
is the reliable way to change a setting that the YAML already contains.

## What the script does

1. Loads the YAML (if given) and copies the supported keys onto the argparse arguments.
2. `set_global_seed(seed)` (from `utils.seed`).
3. Creates `runs/<exp_name>/seed_<seed>/{checkpoints,samples}`.
4. Builds `VAEConfig` (with `channel_multipliers` fixed to `(1, 2, 4, 4)`) and a `VAEDataWrapper`
   LightningModule — a script-local equivalent of `VAELightning` that accepts the
   `(ecg, features)` tuples produced by `MIMICIVECGDataset`.
5. Builds a script-local `VAEMIMICDataModule` that creates `MIMICIVECGDataset` for the
   `train` / `val` / `test` splits (subject-level split with `val_split`, `test_split`, `seed`;
   `max_samples` caps **each** split). Train loader shuffles; all loaders use `pin_memory=True`.
6. Loggers: `TensorBoardLogger(save_dir=run_dir, name="tb")` always; `WandbLogger` if W&B is enabled
   (`log_model=True`, `save_dir=run_dir`).
7. Callbacks:
    - `ModelCheckpoint(dirpath=checkpoints/, filename="epoch{epoch:03d}-step{step:06d}", monitor="val_loss", mode="min", save_top_k=--save-top-k, save_last=True)`
    - `EarlyStopping(monitor="val_loss", patience=--patience, mode="min")`
    - `LearningRateMonitor(logging_interval="step")`
    - `VAEVisualizationCallback(save_dir=samples/, ...)` — see [Visualization](../visualization/guide.md)
8. `pl.Trainer(..., gradient_clip_val=--gradient-clip, deterministic=True)` and
   `trainer.fit(model, datamodule, ckpt_path=--resume)`.

The optimiser is Adam (`lr`, `b1`, `b2`) with no scheduler. Logged metrics are the same as
`VAELightning`: `train/{total,recon,kl}_loss`, `val/{total,recon,kl}_loss`, `val_loss`.

## Command-line flags

| Flag | Default | YAML key |
|------|---------|----------|
| `--config` | none | — |
| `--exp-name` | `vae_mimic` | `experiment.name` |
| `--seed` | `42` | `experiment.seed` |
| `--runs-root` | `runs` | `experiment.runs_root` |
| `--data-dir` | none (required without `--config`) | `data.params.mimic_path` |
| `--batch-size` | `32` | `data.params.batch_size` |
| `--num-workers` | `4` | `data.params.num_workers` |
| `--max-samples` | none (all) | `data.params.max_samples` |
| `--skip-missing-check` | off | `data.params.skip_missing_check` |
| `--val-split` / `--test-split` | `0.1` / `0.1` | `data.params.val_split` / `test_split` |
| `--seq-length` | `5000` | `data.params.seq_length` |
| `--in-channels` | `12` | `model.params.config.in_channels` |
| `--base-channels` | `64` | `model.params.config.base_channels` |
| `--latent-channels` | `8` | `model.params.config.latent_channels` |
| `--num-res-blocks` | `2` | `model.params.config.num_res_blocks` |
| `--lr` | `1e-4` | `model.params.config.lr` |
| `--kl-weight` | `0.0001` | `model.params.config.kl_weight` |
| `--b1` / `--b2` | `0.9` / `0.999` | `model.params.config.b1` / `b2` |
| `--max-epochs` | `100` | `trainer.max_epochs` |
| `--accelerator` | `gpu` | `trainer.accelerator` |
| `--devices` | `0` (list of ints) | `trainer.devices` |
| `--log-every-n-steps` | `50` | `trainer.log_every_n_steps` |
| `--check-val-every-n-epoch` | `1` | `trainer.check_val_every_n_epoch` |
| `--gradient-clip` | `1.0` | `trainer.gradient_clip_val` |
| `--patience` | `10` | — (CLI only) |
| `--save-top-k` | `3` | — (CLI only) |
| `--resume` | none | — (CLI only) |
| `--wandb` | off | `wandb.enabled` |
| `--wandb-project` | `ecg-vae` | `wandb.project` |
| `--wandb-entity` | none | `wandb.entity` |
| `--wandb-run-name` | `<exp_name>_seed<seed>` | `wandb.run_name` |
| `--wandb-tags` | `vae ecg` | `wandb.tags` |
| `--viz-every-n-epochs` | `5` | `visualization.every_n_epochs` |
| `--viz-num-samples` | `4` | `visualization.num_samples` |
| `--viz-plot-all-leads` | off (overlay) | `visualization.plot_all_leads` |
| `--viz-log-to-tensorboard` | on (cannot be switched off from the CLI) | `visualization.log_to_tensorboard` |
| `--viz-log-to-wandb` | off | `visualization.log_to_wandb` |

Note that `configs/experiments/vae_mimic.yaml` sets `check_val_every_n_epoch: 5`, `wandb.enabled: true`,
`visualization.log_to_tensorboard: false` and `visualization.log_to_wandb: true`, which differ from
the flag defaults.

!!! warning "No reconstruction plots with the shipped config"
    Lightning validates on 0-based epochs 4, 9, 14, ... when `check_val_every_n_epoch: 5`, while
    `VAEVisualizationCallback` only plots when `epoch % every_n_epochs == 0`. With both set to 5 (as in
    `vae_mimic.yaml`) the callback never fires. Set `visualization.every_n_epochs: 1`.

### Examples

```bash
# Smaller, faster model
python scripts/train/train_vae_mimic.py --data-dir /path/to/data \
    --base-channels 32 --batch-size 64 --max-epochs 50

# Weights & Biases
python scripts/train/train_vae_mimic.py --data-dir /path/to/data \
    --wandb --wandb-project my-project --viz-log-to-wandb

# Multiple GPUs
python scripts/train/train_vae_mimic.py --data-dir /path/to/data --devices 0 1

# Resume (works together with --config)
python scripts/train/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml \
    --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
```

## Outputs

```
runs/<exp_name>/seed_<seed>/
├── checkpoints/
│   ├── epochNNN-stepNNNNNN.ckpt   # top-k by val_loss
│   └── last.ckpt
├── samples/
│   └── epoch_NNNN.png             # VAEVisualizationCallback
├── tb/version_N/                  # TensorBoard
└── wandb/                         # only if W&B is enabled
```

This script does **not** write `config_resolved.yaml` or `metadata.json` (only
`training/trainers/gan_trainer.py` does).

Monitor with:

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

## Using a trained model

```python
import torch
from models.vae import VAELightning, VAEConfig

ckpt = torch.load("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
model = VAELightning(VAEConfig(**ckpt["hyper_parameters"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

samples = model.sample(n_samples=16, seq_length=5000)   # generate
recon = model.reconstruct(ecg_batch)                     # (B, 12, 5000) -> (B, 12, 5000)
latent = model.vae.encode_to_latent(ecg_batch)           # (B, 8, 625)
```

!!! note
    `scripts/generate/generate_vae.py` is a stub that only prints a message; use the snippet above
    to generate samples.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'ecgen'` | `export PYTHONPATH="$PWD/src:$PYTHONPATH"` (the script still imports the legacy package) |
| `Error: Either --config or --data-dir must be provided` | Pass one of them |
| `FileNotFoundError: machine_measurements.csv not found at ...` | `mimic_path` / `--data-dir` must point at the directory containing `machine_measurements.csv` and `files/` |
| A CLI flag seems to be ignored | The YAML has that key; YAML values win — edit the YAML instead |
| `Warning: wandb not installed` | The script continues with TensorBoard only; `pip install wandb` and log in, or set `wandb.enabled: false` |
| CUDA out of memory | Lower `batch_size`, `base_channels` or `num_res_blocks` |
| Slow start-up | The missing-file check stats every record; set `skip_missing_check: true` once the data is known to be complete |
| Blurry reconstructions | Lower `kl_weight`, or increase model capacity |
| KL collapses to ~0 / samples are poor | Raise `kl_weight` |

## Related

- [VAE model](../models/vae.md)
- [Configuration](../getting_started/configuration.md)
- [Training scripts compared](scripts_comparison.md)
- [Callbacks](callbacks.md) and [Visualization](../visualization/guide.md)
