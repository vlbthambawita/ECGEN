# VQ-VAE Training

The [VQ-VAE](../models/vqvae.md) is trained in two separate runs:

| Stage | What is trained | Python script | Shell wrapper | Experiment config |
|---|---|---|---|---|
| 1 | `VQVAELightning` (encoder, codebook, decoder) | `scripts/train/train_vqvae_mimic.py` | `scripts/shell/run_train_vqvae_mimic.sh` | `configs/experiments/vqvae_mimic.yaml` |
| 2 | `PriorLightning` (PixelCNN over code indices; VQ-VAE frozen) | `scripts/train/train_prior_mimic.py` | `scripts/shell/run_train_vqvae_prior_mimic.sh` | `configs/experiments/prior_mimic.yaml` |

Stage 2 needs a stage-1 checkpoint. Both stages train on MIMIC-IV-ECG. See
[MIMIC-IV-ECG](../datasets/mimic.md) for the dataset.

`PriorLightning` loads the stage-1 model with a plain `VQVAELightning.load_from_checkpoint(path)`
and records its config in the prior's hyperparameters, so a finished prior checkpoint can be
restored on its own with `PriorLightning.load_from_checkpoint(path)`. See
[Loading checkpoints](../models/vqvae.md#loading-checkpoints).

## Quick start

```bash
# Stage 1: VQ-VAE
./scripts/shell/run_train_vqvae_mimic.sh
#   or with another config:
./scripts/shell/run_train_vqvae_mimic.sh configs/experiments/my_vqvae.yaml

# Find a real checkpoint (there is no best.ckpt, see "Checkpoint names" below)
ls runs/vqvae_mimic/seed_42/checkpoints/

# Stage 2: PixelCNN prior. The checkpoint argument overrides the
# vqvae_checkpoint path in prior_mimic.yaml
./scripts/shell/run_train_vqvae_prior_mimic.sh \
    runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
```

Both wrappers `cd` to the repository root before running, so relative paths in the configs
and arguments resolve against the repo root. They use whichever `python` is on your `PATH`.

## Shell wrappers

=== "Stage 1"

    ```text
    scripts/shell/run_train_vqvae_mimic.sh [config_path]
    ```

    - `config_path` defaults to `configs/experiments/vqvae_mimic.yaml`.
    - The wrapper exits with an error if the config file does not exist.
    - It runs `python scripts/train/train_vqvae_mimic.py --config "<config_path>"`.

=== "Stage 2"

    ```text
    scripts/shell/run_train_vqvae_prior_mimic.sh <vqvae_checkpoint> [config_path]
    ```

    - `vqvae_checkpoint` is **required**. The wrapper exits if the file does not exist.
    - `config_path` defaults to `configs/experiments/prior_mimic.yaml`.
    - It runs `python scripts/train/train_prior_mimic.py --config "<config_path>" --vqvae-checkpoint "<vqvae_checkpoint>"`.

    The wrapper's usage examples use `runs/vqvae_mimic/seed_42/checkpoints/last.ckpt`. The
    checkpoint you pass overrides the path in the YAML.

The stage-2 wrapper is `run_train_vqvae_prior_mimic.sh`. `models/vqvae/README.md` and
`VQVAE_IMPLEMENTATION.md` refer to it as `run_train_prior_mimic.sh`, which does not exist.

## Python scripts and CLI flags

You can run both scripts directly from the repo root. You must pass either `--config` or
`--data-dir`, or the script exits.

!!! warning "YAML values override CLI flags"
    When `--config` is given, the script parses the CLI first and then **overwrites** the parsed
    arguments with any matching keys from the YAML. For example,
    `--config configs/experiments/vqvae_mimic.yaml --max-epochs 5` still trains for the YAML's
    `max_epochs: 100`. To change a value that the YAML sets, edit the YAML or a copy of it.
    CLI flags only take effect for settings the YAML does not contain. The one exception is
    stage 2's `--vqvae-checkpoint`, which overrides the YAML. The table at the end
    of this section lists which settings each script reads from the YAML.

### Shared flags (both scripts)

| Flag | Default | Notes |
|---|---|---|
| `--config` | `None` | Experiment YAML |
| `--exp-name` | `vqvae_mimic` / `prior_mimic` | Run directory name |
| `--seed` | `42` | Passed to `set_global_seed` and to the dataset split |
| `--runs-root` | `runs` | |
| `--data-dir` | `None` | MIMIC-IV-ECG root. Required without `--config` |
| `--batch-size` | `32` | |
| `--num-workers` | `4` | |
| `--max-samples` | `None` | Limit the dataset size for quick tests |
| `--skip-missing-check` | off | Flag |
| `--val-split` / `--test-split` | `0.1` / `0.1` | |
| `--seq-length` | `5000` | |
| `--lr`, `--b1`, `--b2` | stage 1: `1e-4`, `0.9`, `0.999`; stage 2: `1e-3`, `0.9`, `0.999` | Adam |
| `--num-embeddings` | `512` | Codebook size. Stage 2 must match stage 1 |
| `--max-epochs` | `100` | |
| `--accelerator` | `gpu` | |
| `--devices` | `0` | One or more ints, e.g. `--devices 0 1` |
| `--log-every-n-steps` | `50` | |
| `--check-val-every-n-epoch` | `1` | Both YAMLs set `5` |
| `--gradient-clip` | `1.0` | YAML key: `trainer.gradient_clip_val` |
| `--patience` | `10` | EarlyStopping patience. **CLI only** |
| `--save-top-k` | `3` | ModelCheckpoint `save_top_k`. **CLI only** |
| `--wandb` | off | Enables W&B. The YAML can also enable it |
| `--wandb-project` | `ecg-vqvae` / `ecg-vqvae-prior` | |
| `--wandb-entity` | `None` | |
| `--wandb-run-name` | `None` | Defaults to `<exp_name>_seed<seed>` |
| `--wandb-tags` | `vqvae ecg` / `vqvae prior pixelcnn` | |
| `--resume` | `None` | Lightning checkpoint to resume from (`trainer.fit(ckpt_path=...)`) |

### Stage-1-only flags (`train_vqvae_mimic.py`)

| Flag | Default |
|---|---|
| `--in-channels` | `12` (also used as the dataset's `num_leads`) |
| `--base-channels` | `64` |
| `--latent-channels` | `64` |
| `--num-res-blocks` | `2` |
| `--commitment-cost` | `0.25` |
| `--viz-every-n-epochs` | `5` |
| `--viz-num-samples` | `4` |
| `--viz-plot-all-leads` | off (overlay) |
| `--viz-log-to-tensorboard` | on. This is a `store_true` flag with default `True`, so the CLI cannot turn it off. Set `visualization.log_to_tensorboard: false` in the YAML instead |
| `--viz-log-to-wandb` | off |

There is no flag for `channel_multipliers`. The script hard-codes `(1, 2, 4, 4)` and ignores
`channel_multipliers` in the YAML.

### Stage-2-only flags (`train_prior_mimic.py`)

| Flag | Default |
|---|---|
| `--vqvae-checkpoint` | `None`. Overrides `model.params.config.vqvae_checkpoint` from the YAML. One of the two must be set, or the script exits with an error |
| `--hidden-dim` | `128` |
| `--num-layers` | `3` |

Stage 2 fixes `num_leads=12` for the dataset and does not add a visualization callback.

### YAML keys each script reads

| YAML key | Stage 1 | Stage 2 |
|---|---|---|
| `experiment.{name, seed, runs_root}` | yes | yes |
| `model.params.config.*` | `in_channels`, `base_channels`, `latent_channels`, `num_res_blocks`, `num_embeddings`, `commitment_cost`, `lr`, `b1`, `b2` | `num_embeddings`, `hidden_dim`, `num_layers`, `lr`, `b1`, `b2`, `vqvae_checkpoint` |
| `data.params.*` | `mimic_path`, `batch_size`, `num_workers`, `max_samples`, `skip_missing_check`, `val_split`, `test_split`, `seq_length` | same |
| `trainer.*` | `max_epochs`, `accelerator`, `devices`, `log_every_n_steps`, `check_val_every_n_epoch`, `gradient_clip_val` | same |
| `wandb.*` | `enabled`, `project`, `entity`, `run_name`, `tags` | same |
| `visualization.*` | `every_n_epochs`, `num_samples`, `plot_all_leads`, `log_to_tensorboard`, `log_to_wandb` | not read |

**Ignored keys:** `model.target`, `data.target`, `model.params.config.channel_multipliers`,
`data.params.num_leads`, the whole `callbacks:` list, `trainer.deterministic` (always `True`),
`wandb.notes`, `wandb.log_model`, and `wandb.log_freq`. The checkpoint, early-stopping and LR
monitor callbacks are built in code. The `callbacks:` section in the YAML has no effect, and that
includes its `dirpath`, `patience` and `save_top_k` values.

## Configuration

### `configs/experiments/vqvae_mimic.yaml` (stage 1)

| Section | Notable values |
|---|---|
| `experiment` | `name: vqvae_mimic`, `seed: 42`, `runs_root: runs` |
| `wandb` | **`enabled: true`**, `project: ecg-vqvae` |
| `model.params.config` | Same as the `VQVAEConfig` defaults: 64/64 channels, `[1, 2, 4, 4]`, 2 res blocks, 512 codes, β = 0.25, lr 1e-4 |
| `data.params` | `mimic_path` (hard-coded cluster path), `batch_size: 32`, `num_workers: 4`, **`max_samples: 1000`**, `seq_length: 5000` |
| `trainer` | `max_epochs: 100`, `devices: [0]`, `check_val_every_n_epoch: 5`, `gradient_clip_val: 1.0` |
| `visualization` | `every_n_epochs: 1`, `num_samples: 4`, `plot_all_leads: true`, `log_to_tensorboard: false`, `log_to_wandb: true` |

!!! note
    The checked-in stage-1 config has `max_samples: 1000`, which is a smoke-test setting, even
    though the comment beside it says "Use full dataset". For a full run, set it to `null`.

### `configs/experiments/prior_mimic.yaml` (stage 2)

| Section | Notable values |
|---|---|
| `experiment` | `name: prior_mimic`, `seed: 42` |
| `wandb` | `enabled: false`, `project: ecg-vqvae-prior` |
| `model.params.config` | `num_embeddings: 512`, `hidden_dim: 128`, `num_layers: 3`, `lr: 0.001`, `vqvae_checkpoint: "runs/vqvae_mimic/seed_42/checkpoints/last.ckpt"` (used only when `--vqvae-checkpoint` is not given) |
| `data.params` | Same dataset path. `max_samples: 10000` |
| `trainer` | Same as stage 1 |

Keep `data.params` (path, splits, `seed`) the same in both stages so the prior trains on codes
from the same train split.

### `configs/model/vqvae/`

`vqvae_1d_base.yaml` and `prior_base.yaml` are reference fragments that list the
`VQVAEConfig` / `PriorConfig` defaults under a top-level `model:` key. **The training scripts do
not load them.** To change hyperparameters, edit the `model.params.config` block of the
experiment YAML. See [Configuration](../getting_started/configuration.md) for the general
config layout.

## Choosing the VQ-VAE checkpoint

`train_prior_mimic.py` resolves the stage-1 checkpoint path in this order:

1. `--vqvae-checkpoint`, if given (the shell wrapper always passes it).
2. Otherwise `model.params.config.vqvae_checkpoint` from the YAML. `prior_mimic.yaml` defaults
   to `runs/vqvae_mimic/seed_42/checkpoints/last.ckpt`.

The path is validated after the config is resolved. If neither source sets a path, the script
exits with `Error: no VQ-VAE checkpoint given`; if the file does not exist, it exits with
`Error: VQ-VAE checkpoint not found: <path>`. So with the default YAML you can run
`python scripts/train/train_prior_mimic.py --config configs/experiments/prior_mimic.yaml`
without the flag once stage 1 has written `last.ckpt`.

Stage 1 never writes a `best.ckpt` (see below), so point at `last.ckpt` or one of the
`epoch###-step######.ckpt` files.

### Checkpoint names

Both scripts configure `ModelCheckpoint` the same way:

```python
ModelCheckpoint(
    dirpath=runs/<exp_name>/seed_<seed>/checkpoints,
    filename="epoch{epoch:03d}-step{step:06d}",
    save_last=True, save_top_k=<--save-top-k>, monitor="val_loss", mode="min",
    auto_insert_metric_name=False,
)
```

This produces files such as `epoch049-step001600.ckpt` and `last.ckpt`. If you rerun into the
same directory, Lightning keeps the existing files and adds a version suffix to the new ones,
e.g. `last-v1.ckpt` or `epoch014-step000480-v2.ckpt`. In that case `last.ckpt` may come from an
older run, so check timestamps. You can also change `experiment.seed`/`name`, or clear the
directory. At the end of training, the script prints the best checkpoint path
(`checkpoint_callback.best_model_path`) and a suggested stage-2 command.

## Outputs

Each run writes to `<runs_root>/<exp_name>/seed_<seed>/`:

```text
runs/vqvae_mimic/seed_42/
├── checkpoints/        # epoch###-step######.ckpt (top-k by val_loss) + last.ckpt
├── samples/            # stage 1: epoch_XXXX.png real-vs-reconstruction plots
├── tb/version_N/       # TensorBoard logs (always on)
└── wandb/              # W&B run files (if enabled)
```

The stage-2 run directory (`runs/prior_mimic/seed_42/`) has the same layout. It also creates
`samples/`, but nothing writes to it.

!!! note
    Unlike what the project-level docs describe, these two scripts do **not** write
    `config_resolved.yaml` or `metadata.json`.

## Logging

**TensorBoard** is always enabled (`TensorBoardLogger(save_dir=run_dir, name="tb")`):

```bash
tensorboard --logdir runs/vqvae_mimic/seed_42/tb
```

**Weights & Biases** is enabled by `--wandb` or by `wandb.enabled: true` in the YAML. Stage 1's
YAML enables it by default. Stage 2's does not. Note the following:

- If the YAML sets `enabled: true`, you cannot turn W&B off from the CLI. Edit the YAML, or run
  with `WANDB_MODE=disabled` / `WANDB_MODE=offline`.
- `WandbLogger` is created with `log_model=True`, whatever `wandb.log_model` says in the YAML,
  so checkpoints are uploaded as W&B artifacts.
- If `wandb` is not installed or fails to initialise, the script prints a warning and continues
  with TensorBoard only.
- The W&B run config records the main model and data hyperparameters. For stage 2 this
  includes `vqvae_checkpoint`.

Metrics logged by each stage are listed in
[VQ-VAE model → Lightning wrappers](../models/vqvae.md#lightning-wrappers). Both stages also log
the learning rate (`LearningRateMonitor`, per step) and monitor `val_loss`.

### Reconstruction plots (stage 1)

Stage 1 adds `VAEVisualizationCallback` (`training/callbacks/visualization.py`). At the end of a
validation epoch where `current_epoch % every_n_epochs == 0`, it reconstructs the first
`num_samples` ECGs of a validation batch and saves `samples/epoch_XXXX.png`. Depending on the
settings, it also logs the figure as `vae_reconstruction` to TensorBoard and/or W&B. It only runs
on epochs that have validation, so with `check_val_every_n_epoch: 5` you get plots every 5 epochs
at most. See [Callbacks](callbacks.md) and the [Visualization guide](../visualization/guide.md).

## Practical tips

- **Smoke test:** set `data.params.max_samples` to a small number such as `1000` and lower
  `trainer.max_epochs` **in the YAML**. CLI flags will not override them.
- **Early stopping counts validation runs, not epochs.** With `patience=10` and
  `check_val_every_n_epoch: 5`, training stops only after 50 epochs without improvement.
- **Match the codebook:** stage 2's `num_embeddings` must equal stage 1's. Otherwise the
  embedding lookup goes out of range or the logits have the wrong size.
- **Watch `train/codebook_usage` / `val/codebook_usage`** in stage 1. Low usage means codebook
  collapse.
- **Resuming:** `--resume path/to/last.ckpt` resumes the optimizer and loop state through
  Lightning. This is different from `--vqvae-checkpoint`.
- **GPU required by default:** `trainer.accelerator` is `"gpu"` in both YAMLs. To run on CPU,
  set it to `cpu` in the YAML.

## Related

- [VQ-VAE + Prior model](../models/vqvae.md)
- [Training Guide](guide.md)
- [Configuration](../getting_started/configuration.md)
- [Standalone VQ-VAE](../standalone/vqvae_standalone.md): the self-contained `single_file/`
  implementation, which is independent of the scripts on this page
