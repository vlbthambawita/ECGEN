# VQ-VAE Training

The [VQ-VAE](../models/vqvae.md) is trained in two separate runs:

| Stage | What is trained | `model.name` | Shell wrapper | Experiment config |
|---|---|---|---|---|
| 1 | `VQVAELightning` (encoder, codebook, decoder) | `vqvae` | `scripts/shell/run_train_vqvae_mimic.sh` | `configs/experiments/vqvae_mimic.yaml` |
| 2 | `PriorLightning` (PixelCNN over code indices; VQ-VAE frozen) | `vqvae_prior` | `scripts/shell/run_train_vqvae_prior_mimic.sh` | `configs/experiments/prior_mimic.yaml` |

Both stages run through `ecgen train` (see [Training with `ecgen train`](runner.md)). Stage 2 needs
a stage-1 checkpoint. Both stages train on MIMIC-IV-ECG. See
[MIMIC-IV-ECG](../datasets/mimic.md) for the dataset.

`PriorLightning` loads the stage-1 model with a plain `VQVAELightning.load_from_checkpoint(path)`
and records its config in the prior's hyperparameters, so a finished prior checkpoint can be
restored on its own with `PriorLightning.load_from_checkpoint(path)`. See
[Loading checkpoints](../models/vqvae.md#loading-checkpoints).

!!! warning "Old stage-2 checkpoints must be retrained"
    The PixelCNN prior was rewritten as a stack of causal dilated `CausalGatedBlock`s (default
    receptive field 1024 codes). Prior checkpoints from the earlier mask-A/B architecture have
    different `state_dict` keys and do not load into the current `PriorLightning`, so rerun
    stage 2. Stage-1 VQ-## Quick start

```bash
# Stage 1: VQ-VAE
ecgen train configs/experiments/vqvae_mimic.yaml
#   or: ./scripts/shell/run_train_vqvae_mimic.sh [config] [overrides]

# Find a real checkpoint (there is no best.ckpt, see "Checkpoint names" below)
ls runs/vqvae_mimic/seed_42/checkpoints/

# Stage 2: PixelCNN prior on the frozen stage-1 model
ecgen train configs/experiments/prior_mimic.yaml \
    model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
#   or: ./scripts/shell/run_train_vqvae_prior_mimic.sh runs/vqvae_mimic/seed_42/checkpoints/last.ckpt [config] [overrides]
```

The wrappers `cd` to the repository root first, so relative paths resolve against the repo root,
and pass any extra `key=value` arguments to `ecgen train`. The stage-2 wrapper exits early if the
checkpoint file does not exist, and otherwise adds `model.params.vqvae_checkpoint=<checkpoint>`.

Command-line overrides are applied after the YAML, so they always win, e.g.
`ecgen train configs/experiments/vqvae_mimic.yaml trainer.max_epochs=5 data.dataset.params.max_samples=1000`.

## Configuration

### `configs/experiments/vqvae_mimic.yaml` (stage 1)

| Section | Notable values |
|---|---|
| `experiment` | `name: vqvae_mimic`, `seed: 42`, `runs_root: runs` |
| `wandb` | **`enabled: true`**, `project: ecg-vqvae` |
| `model` | `name: vqvae`; `params` = the `VQVAEConfig` defaults: 64/64 channels, `channel_multipliers: [1, 2, 4, 4]`, 2 res blocks, 512 codes, β = 0.25, lr 1e-4 |
| `data` | `dataset: mimic_iv_ecg` with `root` (hard-coded cluster path), **`max_samples: 1000`**, `leads: null` (12), `seq_length: 5000`, `scale: zscore`; `batch_size: 32`, `num_workers: 4` |
| `checkpoint` | `monitor: val_loss`, `save_top_k: 3` |
| `callbacks` | `early_stopping` (patience 10), `lr_monitor`, `vae_visualization` (`log_every_n_epochs: 1`, `num_samples: 4`, `log_to_wandb: true`) |
| `trainer` | `max_epochs: 100`, `devices: [0]`, `check_val_every_n_epoch: 5`, `gradient_clip_val: 1.0` |

!!! note
    The checked-in stage-1 config has `max_samples: 1000`, a smoke-test setting. For a full run,
    pass `data.dataset.params.max_samples=null`.

### `configs/experiments/prior_mimic.yaml` (stage 2)

| Section | Notable values |
|---|---|
| `experiment` | `name: prior_mimic`, `seed: 42` |
| `wandb` | `enabled: false`, `project: ecg-vqvae-prior` |
| `model` | `name: vqvae_prior`; `num_embeddings: 512`, `hidden_dim: 128`, `num_layers: 10`, `kernel_size: 2`, `dilation_cycle: 10`, `lr: 0.001`, `vqvae_checkpoint: runs/vqvae_mimic/seed_42/checkpoints/last.ckpt` |
| `data` | Same dataset path and `leads`/`seq_length`/`scale`; `max_samples: 10000` |
| `callbacks` | `early_stopping`, `lr_monitor` (no visualization) |
| `trainer` | Same as stage 1 |

Keep the dataset params (path, splits, `seed`, `leads`, `seq_length`, `scale`) the same in both
stages, so the prior trains on codes from the same train split and the input matches what the
VQ-VAE was trained on. The runner's [compatibility check](runner.md#compatibility-check) compares
the dataset with the frozen VQ-VAE's `in_channels`.

### `configs/model/vqvae/`

`vqvae_1d_base.yaml` and `prior_base.yaml` are reference fragments that list the
`VQVAEConfig` / `PriorConfig` defaults under a top-level `model:` key. **The runner does not load
them.** Change hyperparameters in `model.params` of the experiment YAML or with overrides such as
`model.params.num_embeddings=1024`.

## Choosing the VQ-VAE checkpoint

The stage-1 checkpoint is `model.params.vqvae_checkpoint`: the value in `prior_mimic.yaml`, unless
an override (or the stage-2 wrapper's argument) replaces it. `PriorLightning` raises
`FileNotFoundError: VQ-VAE checkpoint not found: <path>` if the file does not exist. With the
default YAML you can run `ecgen train configs/experiments/prior_mimic.yaml` without an override
once stage 1 has written `last.ckpt`.

Stage 1 never writes a `best.ckpt` (see below), so point at `last.ckpt` or one of the
`epoch###-step######.ckpt` files. The best path is recorded in
`runs/vqvae_mimic/seed_42/metrics/train_val_metrics.json` (`best_model_path`).

### Checkpoint names

The runner builds `ModelCheckpoint` from the `checkpoint:` section:

```python
ModelCheckpoint(
    dirpath=runs/<exp_name>/seed_<seed>/checkpoints,
    filename="epoch{epoch:03d}-step{step:06d}",
    save_last=True, save_top_k=3, monitor="val_loss", mode="min",
    auto_insert_metric_name=False,
)
```

This produces files such as `epoch049-step001600.ckpt` and `last.ckpt`. If you rerun into the
same directory, Lightning keeps the existing files and adds a version suffix to the new ones,
e.g. `last-v1.ckpt` or `epoch014-step000480-v2.ckpt`. In that case `last.ckpt` may come from an
older run, so check timestamps, change `experiment.seed`/`name`, or clear the directory.

## Outputs

Each run writes to `<runs_root>/<exp_name>/seed_<seed>/`:

```text
runs/vqvae_mimic/seed_42/
├── config_resolved.yaml
├── metadata.json
├── checkpoints/        # epoch###-step######.ckpt (top-k by val_loss) + last.ckpt
├── samples/            # stage 1: epoch_XXXX.png real-vs-reconstruction plots
├── metrics/train_val_metrics.json
├── tb/version_N/       # TensorBoard logs (always on)
└── wandb/              # W&B run files (if enabled)
```

The stage-2 run directory (`runs/prior_mimic/seed_42/`) has the same layout; its `samples/` stays
empty.

## Logging

**TensorBoard** is always enabled:

```bash
tensorboard --logdir runs/vqvae_mimic/seed_42/tb
```

**Weights & Biases** is enabled by `wandb.enabled: true` (stage 1's YAML) or the override
`wandb.enabled=true`; turn it off with `wandb.enabled=false`. `log_model` follows
`wandb.log_model` (default `false`). If `wandb` is not installed, the runner logs a warning and
continues with TensorBoard only. The W&B run config is the whole resolved experiment config.

Metrics logged by each stage are listed in
[VQ-VAE model → Lightning wrappers](../models/vqvae.md#lightning-wrappers). Both stages also log
the learning rate (`lr_monitor`, per step) and monitor `val_loss`.

### Reconstruction plots (stage 1)

Stage 1's `callbacks:` list includes `vae_visualization` (`VAEVisualizationCallback`). At the end of
a validation epoch where `current_epoch % log_every_n_epochs == 0`, it reconstructs the first
`num_samples` ECGs of a validation batch and saves `samples/epoch_XXXX.png`. Depending on the
settings, it also logs the figure as `vae_reconstruction` to TensorBoard and/or W&B. It only runs
on epochs that have validation, so with `check_val_every_n_epoch: 5` you get plots every 5 epochs
at most. See [Callbacks](callbacks.md) and the [Visualization guide](../visualization/guide.md).

de](../visualization/guide.md).

## Practical tips

- **Smoke test:** `data.dataset.params.max_samples=1000 trainer.max_epochs=2
  trainer.check_val_every_n_epoch=1`, or `data.dataset.name=synthetic` without any data (see
  [Smoke runs](runner.md#smoke-runs-without-data)).
- **Early stopping counts validation runs, not epochs.** With `patience: 10` and
  `check_val_every_n_epoch: 5`, training stops only after 50 epochs without improvement.
- **Match the codebook:** stage 2's `num_embeddings` must equal stage 1's. Otherwise the
  embedding lookup goes out of range or the logits have the wrong size.
- **Watch `train/codebook_usage` / `val/codebook_usage`** in stage 1. Low usage means codebook
  collapse.
- **Resuming:** `--resume path/to/last.ckpt` resumes the optimizer and loop state through
  Lightning. This is different from `model.params.vqvae_checkpoint`.
- **GPU required by default:** `trainer.accelerator` is `"gpu"` in both YAMLs. To run on CPU,
  pass `trainer.accelerator=cpu trainer.devices=1`.

## Related

- [VQ-VAE + Prior model](../models/vqvae.md)
- [Training Guide](guide.md)
- [Configuration](../getting_started/configuration.md)
- [Standalone VQ-VAE](../standalone/vqvae_standalone.md): the self-contained `single_file/`
  implementation, which is independent of the runner on this page
