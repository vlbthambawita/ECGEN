# Training Guide

General workflow for training models in ECGEN. Model-specific pages:
[VAE](vae_training.md), [VQ-VAE](vqvae_training.md), [Pulse2Pulse](pulse2pulse_training.md).

## Workflow

### 1. Get the data

Training reads raw MIMIC-IV-ECG directly (WFDB records + `machine_measurements.csv`); there is no
preprocessing step. See [MIMIC-IV-ECG](../datasets/mimic.md).

!!! note
    `scripts/data/prepare_mimic.py` and `scripts/data/validate_dataset.py` are placeholders that only
    print a message.

### 2. Configure an experiment

Copy and edit an experiment file in `configs/experiments/`:

```yaml
# configs/experiments/my_vae.yaml (excerpt)
experiment:
  name: my_vae
  seed: 42
  runs_root: runs

model:
  name: vae
  params:
    in_channels: 12
    base_channels: 64
    latent_channels: 8
    kl_weight: 0.0001

data:
  dataset:
    name: mimic_iv_ecg
    params:
      root: /path/to/mimic-iv-ecg
      max_samples: 1000      # null for the full dataset
  batch_size: 32

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: [0]
```

Every section is read by the same runner; see [Training with `ecgen train`](runner.md) for the full
schema and [Configuration](../getting_started/configuration.md) for the per-family `params`.

### 3. Train

```bash
ecgen train configs/experiments/my_vae.yaml
ecgen train configs/experiments/my_vae.yaml trainer.max_epochs=5     # overrides win over the YAML
ecgen train configs/experiments/vqvae_mimic.yaml                     # VQ-VAE stage 1
ecgen train configs/experiments/prior_mimic.yaml model.params.vqvae_checkpoint=<vqvae.ckpt>   # stage 2
```

The `scripts/shell/run_train_*.sh` wrappers call the same command; see
[Training entry points](scripts_comparison.md).

## Training components

### Callbacks — `ecgen.training.callbacks`

- `visualization.py`: `VAEVisualizationCallback`, `ECGVisualizationCallback`,
  `GeneratedSamplesCallback`
- `checkpointing.py`, `logging.py`: empty placeholder classes (`CustomCheckpointCallback`,
  `CustomLoggingCallback`)

The runner always adds a `ModelCheckpoint` built from the `checkpoint:` section; the other callbacks,
including Lightning's `early_stopping` and `lr_monitor`, are listed by name in `callbacks:`. See
[Callbacks](callbacks.md).

### Losses

Each family keeps its loss next to its model:

| Module | Functions |
|------|-----------|
| `ecgen.families.vae.losses` | `vae_loss` |
| `ecgen.families.vqvae.losses` | `vqvae_loss`, `prior_loss` |
| `ecgen.families.gan.losses` | `wgan_discriminator_loss`, `wgan_generator_loss`, `adversarial_loss`, `least_squares_discriminator_loss`, `least_squares_generator_loss` |
| `training/losses/common.py` | `mse_loss`, `l1_loss` |

The old `training/losses/{vae,vqvae,gan}_losses.py` modules re-export these for one release.

### Metrics — `training/metrics/`

| File | Functions |
|------|-----------|
| `reconstruction.py` | `reconstruction_mse`, `reconstruction_mae` |
| `generation.py` | `signal_to_noise_ratio` |
| `clinical.py` | `wasserstein_distance`, `gradient_penalty_loss` |

Evaluation metrics for generated samples live in the separate `evaluation/` package; see
[Evaluation metrics](../evaluation/metrics.md).

### Trainers — `training/trainers/`

- `gan_trainer.py`: deprecated; forwards `--config <yaml>` to the runner (`ecgen.training.train`).
- `base_trainer.py`, `vae_trainer.py`: placeholder classes.

## Monitoring

### TensorBoard

Every run logs to `runs/<exp_name>/seed_<seed>/tb/`:

```bash
tensorboard --logdir runs/                          # all runs
tensorboard --logdir runs/vae_mimic/seed_42/tb      # one run
```

### Weights & Biases

Enable it in the experiment YAML (the runner creates the `WandbLogger` for you):

```yaml
wandb:
  enabled: true
  project: ecg-vae
  entity: null
  run_name: null   # defaults to <exp_name>_seed<seed>
  tags: [vae]
```

or on the command line with `wandb.enabled=true wandb.project=<name>`.

## Reproducibility

The runner calls `ecgen.utils.seed.set_global_seed(experiment.seed)`; the shipped configs also set
`trainer.deterministic: true`. The MIMIC-IV-ECG split is done at the subject level and seeded by the
dataset's own `seed` parameter (default 42). Each run writes `config_resolved.yaml` and
`metadata.json` (command, package versions, git info) next to its checkpoints.

## Practical advice

1. **Start small**: set `data.dataset.params.max_samples=1000` and a few epochs to check the
   pipeline, or use `data.dataset.name=synthetic` with no data at all
   (see [Smoke runs](runner.md#smoke-runs-without-data)).
2. **Watch the plots**: the reconstruction images in `samples/` reveal problems before the loss does.
3. **Mind `check_val_every_n_epoch`**: checkpointing, early stopping and visualization all run
   after validation. Validation runs on 0-based epochs 4, 9, 14, ... when it is set to 5; with
   `log_every_n_epochs: 5` on `vae_visualization` as well (the `vae_mimic.yaml` defaults) no
   reconstruction plot is ever produced — see
   [Visualization](../visualization/guide.md#reconstruction-plots-during-training).

## Troubleshooting

### Out of memory
- Reduce `batch_size`
- Reduce model width/depth (`base_channels`, `num_res_blocks`)

### Slow training
- Increase `num_workers`
- Use `data.dataset.params.skip_missing_check: true` once the dataset is known to be complete
