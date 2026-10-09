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
  params:
    config:
      in_channels: 12
      base_channels: 64
      latent_channels: 8
      kl_weight: 0.0001

data:
  params:
    mimic_path: /path/to/mimic-iv-ecg
    batch_size: 32
    max_samples: 1000      # null for the full dataset

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: [0]
```

Only specific keys are read by each script; see [Configuration](../getting_started/configuration.md).

### 3. Train

```bash
./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
./scripts/shell/run_train_vqvae_mimic.sh                       # VQ-VAE stage 1
./scripts/shell/run_train_vqvae_prior_mimic.sh <vqvae.ckpt>    # VQ-VAE stage 2
```

See [Training scripts compared](scripts_comparison.md) for every entry point.

## Training components

### Callbacks — `training/callbacks/`

- `visualization.py`: `VAEVisualizationCallback`, `ECGVisualizationCallback`,
  `GeneratedSamplesCallback`
- `checkpointing.py`, `logging.py`: empty placeholder classes (`CustomCheckpointCallback`,
  `CustomLoggingCallback`)

Checkpointing, early stopping and LR monitoring use the stock Lightning callbacks. See
[Callbacks](callbacks.md).

### Losses — `training/losses/`

| File | Functions |
|------|-----------|
| `vae_losses.py` | `vae_loss` |
| `vqvae_losses.py` | `vqvae_loss`, `prior_loss` |
| `gan_losses.py` | `wgan_discriminator_loss`, `wgan_generator_loss`, `adversarial_loss`, `least_squares_discriminator_loss`, `least_squares_generator_loss` |
| `common.py` | `mse_loss`, `l1_loss` |

### Metrics — `training/metrics/`

| File | Functions |
|------|-----------|
| `reconstruction.py` | `reconstruction_mse`, `reconstruction_mae` |
| `generation.py` | `signal_to_noise_ratio` |
| `clinical.py` | `wasserstein_distance`, `gradient_penalty_loss` |

Evaluation metrics for generated samples live in the separate `evaluation/` package; see
[Evaluation metrics](../evaluation/metrics.md).

### Trainers — `training/trainers/`

- `gan_trainer.py`: a generic config-driven trainer (`--config`) that instantiates `target` classes
  from the YAML and writes `config_resolved.yaml` / `metadata.json`. It imports the legacy
  `ecgen.utils` modules.
- `base_trainer.py`, `vae_trainer.py`: placeholder classes.

## Monitoring

### TensorBoard

Every script logs to `runs/<exp_name>/seed_<seed>/tb/`:

```bash
tensorboard --logdir runs/                          # all runs
tensorboard --logdir runs/vae_mimic/seed_42/tb      # one run
```

### Weights & Biases

Enable it in the experiment YAML (the scripts create the `WandbLogger` for you):

```yaml
wandb:
  enabled: true
  project: ecg-vae
  entity: null
  run_name: null   # defaults to <exp_name>_seed<seed>
  tags: [vae]
```

or with `--wandb --wandb-project <name>` when running without a config.

## Reproducibility

The scripts call `utils.seed.set_global_seed(seed)` and build the `Trainer` with
`deterministic=True`. The dataset split is seeded by the same `seed` and is done at the subject level.

## Practical advice

1. **Start small**: set `data.params.max_samples` (e.g. 1000) and a few epochs to check the pipeline.
2. **Watch the plots**: the reconstruction images in `samples/` reveal problems before the loss does.
3. **Mind `check_val_every_n_epoch`**: checkpointing, early stopping and visualization all run
   after validation. Validation runs on 0-based epochs 4, 9, 14, ... when it is set to 5; with
   `visualization.every_n_epochs: 5` as well (the `vae_mimic.yaml` defaults) no reconstruction plot
   is ever produced — see [Visualization](../visualization/guide.md#reconstruction-plots-during-training).

## Troubleshooting

### Out of memory
- Reduce `batch_size`
- Reduce model width/depth (`base_channels`, `num_res_blocks`)

### Slow training
- Increase `num_workers`
- Use `skip_missing_check: true` once the dataset is known to be complete
