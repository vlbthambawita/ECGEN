# Configuration

Training is driven by YAML experiment files in
[`configs/experiments/`](https://github.com/vlbthambawita/ECGEN/tree/main/configs/experiments),
all run by the same command, `ecgen train <config.yaml> [key=value ...]`.

| File | `model.name` | Model |
|------|-------------|-------|
| `vae_mimic.yaml` | `vae` | VAE |
| `vqvae_mimic.yaml` | `vqvae` | VQ-VAE (stage 1) |
| `prior_mimic.yaml` | `vqvae_prior` | PixelCNN prior (stage 2) |
| `pulse2pulse_mimic.yaml` | `pulse2pulse` | Pulse2Pulse GAN |

[Training with `ecgen train`](../training/runner.md) has the full schema, an annotated example, the
registry names and the run-directory layout. This page summarises how a config is applied.

## How configs are applied

1. The YAML is loaded, then each command-line override `dotted.key=value` is applied on top (values
   parsed as YAML). **Overrides always win.**
2. Unknown top-level sections are rejected. The allowed sections are `experiment`, `model`, `data`,
   `checkpoint`, `callbacks`, `trainer`, `wandb` and `resume`; `model` and `data` are required.
3. Defaults are filled in and `${run_dir}`, `${name}` and `${seed}` are substituted in strings.
4. The model is built from `model: {name | target, params}`, the dataset from
   `data.dataset: {name | target, params}`, and each `callbacks:` entry the same way. `params` are
   passed to the constructor as keyword arguments, so every constructor argument is configurable
   without editing code.
5. The resolved config is written to `runs/<name>/seed_<seed>/config_resolved.yaml`.

```bash
ecgen train configs/experiments/vae_mimic.yaml --print-config        # inspect the result
ecgen train configs/experiments/vae_mimic.yaml model.params.kl_weight=1e-3 'trainer.devices=[1]'
```

## Sections

```yaml
experiment:     # name, seed, runs_root
model:          # name | target, params (flat config-dataclass fields)
data:           # dataset: {name | target, params}, batch_size, num_workers, pin_memory, split_params
checkpoint:     # ModelCheckpoint: monitor, mode, save_top_k, save_last, filename
callbacks:      # list of {name | target, params}
trainer:        # pl.Trainer keyword arguments
wandb:          # optional Weights & Biases logging
resume:         # checkpoint to resume from
```

### `model.params` per family

`model.params` holds the fields of the family's config dataclass, flat. The legacy nested form
`params: {config: {...}}` is still accepted.

| `model.name` | Config dataclass | Fields |
|---|---|---|
| `vae` | `VAEConfig` | `in_channels`, `base_channels`, `latent_channels`, `channel_multipliers`, `num_res_blocks`, `lr`, `kl_weight`, `b1`, `b2` |
| `vqvae` | `VQVAEConfig` | `in_channels`, `base_channels`, `latent_channels`, `channel_multipliers`, `num_res_blocks`, `num_embeddings`, `commitment_cost`, `lr`, `b1`, `b2` |
| `vqvae_prior` | `PriorConfig` | `num_embeddings`, `hidden_dim`, `num_layers`, `kernel_size`, `dilation_cycle`, `lr`, `b1`, `b2`, `vqvae_checkpoint` |
| `pulse2pulse` | `Pulse2PulseConfig` | `model_size`, `num_channels`, `seq_length`, `lr`, `b1`, `b2`, `lmbda`, `n_critic` |

The model's channel count must equal the number of leads the dataset yields; the runner checks this
before training.

### `data.dataset.params` for MIMIC-IV-ECG

`root`, `val_split`, `test_split`, `seed`, `max_samples`, `skip_missing_check`, `leads`,
`num_leads`, `seq_length`, `scale`, `measurements`, `missing`, `feature_normalizer`. See
[MIMIC-IV-ECG](../datasets/mimic.md#dataset-parameters). `split` is filled in by the datamodule;
`data.split_params` overrides params for one split, e.g. `split_params: {val: {max_samples: 1000}}`.

### `callbacks`

Built in order after the runner's own `ModelCheckpoint`. Registered names: `vae_visualization`,
`ecg_visualization`, `generated_samples`, `early_stopping`, `lr_monitor`, `model_checkpoint`. A
callback that takes `save_dir` defaults to `${run_dir}/samples`. See
[Callbacks](../training/callbacks.md).

!!! note "Removed: the `visualization:` section"
    Plot settings used to live in a top-level `visualization:` section read by the old VAE/VQ-VAE
    scripts. They are now the `params` of a `vae_visualization` callback, e.g.
    `{name: vae_visualization, params: {log_every_n_epochs: 1, num_samples: 4}}`. A config that
    still has `visualization:` fails with `Unknown config sections`.

## Reusable fragments

`configs/model/`, `configs/dataset/` and `configs/training/` hold YAML fragments
(`model/vae/vae_1d_base.yaml`, `model/vae/vae_1d_large.yaml`, `model/vqvae/vqvae_1d_base.yaml`,
`model/vqvae/prior_base.yaml`, `model/gan/pulse2pulse.yaml`, `dataset/mimic.yaml`,
`dataset/ptbxl.yaml`, `training/fast_dev.yaml`, `training/production.yaml`, ...). **The runner does
not load these files**; they are reference values only. To use them, copy the values into an
experiment file.

## Creating a custom experiment

```bash
cp configs/experiments/vae_mimic.yaml configs/experiments/my_vae.yaml
# edit experiment.name, data.dataset.params.root, model.params.*, trainer.max_epochs, ...
ecgen train configs/experiments/my_vae.yaml
```

Change `experiment.name` so the run does not overwrite `runs/vae_mimic/seed_42/`. For one-off
changes, prefer overrides to a new file:

```bash
ecgen train configs/experiments/vae_mimic.yaml experiment.name=vae_kl1e-3 model.params.kl_weight=1e-3
```
