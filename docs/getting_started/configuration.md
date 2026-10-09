# Configuration

Training is driven by YAML experiment files in
[`configs/experiments/`](https://github.com/vlbthambawita/ECGEN/tree/main/configs/experiments).

| File | Consumed by | Model |
|------|-------------|-------|
| `vae_mimic.yaml` | `scripts/train/train_vae_mimic.py` | VAE |
| `vqvae_mimic.yaml` | `scripts/train/train_vqvae_mimic.py` | VQ-VAE (stage 1) |
| `prior_mimic.yaml` | `scripts/train/train_prior_mimic.py` | PixelCNN prior (stage 2) |
| `pulse2pulse_mimic.yaml` | `training/trainers/gan_trainer.py` (generic `target`-based trainer) | Pulse2Pulse GAN |

## How configs are applied

!!! important "The `target:` fields are not used by the VAE / VQ-VAE / prior scripts"
    `train_vae_mimic.py`, `train_vqvae_mimic.py` and `train_prior_mimic.py` do **not** instantiate
    objects from `model.target`, `data.target` or `callbacks[].target`. Each script reads a fixed set
    of keys (listed below) and copies them onto its argparse arguments. Keys not listed are ignored,
    and a new key needs matching handling in the script.

    Only `training/trainers/gan_trainer.py` builds the model, datamodule and callbacks generically
    from `target` + `params`.

!!! warning "Config values take precedence over CLI flags"
    The scripts parse the command line first and then overwrite each argument with the YAML value
    whenever that key is present. A CLI flag therefore only takes effect for keys that are **absent**
    from the YAML file (or for flags that are never read from YAML, such as `--patience`,
    `--save-top-k` and `--resume`).

## Sections

An experiment file has these top-level sections:

```yaml
experiment:     # name, seed, runs_root
wandb:          # optional Weights & Biases logging
model:          # target + params.config
data:           # target + params
callbacks:      # list of target + params (only used by gan_trainer.py)
trainer:        # Lightning Trainer settings
visualization:  # VAEVisualizationCallback settings (VAE and VQ-VAE scripts)
```

### Keys read by `train_vae_mimic.py`

Annotated against [`configs/experiments/vae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vae_mimic.yaml):

```yaml
experiment:
  name: vae_mimic          # -> --exp-name
  seed: 42                 # -> --seed
  runs_root: runs          # -> --runs-root

wandb:
  enabled: true            # turns on WandbLogger (--wandb)
  project: ecg-vae         # -> --wandb-project
  entity: null             # -> --wandb-entity
  run_name: null           # -> --wandb-run-name (default "<exp_name>_seed<seed>")
  tags: [vae, ecg-generation, mimic-iv-ecg]   # -> --wandb-tags
  # notes, log_model, log_freq: present in the file but NOT read by this script
  # (the script always uses log_model=True)

model:
  target: ecgen.models.vae.VAELightning   # not used
  params:
    config:
      in_channels: 12      # also used as the dataset's num_leads
      base_channels: 64
      latent_channels: 8
      channel_multipliers: [1, 2, 4, 4]   # NOT read: hard-coded to (1, 2, 4, 4)
      num_res_blocks: 2
      lr: 0.0001
      kl_weight: 0.0001
      b1: 0.9
      b2: 0.999

data:
  target: ecgen.data.mimic_dataset.MIMICIVECGDataset   # not used
  params:
    mimic_path: /path/to/mimic-iv-ecg   # -> --data-dir
    batch_size: 32
    num_workers: 4
    max_samples: 10000     # per split; null = all
    skip_missing_check: false
    val_split: 0.1
    test_split: 0.1
    num_leads: 12          # NOT read (num_leads comes from model in_channels)
    seq_length: 5000

callbacks: [...]           # NOT read; checkpoint/early-stopping/LR-monitor are built in code

trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: [0]
  log_every_n_steps: 50
  check_val_every_n_epoch: 5
  gradient_clip_val: 1.0   # -> --gradient-clip
  deterministic: true      # NOT read; the script always passes deterministic=True

visualization:
  every_n_epochs: 5        # -> --viz-every-n-epochs (with check_val_every_n_epoch: 5 this never fires; use 1)
  num_samples: 4           # -> --viz-num-samples
  plot_all_leads: true     # -> --viz-plot-all-leads
  log_to_tensorboard: false
  log_to_wandb: true
```

Settings that come only from CLI flags (defaults in parentheses): `--patience` (10, early stopping on
`val_loss`), `--save-top-k` (3), `--resume` (none).

### VQ-VAE and prior scripts

`train_vqvae_mimic.py` reads the same `experiment`, `wandb`, `data.params`, `trainer` and
`visualization` keys, and from `model.params.config`: `in_channels`, `base_channels`,
`latent_channels`, `num_res_blocks`, `num_embeddings`, `commitment_cost`, `lr`, `b1`, `b2`.

`train_prior_mimic.py` reads `experiment`, `wandb`, `data.params` and `trainer` (no
`visualization` section), and from `model.params.config`: `num_embeddings`, `hidden_dim`,
`num_layers`, `lr`, `b1`, `b2` and `vqvae_checkpoint`. A `vqvae_checkpoint` in the YAML
**overrides** the required `--vqvae-checkpoint` flag.

See [VQ-VAE training](../training/vqvae_training.md) for details.

### Pulse2Pulse (`gan_trainer.py`)

`pulse2pulse_mimic.yaml` is written for the generic trainer in
[`training/trainers/gan_trainer.py`](https://github.com/vlbthambawita/ECGEN/blob/main/training/trainers/gan_trainer.py),
which imports `model.target`, `data.target` and every `callbacks[].target` and calls them with their
`params`, passes `trainer` to `pl.Trainer`, and writes `config_resolved.yaml` and `metadata.json` to
the run directory. Its targets point at the legacy `ecgen.*` package. See
[Pulse2Pulse training](../training/pulse2pulse_training.md).

## Reusable fragments

`configs/model/`, `configs/dataset/` and `configs/training/` hold YAML fragments
(`model/vae/vae_1d_base.yaml`, `model/vae/vae_1d_large.yaml`, `model/vqvae/vqvae_1d_base.yaml`,
`model/vqvae/prior_base.yaml`, `model/gan/pulse2pulse.yaml`, `dataset/mimic.yaml`,
`dataset/ptbxl.yaml`, `training/fast_dev.yaml`, `training/production.yaml`, ...). **No script loads
these files**; they are reference values only. To use them, copy the values into an experiment file.

## Creating a custom experiment

```bash
cp configs/experiments/vae_mimic.yaml configs/experiments/my_vae.yaml
# edit experiment.name, data.params.mimic_path, model.params.config.*, trainer.max_epochs, ...
./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
```

Change `experiment.name` so the run does not overwrite `runs/vae_mimic/seed_42/`.

## Without a config file

All three scripts also run from flags alone:

```bash
python scripts/train/train_vae_mimic.py \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name my_vae \
    --batch-size 32 \
    --max-epochs 100 \
    --max-samples 1000
```

Either `--config` or `--data-dir` must be given. Run `python scripts/train/train_vae_mimic.py --help`
for the full flag list; see [VAE training](../training/vae_training.md#command-line-flags).
