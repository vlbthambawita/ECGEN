# Quick Start

This page gets a VAE training on MIMIC-IV-ECG with the config-driven workflow. For the other model
families see [VQ-VAE training](../training/vqvae_training.md) and
[Pulse2Pulse training](../training/pulse2pulse_training.md).

## 1. Install

```bash
pip install -e .
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # train_vae_mimic.py still imports the legacy ecgen package
```

See [Installation](installation.md) for details.

## 2. Point the config at your data

Edit [`configs/experiments/vae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vae_mimic.yaml):

```yaml
data:
  params:
    mimic_path: /path/to/mimic-iv-ecg   # directory containing machine_measurements.csv and files/
    max_samples: 1000                   # small value for a smoke run; null for the full dataset
```

For a short run also lower `trainer.max_epochs`.

## 3. Train

```bash
./scripts/shell/run_train_vae_mimic_config.sh                        # uses configs/experiments/vae_mimic.yaml
./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
# or directly
python scripts/train/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml
```

The shell wrapper `cd`s to the repo root and calls `scripts/train/train_vae_mimic.py --config <yaml>`.

!!! info "W&B is on by default"
    `vae_mimic.yaml` ships with `wandb.enabled: true` and `visualization.log_to_wandb: true`.
    Set `wandb.enabled: false` if you do not have `wandb` installed/configured.

## 4. Monitor

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
ls runs/vae_mimic/seed_42/samples/      # epoch_NNNN.png reconstruction plots
```

!!! tip
    With the shipped `vae_mimic.yaml` (`trainer.check_val_every_n_epoch: 5`,
    `visualization.every_n_epochs: 5`) the plot and validation epochs never coincide, so no images
    are written. Set `visualization.every_n_epochs: 1` to get a plot after every validation run.
    See [Visualization](../visualization/guide.md).

Outputs of `train_vae_mimic.py`:

```
runs/<experiment.name>/seed_<seed>/
├── checkpoints/   # epochNNN-stepNNNNNN.ckpt (top-3 by val_loss) + last.ckpt
├── samples/       # reconstruction plots from VAEVisualizationCallback
└── tb/            # TensorBoard logs
```

## 5. Use the trained model

```python
import torch
from models.vae import VAELightning, VAEConfig

ckpt = torch.load("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
model = VAELightning(VAEConfig(**ckpt["hyper_parameters"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

samples = model.sample(n_samples=16, seq_length=5000)   # (16, 12, 5000)
```

!!! warning
    `VAELightning.load_from_checkpoint(path)` without arguments fails, because the module takes a
    single `config` argument while the checkpoint stores the flattened config fields as
    hyper-parameters. Either rebuild the config as above or pass it explicitly:
    `VAELightning.load_from_checkpoint(path, config=VAEConfig())`.

## Next steps

- [Configuration](configuration.md) — which YAML keys the scripts actually read
- [VAE model](../models/vae.md) and [architecture overview](../models/architecture.md)
- [VAE training guide](../training/vae_training.md)
- [Visualization](../visualization/guide.md)
