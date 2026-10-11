# Quick Start

This page gets a VAE training on MIMIC-IV-ECG with the config-driven workflow. For the other model
families see [VQ-VAE training](../training/vqvae_training.md) and
[Pulse2Pulse training](../training/pulse2pulse_training.md).

## 1. Install

```bash
pip install -e .
```

See [Installation](installation.md) for details.

## 2. Point the config at your data

Edit [`configs/experiments/vae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vae_mimic.yaml)
or pass overrides on the command line:

```yaml
data:
  dataset:
    name: mimic_iv_ecg
    params:
      root: /path/to/mimic-iv-ecg   # directory containing machine_measurements.csv and files/
      max_samples: 1000             # small value for a smoke run; null for the full dataset
```

For a short run also lower `trainer.max_epochs`.

!!! tip "No data yet?"
    Add `data.dataset.name=synthetic 'data.dataset.params={num_samples: 64}'` to the command below
    to train on generated ECG-like signals. See
    [Smoke runs](../training/runner.md#smoke-runs-without-data).

## 3. Train

```bash
ecgen train configs/experiments/vae_mimic.yaml
ecgen train configs/experiments/vae_mimic.yaml \
    data.dataset.params.root=/path/to/mimic-iv-ecg \
    data.dataset.params.max_samples=1000 trainer.max_epochs=10
# or the shell wrapper (same command, run from the repo root)
./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/vae_mimic.yaml trainer.max_epochs=10
```

Command-line overrides are applied after the YAML, so they always win.

!!! info "W&B is on by default"
    `vae_mimic.yaml` ships with `wandb.enabled: true` and `log_to_wandb: true` on its
    `vae_visualization` callback. Pass `wandb.enabled=false` if you do not have `wandb`
    installed/configured.

## 4. Monitor

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
ls runs/vae_mimic/seed_42/samples/      # epoch_NNNN.png reconstruction plots
```

!!! tip
    With the shipped `vae_mimic.yaml` (`trainer.check_val_every_n_epoch: 5`,
    `log_every_n_epochs: 5` on `vae_visualization`) the plot and validation epochs never coincide,
    so no images are written. Add `callbacks.2.params.log_every_n_epochs=1` to get a plot after
    every validation run. See [Visualization](../visualization/guide.md).

Outputs:

```
runs/<experiment.name>/seed_<seed>/
├── config_resolved.yaml
├── metadata.json
├── checkpoints/   # epochNNN-stepNNNNNN.ckpt (top-3 by val_loss) + last.ckpt
├── samples/       # reconstruction plots from VAEVisualizationCallback
├── metrics/       # train_val_metrics.json
└── tb/            # TensorBoard logs
```

## 5. Use the trained model

```python
from ecgen.families.vae import VAELightning

model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
model.eval()

samples = model.sample(n_samples=16, seq_length=5000)   # (16, 12, 5000)
```

!!! note
    The checkpoint stores the flattened `VAEConfig` fields as hyper-parameters, and `VAELightning`
    rebuilds its config from them, so no `config=` argument is needed. Passing one explicitly
    (`load_from_checkpoint(path, config=VAEConfig(...))`) still works and takes precedence.

## Next steps

- [Training with `ecgen train`](../training/runner.md) — CLI, config schema, overrides
- [Configuration](configuration.md) — how a config is applied
- [VAE model](../models/vae.md) and [architecture overview](../models/architecture.md)
- [VAE training guide](../training/vae_training.md)
- [Visualization](../visualization/guide.md)
