# VAE Training

The VAE is trained on MIMIC-IV-ECG by the shared runner, `ecgen train`, with
[`configs/experiments/vae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vae_mimic.yaml).
See [Training with `ecgen train`](runner.md) for the config schema and overrides.

## Prerequisites

- `pip install -e .` (see [Installation](../getting_started/installation.md))
- MIMIC-IV-ECG on disk, laid out as described in [MIMIC-IV-ECG](../datasets/mimic.md)
  (`machine_measurements.csv` plus `files/p<XXXX>/p<subject_id>/s<study_id>/<study_id>.hea/.dat`),
  with `data.dataset.params.root` pointing at it

## Quick start

=== "CLI"

    ```bash
    ecgen train configs/experiments/vae_mimic.yaml
    ecgen train configs/experiments/vae_mimic.yaml \
        data.dataset.params.max_samples=1000 trainer.max_epochs=10 wandb.enabled=false
    ```

=== "Shell wrapper"

    ```bash
    ./scripts/shell/run_train_vae_mimic_config.sh                         # configs/experiments/vae_mimic.yaml
    ./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml trainer.max_epochs=10
    ```

    The wrapper `cd`s to the repo root and runs `python -m ecgen train <config> [overrides]`.

=== "Python"

    ```python
    from ecgen.training.runner import train_from_file

    train_from_file("configs/experiments/vae_mimic.yaml", ["trainer.max_epochs=10"])
    ```

Overrides are applied after the YAML, so they always win.

## What the runner builds

- **Model**: `model.name: vae` → `ecgen.families.vae.VAELightning`, with `model.params` as the flat
  `VAEConfig` fields (`in_channels`, `base_channels`, `latent_channels`, `channel_multipliers`,
  `num_res_blocks`, `lr`, `kl_weight`, `b1`, `b2`). Adam (`lr`, `b1`, `b2`), no scheduler.
- **Data**: `ECGDataModule` over `MIMICIVECG` (`data.dataset.name: mimic_iv_ecg`) for the `train` /
  `val` splits; subject-level split with `val_split`, `test_split`, `seed`; `max_samples` caps
  **each** split. Batches are `ECGBatch` objects; the VAE reads `batch.signal` `[B, 12, 5000]`.
- **Callbacks**: `ModelCheckpoint` from `checkpoint:` (top-3 on `val_loss`, plus `last.ckpt`), then
  the YAML list: `early_stopping` (patience 10), `lr_monitor`, `vae_visualization`.
- **Loggers**: TensorBoard always; W&B when `wandb.enabled` (the shipped config enables it).

Logged metrics: `train/{total,recon,kl}_loss`, `val/{total,recon,kl}_loss`, `val_loss`.

## Common settings

| Setting | Key (override form) | `vae_mimic.yaml` |
|------|---------|----------|
| Data directory | `data.dataset.params.root` | cluster path |
| Records per split | `data.dataset.params.max_samples` | `10000` (`null` = all) |
| Skip file check | `data.dataset.params.skip_missing_check` | `false` |
| Leads | `data.dataset.params.leads` | `null` (12, file order) |
| Batch size / workers | `data.batch_size` / `data.num_workers` | `32` / `4` |
| Width / latent | `model.params.base_channels` / `model.params.latent_channels` | `64` / `8` |
| KL weight | `model.params.kl_weight` | `0.0001` |
| Learning rate | `model.params.lr` | `0.0001` |
| Epochs | `trainer.max_epochs` | `100` |
| GPUs | `trainer.devices` | `[0]` |
| Validation interval | `trainer.check_val_every_n_epoch` | `5` |
| Early stopping patience | `callbacks.0.params.patience` | `10` |
| Top-k checkpoints | `checkpoint.save_top_k` | `3` |
| Plot interval | `callbacks.2.params.log_every_n_epochs` | `5` |
| W&B | `wandb.enabled` / `wandb.project` | `true` / `ecg-vae` |
| Resume | `--resume <ckpt>` or `resume=<ckpt>` | `null` |

!!! warning "No reconstruction plots with the shipped config"
    Lightning validates on 0-based epochs 4, 9, 14, ... when `check_val_every_n_epoch: 5`, while
    `VAEVisualizationCallback` only plots when `epoch % log_every_n_epochs == 0`. With both set to 5
    (as in `vae_mimic.yaml`) the callback never fires. Set
    `callbacks.2.params.log_every_n_epochs=1`.

### Examples

```bash
# Smaller, faster model
ecgen train configs/experiments/vae_mimic.yaml \
    model.params.base_channels=32 data.batch_size=64 trainer.max_epochs=50

# Multiple GPUs
ecgen train configs/experiments/vae_mimic.yaml 'trainer.devices=[0, 1]'

# Resume
ecgen train configs/experiments/vae_mimic.yaml --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt

# No data: smoke-test the pipeline on synthetic ECGs
ecgen train configs/experiments/vae_mimic.yaml data.dataset.name=synthetic \
    'data.dataset.params={num_samples: 64}' trainer.max_epochs=1 trainer.check_val_every_n_epoch=1 \
    trainer.accelerator=cpu trainer.devices=1 wandb.enabled=false
```

## Outputs

```
runs/<exp_name>/seed_<seed>/
├── config_resolved.yaml
├── metadata.json
├── checkpoints/
│   ├── epochNNN-stepNNNNNN.ckpt   # top-k by val_loss
│   └── last.ckpt
├── samples/
│   └── epoch_NNNN.png             # VAEVisualizationCallback
├── metrics/train_val_metrics.json
├── tb/version_N/                  # TensorBoard
└── wandb/                         # only if W&B is enabled
```

Monitor with:

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

## Using a trained model

```python
from ecgen.families.vae import VAELightning

model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
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
| `ModuleNotFoundError: No module named 'ecgen'` / `ecgen: command not found` | `pip install -e .` |
| `FileNotFoundError: machine_measurements.csv not found at ...` | `data.dataset.params.root` must point at the directory containing `machine_measurements.csv` and `files/` |
| `ValueError: Unknown config sections: ['visualization']` | Old-style config; move the plot settings to a `vae_visualization` entry under `callbacks:` |
| `ValueError: Model expects 12 channels but the dataset yields ...` | `model.params.in_channels` must equal the number of selected leads |
| `wandb not installed; continuing without it` | The run continues with TensorBoard only; `pip install wandb` and log in, or set `wandb.enabled=false` |
| CUDA out of memory | Lower `batch_size`, `base_channels` or `num_res_blocks` |
| Slow start-up | The missing-file check stats every record; set `data.dataset.params.skip_missing_check=true` once the data is known to be complete |
| Blurry reconstructions | Lower `kl_weight`, or increase model capacity |
| KL collapses to ~0 / samples are poor | Raise `kl_weight` |

## Related

- [VAE model](../models/vae.md)
- [Configuration](../getting_started/configuration.md)
- [Training with `ecgen train`](runner.md)
- [Callbacks](callbacks.md) and [Visualization](../visualization/guide.md)
