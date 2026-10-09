# Visualization

ECGEN has two visualization pieces:

1. **During training** — `VAEVisualizationCallback` (in `training/callbacks/visualization.py`) saves
   real-vs-reconstruction plots for the VAE and VQ-VAE.
2. **Offline** — the `visualization` package (`visualization/ecg_plots.py`, `latent_space.py`,
   `training_curves.py`) with plotting helpers that return matplotlib figures.

GAN-specific callbacks (`ECGVisualizationCallback`, `GeneratedSamplesCallback`) are described in
[Callbacks](../training/callbacks.md).

## Reconstruction plots during training

`train_vae_mimic.py` and `train_vqvae_mimic.py` add a `VAEVisualizationCallback` automatically,
configured from the `visualization:` section of the experiment YAML:

```yaml
visualization:
  every_n_epochs: 5         # plot every N epochs (only on epochs where validation runs)
  num_samples: 4            # ECGs per plot, taken from the first validation batch
  plot_all_leads: true      # true: one subplot per lead; false: all leads overlaid
  log_to_tensorboard: false
  log_to_wandb: true        # also needs wandb.enabled: true
```

or, without a config, with `--viz-every-n-epochs`, `--viz-num-samples`, `--viz-plot-all-leads` and
`--viz-log-to-wandb` (`--viz-log-to-tensorboard` defaults to on). As with every key, values in the
YAML override these flags. The prior script (`train_prior_mimic.py`) has no visualization.

!!! warning "Plots appear only on validation epochs"
    The callback runs in `on_validation_epoch_end`, skips the sanity check, and fires when
    `current_epoch % every_n_epochs == 0` (0-based epochs). Lightning validates when
    `(current_epoch + 1) % check_val_every_n_epoch == 0`, i.e. on epochs 4, 9, 14, ... for
    `check_val_every_n_epoch: 5`. A plot is produced only on epochs that satisfy both conditions —
    in particular, the shipped
    `vae_mimic.yaml` (`check_val_every_n_epoch: 5`, `every_n_epochs: 5`) **never** produces a plot.
    Use `every_n_epochs: 1` (as `vqvae_mimic.yaml` does) to plot on every validation epoch.

### Plot layouts

**`plot_all_leads: true`** — a grid with `2 × num_samples` rows and one column per lead: for each
sample a "Real" row (blue) above a "Recon" row (orange). Columns are titled with the standard lead
names `I, II, III, aVR, aVL, aVF, V1–V6`.

**`plot_all_leads: false`** — `num_samples` rows × 2 columns ("Real" and "Reconstructed"), with all
leads overlaid in each panel and a lead legend on the first row.

Both figures are titled `VAE Reconstruction - Epoch N` (also for the VQ-VAE).

### Where the plots go

| Destination | Location |
|-------------|----------|
| Disk | `runs/<exp_name>/seed_<seed>/samples/epoch_{epoch:04d}.png` (150 dpi) |
| TensorBoard | Images tab, tag `vae_reconstruction` (when `log_to_tensorboard` is on) |
| Weights & Biases | key `vae_reconstruction` (when `log_to_wandb` is on and a `WandbLogger` exists) |

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
ls runs/vae_mimic/seed_42/samples/
```

### Using the callback directly

```python
import pytorch_lightning as pl
from training.callbacks import VAEVisualizationCallback

viz = VAEVisualizationCallback(
    save_dir="runs/my_exp/seed_42/samples",
    log_every_n_epochs=1,
    num_samples=8,
    plot_all_leads=True,
    log_to_tensorboard=True,
    log_to_wandb=False,
)
trainer = pl.Trainer(callbacks=[viz], max_epochs=10)
```

Requirements on the model and data:

- validation batches must be `(ecg, features)` tuples (as produced by `MIMICIVECGDataset`);
- the LightningModule must expose `vae` or `vqvae` (or be callable itself) returning a tuple whose
  first element is the reconstruction.

### Reading the plots

| Observation | Likely cause | Try |
|-------------|--------------|-----|
| Smooth / blurry reconstructions, missing P or T waves | KL term too strong (VAE) or too little capacity | Lower `kl_weight`; raise `base_channels` / `num_res_blocks` |
| Reconstructions do not improve across epochs | Learning-rate problem | Adjust `lr` |
| Excellent reconstructions but poor `sample()` output (VAE) | Latent not close to N(0, I) | Raise `kl_weight` |

### Troubleshooting

- **No images**: validation runs on 0-based epochs `e` with `(e + 1) % check_val_every_n_epoch == 0`,
  and the callback needs `e % every_n_epochs == 0` on such an epoch (see the warning above). Setting
  `every_n_epochs: 1` always works. Images are also skipped during the sanity check.
- **Nothing in TensorBoard**: `configs/experiments/vae_mimic.yaml` and `vqvae_mimic.yaml` ship with
  `log_to_tensorboard: false`; set it to `true`.
- **Large figures / memory**: lower `num_samples` or use `plot_all_leads: false`.

## The `visualization` package

```python
from visualization import (
    plot_ecg_leads, plot_ecg_comparison, plot_ecg_overlay,
    plot_latent_space_2d, plot_latent_distribution,
    plot_training_curves, plot_multiple_metrics, plot_learning_rate_schedule,
)
```

Every function returns a `matplotlib.figure.Figure`; save it with `fig.savefig(...)`. ECG inputs
can be `torch.Tensor` or `numpy.ndarray`; time axes are in seconds using `sampling_rate` (default
500 Hz).

### ECG plots — `visualization/ecg_plots.py`

| Function | Signature (defaults) | Plots |
|----------|----------------------|-------|
| `plot_ecg_leads` | `(ecg, lead_names=None, sampling_rate=500, title="ECG Signal", figsize=(15, 10))` | All leads of one ECG `(num_leads, seq_length)`, one row per lead |
| `plot_ecg_comparison` | `(real, generated, lead_idx=0, lead_name="Lead I", sampling_rate=500, figsize=(12, 4))` | One lead, real above generated |
| `plot_ecg_overlay` | `(signals, labels, lead_idx=0, lead_name="Lead I", sampling_rate=500, figsize=(12, 4))` | One lead from several signals, overlaid |

`plot_ecg_leads` needs at least two leads (it indexes the subplot array).

### Latent space — `visualization/latent_space.py`

| Function | Signature (defaults) | Plots |
|----------|----------------------|-------|
| `plot_latent_space_2d` | `(latents, labels=None, method="pca", title="Latent Space Visualization", figsize=(10, 8))` | 2-D PCA (`"pca"`) or t-SNE (`"tsne"`) projection; 3-D latents `(N, C, L)` are flattened |
| `plot_latent_distribution` | `(latents, figsize=(12, 4))` | Per-dimension mean ± std and a histogram of all values |

### Training curves — `visualization/training_curves.py`

| Function | Signature (defaults) | Plots |
|----------|----------------------|-------|
| `plot_training_curves` | `(train_losses, val_losses, metric_name="Loss", figsize=(10, 6))` | Train vs val per epoch |
| `plot_multiple_metrics` | `(metrics_dict, figsize=(15, 10))` | `{"metric": {"train": [...], "val": [...]}}` in a 2-column grid |
| `plot_learning_rate_schedule` | `(learning_rates, figsize=(10, 4))` | LR per epoch, log scale |

### Example: inspect a trained VAE

```python
import torch
from models.vae import VAELightning, VAEConfig
from data.datasets.mimic.dataset import MIMICIVECGDataset
from visualization import plot_ecg_leads, plot_ecg_comparison, plot_latent_space_2d

ckpt = torch.load("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
model = VAELightning(VAEConfig(**ckpt["hyper_parameters"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

ds = MIMICIVECGDataset(mimic_path="/path/to/mimic-iv-ecg", split="test", max_samples=64)
x = torch.stack([ds[i][0] for i in range(64)])        # (64, 12, 5000)

recon = model.reconstruct(x)
plot_ecg_comparison(x[0], recon[0], lead_idx=1, lead_name="II").savefig("lead_II.png")
plot_ecg_leads(model.sample(n_samples=1)[0], title="VAE sample").savefig("sample.png")

latents = model.vae.encode_to_latent(x)                # (64, 8, 625)
plot_latent_space_2d(latents, method="pca").savefig("latent_pca.png")
```

## Related

- [Callbacks](../training/callbacks.md)
- [VAE training](../training/vae_training.md)
- [API reference: visualization](../api/visualization.md)
