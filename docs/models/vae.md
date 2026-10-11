# VAE

A 1-D convolutional variational autoencoder for multi-lead ECG, in
[`src/ecgen/families/vae/`](https://github.com/vlbthambawita/ECGEN/tree/main/src/ecgen/families/vae).

!!! tip "Visual guide"
    For animated diagrams of the input, architecture, output and loss, see the
    [visual guide](../visual/vae.md).

| Class | File | Role |
|-------|------|------|
| `VAEConfig` | `src/ecgen/families/vae/config.py` | Dataclass of architecture + optimiser settings |
| `VAE1D` | `src/ecgen/families/vae/vae_1d.py` | `nn.Module`: `Encoder1D` → reparameterisation → `Decoder1D` |
| `VAELightning` | `src/ecgen/families/vae/vae_lightning.py` | LightningModule: loss, logging, Adam, `reconstruct()`, `sample()` |
| `vae_loss` | `src/ecgen/families/vae/losses.py` | MSE + weighted KL |

```python
from ecgen.families.vae import VAE1D, VAELightning, VAEConfig
from ecgen.families.vae.losses import vae_loss
```

## Architecture

`VAE1D` is built from the shared blocks in `src/ecgen/nn/` (see
[Architecture overview](architecture.md#shared-components-srcecgennn)):

- **Encoder** (`Encoder1D`): `Conv1d(k=7)` → per stage, `num_res_blocks` × `ResidualBlock1D`
  (GroupNorm + SiLU, residual shortcut) with a stride-2 `Conv1d(k=4)` between stages → two middle
  residual blocks → `Conv1d(k=3)` producing `2·latent_channels` channels, split into `mean` and
  `logvar`.
- **Latent**: Gaussian, `z = mean + exp(0.5·logvar)·ε`.
- **Decoder** (`Decoder1D`): `Conv1d(k=3)` → two middle residual blocks → per stage residual blocks
  with stride-2 `ConvTranspose1d(k=4)` between stages → GroupNorm → SiLU → `Conv1d(k=7)`.
  No output activation.

The latent is a **sequence**, not a vector: shape `(B, latent_channels, L / 2**(S-1))` where
`S = len(channel_multipliers)`. With the defaults, a `(B, 12, 5000)` input gives a `(B, 8, 625)`
latent (8× temporal compression).

!!! tip "Sequence length"
    `seq_length` should be divisible by `2**(S-1)` for the reconstruction to have the input length.
    5000 works with the default four-stage setup (`5000 / 8 = 625`).

## Configuration (`VAEConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `in_channels` | `12` | Number of ECG leads (also the decoder's output channels) |
| `base_channels` | `64` | Width of the first stage |
| `latent_channels` | `8` | Channels of the latent sequence |
| `channel_multipliers` | `(1, 2, 4, 4)` | Per-stage width multipliers; length sets the number of stages |
| `num_res_blocks` | `2` | Residual blocks per stage |
| `lr` | `1e-4` | Adam learning rate |
| `kl_weight` | `0.0001` | Weight of the KL term |
| `b1`, `b2` | `0.9`, `0.999` | Adam betas |
| `conditioning` | `None` | Conditioning config → conditional VAE (see [Conditioning](../training/conditioning.md#conditional-families)) |
| `learned_prior` | `False` | Learn `p(z \| c)` instead of N(0, I) (needs `conditioning`) |
| `recon_on` | `"all"` | `"unobserved"`: reconstruction loss only where `obs_mask` is 0 |
| `latent_type` | `"spatial"` | `"global"`: one `global_latent_dim` vector per ECG; prior samples decode to coherent ECGs |
| `global_latent_dim` | `64` | Size of the global latent |
| `seq_length` | `5000` | Signal length, needed by `latent_type: global` |

!!! note "Sampling from a spatial-latent VAE"
    With `latent_type: spatial` every latent position has an independent N(0, 1) prior, so
    `sample()` draws incoherent latents and the decoder output is not a realistic ECG. The
    spatial VAE is a good autoencoder (e.g. the first stage of a latent diffusion model), not
    a sampler on its own. Use `latent_type: global` for sampling.

All widths (`base_channels × m`) must be divisible by 8 because of `GroupNorm(8, ·)`.

Parameter counts (`VAE1D`, 12 leads):

| Setting | Parameters |
|---------|-----------:|
| `base_channels=32`, other defaults | ~1.57 M |
| Defaults | ~6.22 M |
| `configs/model/vae/vae_1d_large.yaml` (`base_channels=128`, `latent_channels=16`, `channel_multipliers=(1, 2, 4, 8)`, `num_res_blocks=3`) | ~82.8 M |

## Loss

```python
total, recon, kl = vae_loss(recon, x, mean, logvar, kl_weight=0.0001)
```

- `recon = F.mse_loss(recon, x)` (mean over all elements)
- `kl = -0.5 · Σ(1 + logvar − mean² − exp(logvar)) / (B · C · L)` — summed over the latent and
  normalised by the **input** size `B·C·L`
- `total = recon + kl_weight · kl`

`kl_weight` trades reconstruction fidelity against how close the latent stays to N(0, I); it matters
for the quality of `sample()`.

## Usage

### Plain module

```python
import torch
from ecgen.families.vae import VAE1D
from ecgen.families.vae.losses import vae_loss

model = VAE1D(in_channels=12, base_channels=64, latent_channels=8,
              channel_multipliers=(1, 2, 4, 4), num_res_blocks=2)

x = torch.randn(4, 12, 5000)
recon, mean, logvar = model(x)            # (4,12,5000), (4,8,625), (4,8,625)
total, rec, kl = vae_loss(recon, x, mean, logvar, kl_weight=1e-4)
```

Other `VAE1D` methods: `encode(x) -> (mean, logvar)`, `reparameterize(mean, logvar)`,
`decode(z)`, and the no-grad helpers `encode_to_latent(x)` (returns `mean`) and
`decode_from_latent(z)`, intended for a future latent-diffusion stage.

### Lightning module

```python
from ecgen.families.vae import VAELightning, VAEConfig

model = VAELightning(VAEConfig(latent_channels=8, kl_weight=1e-4))
# or: VAELightning({"latent_channels": 8, "kl_weight": 1e-4})

recon = model.reconstruct(x)                          # no-grad, eval mode
samples = model.sample(n_samples=16, seq_length=5000) # z ~ N(0, I) of shape (16, 8, 625) -> decode
```

Logged metrics: `train/total_loss`, `train/recon_loss`, `train/kl_loss` (step + epoch),
`val/total_loss`, `val/recon_loss`, `val/kl_loss`, and `val_loss` (used for checkpointing / early
stopping). The optimiser is a single Adam with no LR scheduler.

!!! note "Batch format"
    `VAELightning` reads the signal with `get_signal(batch)`, so it trains on the `ECGBatch` objects
    produced by `ECGDataModule` and also accepts legacy `{"ecg_signals": ...}` dicts and
    `(ecg, features)` tuples. Train it with `ecgen train configs/experiments/vae_mimic.yaml`; see
    [VAE training](../training/vae_training.md).

### Loading a checkpoint

Checkpoints from `VAELightning` (and from the removed `train_vae_mimic.py` script's
`VAEDataWrapper`) store the flattened `VAEConfig` fields as hyper-parameters and the weights under
`vae.*`, so they can be loaded into `VAELightning`. The constructor accepts either `config=` or the flat config fields as keyword
arguments, so a plain `load_from_checkpoint` works:

```python
from ecgen.families.vae import VAELightning

model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt", map_location="cpu")
model.eval()
```

Passing `config=VAEConfig(...)` explicitly still works and takes precedence over the stored
hyper-parameters. Unknown keyword arguments raise `TypeError: Unexpected VAEConfig fields: [...]`.

### Latent interpolation

```python
z1 = model.vae.encode_to_latent(x[:1])
z2 = model.vae.encode_to_latent(x[1:2])
frames = [model.vae.decode_from_latent((1 - a) * z1 + a * z2) for a in torch.linspace(0, 1, 10)]
```

## Tuning hints

- Blurry / over-smoothed reconstructions: lower `kl_weight`, or increase `base_channels` /
  `num_res_blocks`.
- Good reconstructions but poor random samples: the latent is far from N(0, I); raise `kl_weight`.
- Out of memory: lower the batch size, `base_channels` or `num_res_blocks`.

## Related

- [VAE training](../training/vae_training.md)
- [Visualization during training](../visualization/guide.md)
- [VQ-VAE](vqvae.md) — discrete-latent variant built from the same encoder/decoder
- [API reference: models](../api/models.md)
