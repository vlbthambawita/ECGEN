# Architecture Overview

ECGEN models operate on multi-lead ECG tensors of shape `(batch, num_leads, seq_length)`; the
MIMIC-IV-ECG pipeline produces `12 × 5000` by default (10 s at 500 Hz).

| Family | Package | Status | Main classes | Details |
|--------|---------|--------|--------------|---------|
| VAE | [`models/vae/`](https://github.com/vlbthambawita/ECGEN/tree/main/models/vae) | Implemented | `VAE1D`, `VAELightning`, `VAEConfig` | [VAE](vae.md) |
| VQ-VAE + PixelCNN prior | [`models/vqvae/`](https://github.com/vlbthambawita/ECGEN/tree/main/models/vqvae) | Implemented | `VQVAE1D`, `VectorQuantizer`, `VQVAELightning`, `PixelCNNPrior`, `PriorLightning`, `VQVAEConfig`, `PriorConfig` | [VQ-VAE](vqvae.md) |
| GAN (Pulse2Pulse) | [`models/gan/`](https://github.com/vlbthambawita/ECGEN/tree/main/models/gan) | Implemented | `WaveGANGenerator`, `WaveGANDiscriminator`, `Pulse2PulseGAN`, `Pulse2PulseConfig` | [GAN](gan.md) |
| Diffusion | [`models/diffusion/`](https://github.com/vlbthambawita/ECGEN/tree/main/models/diffusion) | Placeholder (empty `__init__.py` + README) | — | — |
| State space models | [`models/ssm/`](https://github.com/vlbthambawita/ECGEN/tree/main/models/ssm) | Placeholder (empty `__init__.py` + README) | — | — |

## Package pattern

The VAE and VQ-VAE packages follow the same layout:

- `config.py` — a `@dataclass` holding architecture **and** optimiser hyper-parameters
  (`VAEConfig`, `VQVAEConfig`, `PriorConfig`).
- `*_1d.py` — a plain `nn.Module` (`VAE1D`, `VQVAE1D`).
- `*_lightning.py` — a `pl.LightningModule` wrapper (`VAELightning`, `VQVAELightning`,
  `PriorLightning`) that owns the loss, logging, `configure_optimizers()` and `sample()`.
  The constructor takes either the dataclass or an equivalent `dict`.

Losses live in [`training/losses/`](https://github.com/vlbthambawita/ECGEN/tree/main/training/losses):
`vae_losses.vae_loss`, `vqvae_losses.vqvae_loss` / `prior_loss`, and WGAN / LSGAN helpers in
`gan_losses.py`.

Pulse2Pulse is a single module, `models/gan/pulse2pulse.py`, containing the networks, the config
dataclass, `calc_gradient_penalty()` and the `Pulse2PulseGAN` LightningModule.

## Shared components (`models/components/`)

Exported from `models.components`: `ResidualBlock1D`, `Encoder1D`, `Decoder1D`. Both the VAE and the
VQ-VAE are built from these.

### `ResidualBlock1D(in_channels, out_channels)`

`Conv1d(k=3) → GroupNorm(8) → SiLU → Conv1d(k=3) → GroupNorm(8) → SiLU`, plus a residual shortcut
(1×1 conv when the channel count changes, identity otherwise). Because of `GroupNorm(8, ·)`, every
channel width used in the encoder/decoder must be divisible by 8.

### `Encoder1D`

Arguments (defaults): `in_channels=12`, `base_channels=64`, `latent_channels=8`,
`channel_multipliers=(1, 2, 4, 4)`, `num_res_blocks=2`.

1. `Conv1d(in_channels → base_channels, k=7)`
2. For each multiplier `m`: `num_res_blocks` × `ResidualBlock1D(→ base_channels·m)`; after every
   stage **except the last**, a strided `Conv1d(k=4, stride=2)` halves the length.
3. Two middle `ResidualBlock1D`s, `GroupNorm`, SiLU.
4. `Conv1d(→ 2·latent_channels, k=3)`, split into `(mean, logvar)`.

With *S* = `len(channel_multipliers)` the temporal downsampling factor is `2**(S-1)`; the default
`(1, 2, 4, 4)` gives **8×**, so a 5000-sample input yields a latent of length 625.

### `Decoder1D`

Mirror image of the encoder (multipliers reversed): `Conv1d(latent_channels → ·, k=3)`, two middle
residual blocks, per-stage residual blocks with `ConvTranspose1d(k=4, stride=2)` upsampling between
stages, then `GroupNorm → SiLU → Conv1d(→ out_channels, k=7)`. There is no output activation.

## VAE

`VAE1D` = `Encoder1D` → reparameterisation → `Decoder1D`. Loss = MSE reconstruction +
`kl_weight` × KL (KL normalised by `B·C·L`). See [VAE](vae.md).

## VQ-VAE + PixelCNN prior (two stages)

**Stage 1 — `VQVAE1D` / `VQVAELightning`.** The same `Encoder1D` is used, but only its `mean` output
is kept (default `latent_channels=64`). `VectorQuantizer(num_embeddings=512, embedding_dim=latent_channels, commitment_cost=0.25)`
snaps every latent vector to its nearest codebook entry (L2), with
`vq_loss = ||sg[z_q] - z||² + commitment_cost·||z_q - sg[z]||²` and a straight-through estimator.
`forward()` returns `(recon, vq_loss, indices)` with `indices` of shape `(B, L_latent)`.
Total loss = MSE reconstruction + `vq_loss` (`training.losses.vqvae_loss`).

**Stage 2 — `PixelCNNPrior` / `PriorLightning`.** `PriorLightning` loads a frozen stage-1 checkpoint
(`PriorConfig.vqvae_checkpoint`), encodes each batch to code indices, and trains a WaveNet-style
autoregressive prior: an embedding of the codes shifted right by one position, `num_layers` (default 10)
`CausalGatedBlock`s (causal dilated convs with kernel size 2, dilations `2 ** (i % dilation_cycle)`,
gated `tanh·sigmoid` activations, residual and skip 1×1 convs), and a ReLU/1×1-conv head over the
summed skips that outputs `num_embeddings` logits. The default receptive field is 1024 codes, which
covers the full 625-code sequence. Logit `t` predicts code `t` from codes `< t`, in training and in
sampling. Loss is cross-entropy (`training.losses.prior_loss`).

**Sampling.** `PriorLightning.sample()` draws code indices autoregressively from the prior and decodes
them with the frozen VQ-VAE (`decode_from_indices`). `VQVAELightning.sample()` exists too, but picks
indices uniformly at random and is only a sanity check.

See [VQ-VAE](vqvae.md) and [VQ-VAE training](../training/vqvae_training.md).

## Pulse2Pulse GAN

A WaveGAN-derived, U-Net-style 1-D GAN trained with WGAN-GP:

- `WaveGANGenerator(model_size=50, num_channels=8, ...)` maps a noise tensor of the **same shape as
  the ECG** (`(B, num_channels, seq_length)`, uniform in [-1, 1]) through six strided `Conv1d(k=25)`
  layers and six upsampling layers, four of which concatenate encoder skip connections, ending in
  `tanh`.
- `WaveGANDiscriminator` stacks seven `Conv1d(k=25)` layers with LeakyReLU(0.2) and phase shuffle,
  and lazily creates its final `Linear` layer from the observed input length.
- `Pulse2PulseGAN` uses manual optimisation: one critic step per batch (Wasserstein loss + gradient
  penalty weighted by `lmbda=10`), and a generator step every `n_critic=5` batches; Adam with
  `b1=0.5`, `b2=0.9`, `lr=1e-4`. It defaults to **8 leads** (`num_channels=8`).

!!! note
    `Pulse2PulseGAN` (like `VAELightning`) reads `batch["ecg_signals"]`, i.e. it expects dict
    batches, whereas `MIMICIVECGDataset` yields `(ecg, features)` tuples. The current Pulse2Pulse
    training script still uses the legacy `ecgen` package and its own datamodule.

See [GAN](gan.md) and [Pulse2Pulse training](../training/pulse2pulse_training.md).

## Placeholders: diffusion and SSM

`models/diffusion/` and `models/ssm/` contain only a docstring `__init__.py` and a README stating they
are reserved for future work (the SSM README mentions Mamba and S4). `VAE1D.encode_to_latent()` /
`decode_from_latent()` are provided with a future latent-diffusion stage in mind.

## Adding a new model

1. Create `models/<family>/` with `config.py` (dataclass), the `nn.Module`, and a
   `LightningModule` wrapper; export them from the package `__init__.py`.
2. Reuse `models.components` blocks where they fit; put the loss in `training/losses/`.
3. Copy [`scripts/train/train_template.py`](https://github.com/vlbthambawita/ECGEN/blob/main/scripts/train/train_template.py)
   (or an existing `train_*_mimic.py`) and add an experiment YAML under `configs/experiments/`.
   Remember that the scripts map YAML keys to argparse arguments by hand.
