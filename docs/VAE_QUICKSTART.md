# VAE Quick Start Guide

This guide will help you quickly get started with the VAE model in ECGEN.

## Installation

The VAE model is now part of ECGEN. No additional installation is required if you already have the ECGEN dependencies installed.

Required dependencies:
- PyTorch
- PyTorch Lightning
- NumPy

## Quick Test

Verify the installation:

```bash
cd /work/vajira/DL2026/ECGEN
PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH python tests/test_vae.py
```

## Basic Usage

### 1. Import the Model

```python
from ecgen.models import VAE1D, VAEConfig, VAELightning, vae_loss
import torch
```

### 2. Create a Model

```python
# Option 1: Direct model
model = VAE1D(
    in_channels=12,      # 12-lead ECG
    base_channels=64,    # Base feature channels
    latent_channels=8,   # Latent space dimensions
)

# Option 2: Lightning module (recommended for training)
config = VAEConfig(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
    lr=1e-4,
    kl_weight=0.0001,
)
model = VAELightning(config)
```

### 3. Forward Pass

```python
# Input: [batch_size, channels, length]
x = torch.randn(4, 12, 5000)

# Forward pass
recon, mean, logvar = model(x)

# Calculate loss
total_loss, recon_loss, kl_loss = vae_loss(
    recon, x, mean, logvar, 
    kl_weight=0.0001
)
```

### 4. Training

```python
import pytorch_lightning as pl

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)

# Train (you need to provide your datamodule)
# trainer.fit(model, datamodule=your_datamodule)
```

### 5. Generation

```python
# Reconstruct existing signals
reconstructed = model.reconstruct(ecg_signals)

# Sample new signals from latent space
samples = model.sample(n_samples=16, seq_length=5000)

# Encode to latent space
latent = model.vae.encode_to_latent(ecg_signals)

# Decode from latent space
decoded = model.vae.decode_from_latent(latent)
```

## Example: Complete Training Loop

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ecgen.models import VAEConfig, VAELightning

# 1. Configure model
config = VAEConfig(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
    lr=1e-4,
    kl_weight=0.0001,
)

# 2. Create model
model = VAELightning(config)

# 3. Setup callbacks
checkpoint = ModelCheckpoint(
    dirpath="checkpoints",
    filename="vae-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
)

# 4. Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop],
    accelerator="gpu",
    devices=1,
)

# 5. Train
# trainer.fit(model, datamodule=your_datamodule)
```

## Configuration Options

### Model Architecture

```python
config = VAEConfig(
    in_channels=12,              # Number of ECG leads
    base_channels=64,            # Base number of channels
    latent_channels=8,           # Latent space channels
    channel_multipliers=(1,2,4,4), # Channel scaling per stage
    num_res_blocks=2,            # Residual blocks per stage
)
```

### Training Hyperparameters

```python
config = VAEConfig(
    lr=1e-4,                     # Learning rate
    kl_weight=0.0001,            # KL divergence weight
    b1=0.9,                      # Adam beta1
    b2=0.999,                    # Adam beta2
)
```

## Common Use Cases

### 1. ECG Reconstruction

```python
# Load trained model
model = VAELightning.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Reconstruct ECG
with torch.no_grad():
    reconstructed = model.reconstruct(ecg_signals)
```

### 2. Latent Space Exploration

```python
# Encode multiple ECGs
latents = []
for ecg in ecg_dataset:
    z = model.vae.encode_to_latent(ecg)
    latents.append(z)

# Interpolate in latent space
z1, z2 = latents[0], latents[1]
alpha = torch.linspace(0, 1, 10)
interpolated = [(1-a)*z1 + a*z2 for a in alpha]

# Decode interpolated latents
generated = [model.vae.decode_from_latent(z) for z in interpolated]
```

### 3. Integration with Diffusion Models

```python
# 1. Train VAE
vae_model = VAELightning(vae_config)
# trainer.fit(vae_model, datamodule)

# 2. Freeze VAE and extract latents
vae_model.freeze()
latents = vae_model.vae.encode_to_latent(ecg_signals)

# 3. Train diffusion model on latents
# diffusion_model = DiffusionModel(latent_dim=8)
# trainer.fit(diffusion_model, latent_datamodule)

# 4. Generate new ECGs
# latent_samples = diffusion_model.sample(n_samples=16)
# ecg_samples = vae_model.vae.decode_from_latent(latent_samples)
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'ecgen'`:

```bash
# Set PYTHONPATH
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH

# Or install in development mode
cd /work/vajira/DL2026/ECGEN
pip install -e .
```

### Memory Issues

If you run out of memory:

1. Reduce batch size
2. Reduce `base_channels` (e.g., 32 instead of 64)
3. Reduce `num_res_blocks` (e.g., 1 instead of 2)
4. Use gradient checkpointing

### Poor Reconstruction

If reconstruction quality is poor:

1. Increase training epochs
2. Reduce `kl_weight` (e.g., 0.00001)
3. Increase `base_channels` (e.g., 128)
4. Add more residual blocks

## Next Steps

1. Read the full documentation: `docs/VAE_MODEL.md`
2. Check the training script: `scripts/train_vae.py`
3. Run the tests: `tests/test_vae.py`
4. Adapt the data module for your dataset
5. Start training!

## Support

For issues or questions:
1. Check the documentation in `docs/VAE_MODEL.md`
2. Review the test file `tests/test_vae.py` for examples
3. Examine the training script `scripts/train_vae.py`
