# VAE Model for ECG Generation

This document describes the Variational Autoencoder (VAE) model added to the ECGEN repository.

## Overview

The VAE model is a 1D convolutional variational autoencoder designed for ECG signal generation and representation learning. It consists of an encoder that maps ECG signals to a latent distribution and a decoder that reconstructs signals from latent codes.

## Architecture

### Components

1. **ResidualBlock1D**: Basic building block with group normalization and residual connections
2. **Encoder1D**: Downsampling encoder with residual blocks
3. **Decoder1D**: Upsampling decoder with residual blocks
4. **VAE1D**: Complete VAE combining encoder and decoder

### Key Features

- 1D convolutions optimized for time-series data
- Group normalization for stable training
- Residual connections for better gradient flow
- Reparameterization trick for sampling
- Configurable architecture (channels, layers, etc.)

## Usage

### Basic Usage

```python
from ecgen.models.vae import VAE1D, vae_loss
import torch

# Create model
model = VAE1D(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
    channel_multipliers=(1, 2, 4, 4),
    num_res_blocks=2,
)

# Forward pass
x = torch.randn(4, 12, 5000)  # [batch, channels, length]
recon, mean, logvar = model(x)

# Calculate loss
total_loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar, kl_weight=0.0001)
```

### PyTorch Lightning

```python
from ecgen.models.vae import VAEConfig, VAELightning

# Create config
config = VAEConfig(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
    lr=1e-4,
    kl_weight=0.0001,
)

# Create Lightning module
model = VAELightning(config)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, datamodule=your_datamodule)
```

### Generation

```python
# Reconstruct signals
reconstructed = model.reconstruct(ecg_signals)

# Sample from latent space
samples = model.sample(n_samples=16, seq_length=5000)

# Encode to latent space (for diffusion models)
latent = model.vae.encode_to_latent(ecg_signals)

# Decode from latent space
decoded = model.vae.decode_from_latent(latent)
```

## Configuration

### VAEConfig Parameters

- `in_channels` (int): Number of input ECG leads (default: 12)
- `base_channels` (int): Base number of channels (default: 64)
- `latent_channels` (int): Number of latent channels (default: 8)
- `channel_multipliers` (tuple): Channel multipliers for each stage (default: (1, 2, 4, 4))
- `num_res_blocks` (int): Number of residual blocks per stage (default: 2)
- `lr` (float): Learning rate (default: 1e-4)
- `kl_weight` (float): KL divergence weight (default: 0.0001)
- `b1` (float): Adam beta1 (default: 0.9)
- `b2` (float): Adam beta2 (default: 0.999)

## Training

Use the provided training script:

```bash
python scripts/train_vae.py \
    --data_dir /path/to/data \
    --output_dir runs/vae \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --kl_weight 0.0001 \
    --in_channels 12 \
    --base_channels 64 \
    --latent_channels 8
```

## Model Architecture Details

### Encoder

1. Initial 7x1 convolution
2. Multiple downsampling stages with residual blocks
3. Middle residual blocks
4. Output convolution producing mean and logvar

### Decoder

1. Initial 3x1 convolution
2. Middle residual blocks
3. Multiple upsampling stages with residual blocks
4. Output 7x1 convolution

### Latent Space

The model uses a Gaussian latent space with:
- Mean and log-variance predicted by encoder
- Reparameterization trick for sampling
- KL divergence regularization

## Loss Function

The VAE loss consists of two terms:

1. **Reconstruction Loss**: MSE between input and reconstruction
2. **KL Divergence**: Regularizes latent space to be close to N(0,1)

```
total_loss = recon_loss + kl_weight * kl_loss
```

The `kl_weight` parameter controls the trade-off between reconstruction quality and latent space regularization.

## Integration with Diffusion Models

The VAE can be used as a first-stage model for latent diffusion:

1. Train VAE on ECG data
2. Freeze VAE weights
3. Use `encode_to_latent()` to get latent representations
4. Train diffusion model in latent space
5. Use `decode_from_latent()` to generate final ECG signals

## Testing

Run the test suite:

```bash
cd /work/vajira/DL2026/ECGEN
PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH python tests/test_vae.py
```

## Model Parameters

With default configuration (12 channels, base_channels=64, latent_channels=8):
- Total parameters: ~6.2M
- Latent compression: 8x (for 4-stage architecture)

## References

This implementation is adapted from the deepfakeECGLDM project and integrated into the ECGEN repository with the following adjustments:

1. Added PyTorch Lightning wrapper for easier training
2. Added proper type hints and dataclass configuration
3. Followed ECGEN code style and structure
4. Added sampling and generation methods
5. Made compatible with ECGEN data pipeline

## Future Work

- [ ] Add perceptual loss for better reconstruction quality
- [ ] Implement vector quantization (VQ-VAE)
- [ ] Add multi-scale architecture
- [ ] Integrate with existing ECGEN data loaders
- [ ] Add visualization callbacks
- [ ] Implement latent space interpolation
