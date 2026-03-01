# VAE Models

Variational Autoencoder (VAE) models for ECG generation.

## Files

- `vae_1d.py` - VAE1D model architecture
- `vae_lightning.py` - PyTorch Lightning wrapper
- `config.py` - VAEConfig dataclass

## Usage

```python
from models.vae import VAE1D, VAELightning, VAEConfig

# Create config
config = VAEConfig(
    in_channels=12,
    base_channels=64,
    latent_channels=8,
)

# Create Lightning model
model = VAELightning(config)
```
