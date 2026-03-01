# Architecture Overview

Overview of model architectures in ECGEN.

## Model Categories

### VAE (Variational Autoencoders)
- **Purpose**: Learn compressed latent representations
- **Use cases**: Reconstruction, latent space exploration, feature extraction
- **Models**: VAE1D

### GAN (Generative Adversarial Networks)
- **Purpose**: Generate realistic ECG signals
- **Use cases**: Data augmentation, synthetic data generation
- **Models**: Pulse2Pulse

### Diffusion Models (Future)
- **Purpose**: High-quality generation via iterative denoising
- **Use cases**: High-fidelity ECG synthesis
- **Models**: To be implemented

### State Space Models (Future)
- **Purpose**: Efficient sequence modeling
- **Use cases**: Long-sequence ECG modeling
- **Models**: To be implemented (Mamba, S4)

## Shared Components

All models share common building blocks:

- **Encoders**: `models/components/encoders.py`
- **Decoders**: `models/components/decoders.py`
- **Blocks**: `models/components/blocks.py` (ResidualBlock1D, etc.)

## Adding New Models

1. Choose appropriate category (vae, gan, diffusion, ssm)
2. Create model file in category folder
3. Inherit from base classes if applicable
4. Add documentation
5. Update category `__init__.py`
