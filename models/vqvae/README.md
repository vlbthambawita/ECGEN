# VQ-VAE for ECG Generation

Vector Quantized Variational Autoencoder (VQ-VAE) implementation for ECG signal generation using discrete latent representations.

## Overview

VQ-VAE learns discrete latent representations of ECG signals through a two-stage training process:

1. **Stage 1: VQ-VAE Training** - Learn discrete codebook and encoder/decoder
2. **Stage 2: Prior Training** - Learn distribution over discrete codes using PixelCNN

## Architecture

### VQ-VAE (Stage 1)

The VQ-VAE consists of three main components:

- **Encoder**: Compresses ECG signals to continuous latent space
  - Uses `Encoder1D` from `models/components/encoders.py`
  - Input: `[B, 12, 5000]` ECG signals
  - Output: `[B, 64, 312]` continuous latent codes

- **Vector Quantizer**: Discretizes continuous latents using a learned codebook
  - Codebook size: 512 discrete codes
  - Embedding dimension: 64
  - Uses straight-through estimator for gradient flow
  - Losses: codebook loss + commitment loss

- **Decoder**: Reconstructs ECG from quantized latents
  - Uses `Decoder1D` from `models/components/decoders.py`
  - Input: `[B, 64, 312]` quantized latent codes
  - Output: `[B, 12, 5000]` reconstructed ECG

### PixelCNN Prior (Stage 2)

The prior model learns the distribution p(z) over discrete codes:

- **Architecture**: Autoregressive PixelCNN with gated masked convolutions
- **Input**: Discrete code indices `[B, 312]`
- **Output**: Logits over codebook `[B, 512, 312]`
- **Training**: Cross-entropy loss for next-code prediction

## Key Differences from Standard VAE

| Feature | VAE | VQ-VAE |
|---------|-----|--------|
| Latent space | Continuous (Gaussian) | Discrete (codebook) |
| Regularization | KL divergence | Codebook + commitment loss |
| Prior | Standard normal | Learned (PixelCNN) |
| Sampling | Reparameterization trick | Codebook lookup |

## Training

### Stage 1: Train VQ-VAE

Train the VQ-VAE autoencoder to learn discrete representations:

```bash
# Using config file
./scripts/shell/run_train_vqvae_mimic.sh configs/experiments/vqvae_mimic.yaml

# Or directly with Python
python scripts/train/train_vqvae_mimic.py \
    --config configs/experiments/vqvae_mimic.yaml
```

**Training tips:**
- Monitor codebook usage (should use most of the 512 codes)
- Reconstruction loss should decrease steadily
- Typical training: 50-100 epochs

### Stage 2: Train Prior

Train the PixelCNN prior on discrete codes from frozen VQ-VAE:

```bash
# Using shell script (recommended)
./scripts/shell/run_train_prior_mimic.sh \
    runs/vqvae_mimic/seed_42/checkpoints/best.ckpt \
    configs/experiments/prior_mimic.yaml

# Or directly with Python
python scripts/train/train_prior_mimic.py \
    --vqvae-checkpoint runs/vqvae_mimic/seed_42/checkpoints/best.ckpt \
    --config configs/experiments/prior_mimic.yaml
```

**Training tips:**
- Requires trained VQ-VAE checkpoint from Stage 1
- VQ-VAE is frozen during prior training
- Cross-entropy loss should decrease
- Typical training: 50-100 epochs

## Usage

### Reconstruction

```python
from models.vqvae import VQVAELightning

# Load trained VQ-VAE
vqvae = VQVAELightning.load_from_checkpoint("path/to/vqvae.ckpt")

# Reconstruct ECG
recon = vqvae.reconstruct(ecg_signals)
```

### Generation (with trained prior)

```python
from models.vqvae import PriorLightning

# Load trained prior (automatically loads VQ-VAE)
prior = PriorLightning.load_from_checkpoint("path/to/prior.ckpt")

# Generate samples
samples = prior.sample(
    n_samples=16,
    seq_length=5000,
    temperature=1.0,  # Higher = more diverse
)
```

### Extract Discrete Codes

```python
# Encode ECG to discrete indices
indices = vqvae.encode_to_indices(ecg_signals)  # [B, 312]

# Decode from indices
reconstructed = vqvae.decode_from_indices(indices)
```

## Configuration

### VQ-VAE Config

Key hyperparameters in `configs/model/vqvae/vqvae_1d_base.yaml`:

```yaml
num_embeddings: 512        # Codebook size
latent_channels: 64        # Embedding dimension
commitment_cost: 0.25      # Weight for commitment loss
lr: 0.0001                 # Learning rate
```

### Prior Config

Key hyperparameters in `configs/model/vqvae/prior_base.yaml`:

```yaml
num_embeddings: 512        # Must match VQ-VAE
hidden_dim: 128            # PixelCNN hidden dimension
num_layers: 3              # Number of gated conv layers
lr: 0.001                  # Learning rate (higher than VQ-VAE)
```

## Model Components

### VectorQuantizer

```python
from models.vqvae import VectorQuantizer

vq = VectorQuantizer(
    num_embeddings=512,
    embedding_dim=64,
    commitment_cost=0.25,
)

# Quantize continuous latents
vq_loss, quantized, indices = vq(z)
```

### VQVAE1D

```python
from models.vqvae import VQVAE1D

model = VQVAE1D(
    in_channels=12,
    base_channels=64,
    latent_channels=64,
    num_embeddings=512,
)

# Full forward pass
recon, vq_loss, indices = model(ecg)
```

### PixelCNNPrior

```python
from models.vqvae import PixelCNNPrior

prior = PixelCNNPrior(
    num_embeddings=512,
    hidden_dim=128,
    num_layers=3,
)

# Predict next codes
logits = prior(indices)

# Sample autoregressively
sampled_indices = prior.sample(batch_size=16, latent_length=312)
```

## Performance Tips

1. **Codebook Collapse**: If only a few codes are used:
   - Increase commitment cost
   - Use exponential moving average (EMA) for codebook updates
   - Increase codebook size

2. **Poor Reconstruction**: 
   - Train longer (100+ epochs)
   - Increase model capacity (base_channels)
   - Check data preprocessing

3. **Prior Training**:
   - Ensure VQ-VAE is well-trained first
   - Use higher learning rate than VQ-VAE
   - Monitor cross-entropy loss convergence

## References

- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)

## File Structure

```
models/vqvae/
├── __init__.py              # Module exports
├── vector_quantizer.py      # VectorQuantizer module
├── vqvae_1d.py             # VQVAE1D model
├── pixelcnn_prior.py       # PixelCNN prior
├── config.py               # Config dataclasses
├── vqvae_lightning.py      # Lightning wrapper (Stage 1)
├── prior_lightning.py      # Lightning wrapper (Stage 2)
└── README.md               # This file
```
