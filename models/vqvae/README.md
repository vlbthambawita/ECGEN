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
  - Output: `[B, 64, 625]` continuous latent codes

- **Vector Quantizer**: Discretizes continuous latents using a learned codebook
  - Codebook size: 512 discrete codes
  - Embedding dimension: 64
  - Uses straight-through estimator for gradient flow
  - Losses: codebook loss + commitment loss

- **Decoder**: Reconstructs ECG from quantized latents
  - Uses `Decoder1D` from `models/components/decoders.py`
  - Input: `[B, 64, 625]` quantized latent codes
  - Output: `[B, 12, 5000]` reconstructed ECG

### PixelCNN Prior (Stage 2)

The prior model learns the distribution p(z) over discrete codes:

- **Architecture**: WaveNet-style autoregressive prior. Code embeddings are shifted right by one
  position, then passed through `num_layers` (default 10) `CausalGatedBlock`s: causal dilated
  convolutions (kernel size 2, dilation `2 ** (i % dilation_cycle)` = 1, 2, ..., 512) with gated
  `tanh * sigmoid` activations, residual and skip 1x1 convs. The summed skips go through a
  ReLU / 1x1 conv / ReLU / 1x1 conv head. Receptive field: 1024 codes (`prior.receptive_field`),
  enough to cover all 625 codes. About 1.14 M parameters with the defaults
- **Input**: Discrete code indices `[B, 625]`
- **Output**: Logits over codebook `[B, 512, 625]`; the logit at position t predicts code t from codes < t
- **Training**: Cross-entropy loss of each code given the codes before it (same convention as sampling)

> Prior checkpoints from the earlier mask-A/B `GatedMaskedConv1d` architecture cannot be loaded
> into the current `PriorLightning` (different `state_dict` keys). Retrain stage 2; stage-1
> VQ-VAE checkpoints are unaffected.

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
# Using the shell wrapper
./scripts/shell/run_train_vqvae_mimic.sh configs/experiments/vqvae_mimic.yaml

# Or directly
ecgen train configs/experiments/vqvae_mimic.yaml
```

**Training tips:**
- Monitor codebook usage (should use most of the 512 codes)
- Reconstruction loss should decrease steadily
- Typical training: 50-100 epochs

### Stage 2: Train Prior

Train the PixelCNN prior on discrete codes from frozen VQ-VAE:

```bash
# Using the shell wrapper
./scripts/shell/run_train_vqvae_prior_mimic.sh \
    runs/vqvae_mimic/seed_42/checkpoints/last.ckpt \
    configs/experiments/prior_mimic.yaml

# Or directly
ecgen train configs/experiments/prior_mimic.yaml \
    model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
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
indices = vqvae.encode_to_indices(ecg_signals)  # [B, 625]

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
num_layers: 10             # Number of gated causal conv blocks
kernel_size: 2             # Kernel size of the causal convolutions
dilation_cycle: 10         # Dilations 1, 2, ..., 512 -> receptive field of 1024 codes
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
    num_layers=10,
    kernel_size=2,
    dilation_cycle=10,
)

# Logits at position t predict code t from codes < t
logits = prior(indices)
print(prior.receptive_field)  # 1024

# Sample autoregressively
sampled_indices = prior.sample(batch_size=16, latent_length=625)
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
