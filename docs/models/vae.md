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
# VAE Visualization Guide

This document explains how to use the VAE visualization feature to monitor reconstruction quality during training.

## Overview

The VAE visualization callback automatically generates comparison plots of real vs reconstructed ECG signals during training. This helps you:

- **Monitor reconstruction quality** visually throughout training
- **Detect training issues** early (e.g., mode collapse, poor reconstruction)
- **Compare different model configurations** by examining visual outputs
- **Track improvement** over epochs

## Features

- ✅ **Automatic visualization** at configurable intervals (every N epochs)
- ✅ **Two plot styles**: separate leads or overlaid leads
- ✅ **Multiple samples** per visualization (configurable)
- ✅ **TensorBoard integration** for easy viewing
- ✅ **Weights & Biases integration** (optional)
- ✅ **Saved to disk** for later inspection

## Configuration

### Via YAML Config File

Edit your config file (e.g., `configs/experiments/vae_mimic.yaml`):

```yaml
visualization:
  every_n_epochs: 5        # Generate visualizations every 5 epochs
  num_samples: 4           # Visualize 4 samples per epoch
  plot_all_leads: true     # true = separate subplots, false = overlay
  log_to_tensorboard: true # Log to TensorBoard
  log_to_wandb: false      # Log to Weights & Biases (requires wandb enabled)
```

### Via Command Line

Override config settings with command-line arguments:

```bash
./scripts/run_train_vae_mimic_config.sh configs/experiments/vae_mimic.yaml \
    --viz-every-n-epochs 10 \
    --viz-num-samples 8 \
    --viz-plot-all-leads \
    --viz-log-to-wandb
```

## Visualization Styles

### 1. Separate Leads (Recommended)

Set `plot_all_leads: true` to plot each ECG lead in its own subplot.

**Advantages:**
- Clear view of each individual lead
- Easy to spot lead-specific issues
- Professional medical-style layout

**Output:** Each sample shows 2 rows × 12 columns (real + reconstructed for all 12 leads)

### 2. Overlaid Leads

Set `plot_all_leads: false` to overlay all leads in a single plot.

**Advantages:**
- Compact visualization
- Good for quick overview
- Shows overall signal shape

**Output:** Each sample shows 2 plots side-by-side (real vs reconstructed)

## Output Locations

Visualizations are saved to multiple locations:

1. **Disk**: `runs/{exp_name}/seed_{seed}/samples/epoch_{epoch:04d}.png`
2. **TensorBoard**: View in the "IMAGES" tab
3. **Weights & Biases**: View in the "Media" panel (if enabled)

## Viewing Visualizations

### During Training

**TensorBoard:**
```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

Then open http://localhost:6006 and navigate to the "IMAGES" tab.

**Weights & Biases:**

If W&B is enabled, visualizations appear automatically in your run's dashboard.

### After Training

Browse the saved images:

```bash
ls runs/vae_mimic/seed_42/samples/
# epoch_0000.png, epoch_0005.png, epoch_0010.png, ...
```

View with any image viewer or Python:

```python
from PIL import Image
img = Image.open("runs/vae_mimic/seed_42/samples/epoch_0050.png")
img.show()
```

## Interpreting Visualizations

### What to Look For

**Good Reconstruction:**
- Reconstructed signals closely match real signals
- All leads are preserved
- QRS complexes, P waves, and T waves are visible
- Amplitude and timing are similar

**Poor Reconstruction:**
- Blurry or smoothed signals
- Missing features (e.g., P waves)
- Wrong amplitudes
- Phase shifts or timing issues
- Artifacts or noise

### Common Issues

| Observation | Possible Cause | Solution |
|------------|----------------|----------|
| Blurry reconstructions | KL weight too high | Reduce `kl_weight` |
| Perfect reconstructions, high KL loss | KL weight too low | Increase `kl_weight` |
| Noisy reconstructions | Model underfitting | Increase `base_channels` or `num_res_blocks` |
| Missing high-frequency details | Insufficient model capacity | Increase model size |
| Reconstructions don't improve | Learning rate issues | Adjust `lr` |

## Performance Considerations

### Frequency

- **Default: Every 5 epochs** - Good balance between monitoring and performance
- **Quick testing: Every 1 epoch** - Use during debugging
- **Long training: Every 10-20 epochs** - Reduce overhead for very long runs

### Number of Samples

- **Default: 4 samples** - Sufficient for most cases
- **More samples (8-16)** - Better statistical view, but slower
- **Fewer samples (1-2)** - Faster, good for quick checks

### Disk Space

Each visualization is ~500KB-1MB depending on settings. For 100 epochs with `every_n_epochs=5`:
- Total images: 20
- Disk space: ~10-20 MB

## Advanced Usage

### Custom Visualization Callback

You can customize the visualization by using the callback directly:

```python
from ecgen.training.callbacks import VAEVisualizationCallback

# Create custom callback
custom_viz = VAEVisualizationCallback(
    save_dir="custom_output",
    log_every_n_epochs=1,
    num_samples=8,
    plot_all_leads=True,
    log_to_tensorboard=True,
    log_to_wandb=False,
)
```

### Programmatic Access

```python
import pytorch_lightning as pl
from ecgen.training.callbacks import VAEVisualizationCallback

# Add to your trainer
trainer = pl.Trainer(
    callbacks=[
        # ... other callbacks ...
        VAEVisualizationCallback(
            save_dir="visualizations",
            log_every_n_epochs=5,
        ),
    ]
)
```

## Troubleshooting

### Visualizations Not Generated

1. Check that validation is running: `check_val_every_n_epoch` in trainer config
2. Ensure `viz_every_n_epochs` is less than `max_epochs`
3. Check the samples directory exists: `runs/{exp_name}/seed_{seed}/samples/`

### TensorBoard Not Showing Images

1. Verify TensorBoard is pointing to the correct log directory
2. Refresh the browser
3. Check `viz_log_to_tensorboard: true` in config

### Out of Memory

1. Reduce `viz_num_samples`
2. Reduce `batch_size` (affects validation batch)
3. Use `plot_all_leads: false` for smaller plots

## Example Workflow

1. **Start training** with default visualization settings:
   ```bash
   ./scripts/run_train_vae_mimic_config.sh
   ```

2. **Monitor in TensorBoard**:
   ```bash
   tensorboard --logdir runs/vae_mimic/seed_42/tb
   ```

3. **Check early epochs** (0, 5, 10) to ensure training is working

4. **Adjust hyperparameters** if reconstructions are poor

5. **Compare runs** by viewing different experiment directories

## References

- Visualization callback: `src/ecgen/training/callbacks.py` (VAEVisualizationCallback)
- Training script: `scripts/train_vae_mimic.py`
- Config file: `configs/experiments/vae_mimic.yaml`
- Main training script: `scripts/run_train_vae_mimic_config.sh`
