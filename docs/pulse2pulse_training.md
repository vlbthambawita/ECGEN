# Pulse2Pulse WaveGAN Training Guide

This guide explains how to train the Pulse2Pulse WaveGAN model for ECG generation using the ECGEN framework.

## Overview

Pulse2Pulse is a WaveGAN-based model for generating realistic 8-lead ECG signals. The model uses:
- **Generator**: U-Net style architecture with skip connections
- **Discriminator**: Convolutional network with phase shuffling
- **Training**: WGAN-GP (Wasserstein GAN with Gradient Penalty)

## Quick Start

### Method 1: Using the Standalone Script

```bash
cd /work/vajira/DL2026/ECGEN

python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --exp-name pulse2pulse_mimic \
    --batch-size 128 \
    --max-epochs 300 \
    --accelerator gpu \
    --devices 0
```

### Method 2: Using the Config-Based Trainer

```bash
cd /work/vajira/DL2026/ECGEN

python -m ecgen.training.train \
    --config configs/experiments/pulse2pulse_mimic.yaml
```

## Configuration

### Model Parameters

- `model_size`: Base channel multiplier (default: 50)
- `num_channels`: Number of ECG leads (default: 8)
- `seq_length`: ECG sequence length in samples (default: 5000)
- `lr`: Learning rate (default: 1e-4)
- `b1`, `b2`: Adam optimizer betas (default: 0.5, 0.9)
- `lmbda`: Gradient penalty coefficient (default: 10.0)
- `n_critic`: Discriminator updates per generator update (default: 5)

### Data Parameters

- `data_dir`: Path to MIMIC-IV-ECG dataset
- `batch_size`: Training batch size (default: 128)
- `num_workers`: Number of data loading workers (default: 4)
- `max_samples`: Limit dataset size for debugging (default: None)
- `skip_missing_check`: Skip file existence check (default: True)

### Training Parameters

- `max_epochs`: Maximum training epochs (default: 300)
- `accelerator`: Device type (gpu/cpu)
- `devices`: GPU device IDs
- `log_every_n_steps`: Logging frequency (default: 50)
- `check_val_every_n_epoch`: Validation frequency (default: 5)

## Output Structure

Training outputs are organized as follows:

```
runs/
└── pulse2pulse_mimic/
    └── seed_42/
        ├── checkpoints/
        │   ├── last.ckpt
        │   ├── epoch000-step000123.ckpt
        │   └── ...
        ├── samples/
        │   ├── sample_epoch_0010.png
        │   ├── generated_epoch_0025.png
        │   ├── generated_epoch_0025.pt
        │   └── ...
        ├── tb/
        │   └── (TensorBoard logs)
        ├── config_resolved.yaml
        └── metadata.json
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb
```

### Metrics

The following metrics are logged during training:

- `train/d_loss`: Discriminator loss (lower is better)
- `train/d_wasserstein`: Wasserstein distance estimate (higher is better)
- `train/g_loss`: Generator loss (lower is better)
- `val/d_wasserstein`: Validation Wasserstein distance
- `val_loss`: Validation loss (used for checkpointing)

## Callbacks

### ECGVisualizationCallback

Generates comparison plots of real vs. generated ECG samples.

- **Frequency**: Every 10 epochs (configurable)
- **Output**: `samples/sample_epoch_XXXX.png`

### GeneratedSamplesCallback

Saves generated ECG samples as tensors and plots.

- **Frequency**: Every 25 epochs (configurable)
- **Output**: 
  - `samples/generated_epoch_XXXX.pt` (tensor)
  - `samples/generated_epoch_XXXX.png` (plot)

## Checkpointing

Checkpoints are saved automatically:

- `last.ckpt`: Latest checkpoint (updated every epoch)
- `epochXXX-stepXXXXXX.ckpt`: Top-3 checkpoints based on validation Wasserstein distance

### Resume Training

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --resume runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt
```

## Model Architecture

### Generator (WaveGANGenerator)

- **Input**: Noise tensor (B, 8, 5000)
- **Architecture**: 
  - Encoder: 6 convolutional layers with stride 2/5
  - Decoder: 6 transposed convolutional layers with skip connections
  - Activation: LeakyReLU (encoder), ReLU (decoder), Tanh (output)
- **Output**: Generated ECG (B, 8, 5000)

### Discriminator (WaveGANDiscriminator)

- **Input**: ECG tensor (B, 8, 5000)
- **Architecture**:
  - 7 convolutional layers with stride 2/4
  - Phase shuffling after each layer
  - Dynamic final linear layer
  - Activation: LeakyReLU
- **Output**: Scalar score (B, 1)

## Training Algorithm

The training follows WGAN-GP:

1. **Discriminator Update** (every step):
   - Sample real ECGs from dataset
   - Generate fake ECGs from noise
   - Compute D(real) and D(fake)
   - Compute gradient penalty on interpolated samples
   - Update D to maximize: D(real) - D(fake) - λ·GP

2. **Generator Update** (every n_critic steps):
   - Generate fake ECGs from noise
   - Compute D(fake)
   - Update G to maximize: D(fake)

## Tips and Best Practices

1. **Batch Size**: Use the largest batch size that fits in GPU memory (128-256 recommended)
2. **Learning Rate**: Start with 1e-4; reduce if training is unstable
3. **Gradient Penalty**: λ=10.0 works well; increase if discriminator dominates
4. **Training Time**: Expect 24-48 hours for 300 epochs on a single GPU
5. **Validation**: Check generated samples every 10-25 epochs for quality assessment
6. **Convergence**: Monitor Wasserstein distance; should increase over time

## Troubleshooting

### Issue: Discriminator loss explodes

**Solution**: Reduce learning rate or increase gradient penalty coefficient

### Issue: Generator produces mode collapse

**Solution**: Increase n_critic (e.g., 10) or reduce generator learning rate

### Issue: Training is too slow

**Solution**: 
- Increase batch size
- Reduce validation frequency
- Use mixed precision training (add `precision=16` to trainer)

### Issue: Out of memory

**Solution**:
- Reduce batch size
- Reduce model_size parameter
- Use gradient accumulation

## Advanced Usage

### Custom Callbacks

Add custom callbacks in the config:

```yaml
callbacks:
  - target: ecgen.training.callbacks.ECGVisualizationCallback
    params:
      save_dir: runs/pulse2pulse_mimic/seed_42/samples
      every_n_epochs: 10
  - target: your.custom.Callback
    params:
      param1: value1
```

### Multi-GPU Training

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --accelerator gpu \
    --devices 0 1 2 3
```

### Mixed Precision Training

Add to config or script:

```python
trainer = pl.Trainer(
    ...,
    precision=16,
)
```

## References

- Original WaveGAN paper: Donahue et al., "Adversarial Audio Synthesis" (2019)
- WGAN-GP: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
- MIMIC-IV-ECG: Goldberger et al., PhysioNet (2000)
