# Pulse2Pulse Training in ECGEN

This document describes the integration of Pulse2Pulse WaveGAN for ECG generation into the ECGEN framework.

## Overview

Pulse2Pulse is a WaveGAN-based generative model for synthesizing realistic 8-lead ECG signals. The implementation has been integrated into ECGEN with PyTorch Lightning for streamlined training and evaluation.

## Quick Start

### 1. Test the Setup

```bash
cd /work/vajira/DL2026/ECGEN
python scripts/test_models_only.py
```

Expected output: All tests should pass, confirming the model architecture is working correctly.

### 2. Train the Model

#### Option A: Using the Standalone Script (Recommended for Quick Start)

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --exp-name pulse2pulse_mimic \
    --batch-size 128 \
    --max-epochs 300 \
    --accelerator gpu \
    --devices 0
```

#### Option B: Using the Config-Based Trainer

```bash
python -m ecgen.training.train \
    --config configs/experiments/pulse2pulse_mimic.yaml
```

### 3. Generate Samples

After training, generate ECG samples from a checkpoint:

```bash
python scripts/generate_pulse2pulse.py \
    --checkpoint runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt \
    --output-dir generated_samples \
    --n-samples 16
```

## Project Structure

```
ECGEN/
├── src/ecgen/
│   ├── models/
│   │   └── pulse2pulse.py          # Generator, Discriminator, Lightning module
│   ├── data/
│   │   ├── pulse2pulse_mimic.py    # MIMIC-IV-ECG data module
│   │   └── mimic_dataset.py        # Base MIMIC dataset
│   └── training/
│       ├── train.py                # Main training script
│       ├── callbacks.py            # Visualization callbacks
│       ├── losses.py               # Loss functions
│       └── metrics.py              # Evaluation metrics
├── scripts/
│   ├── train_pulse2pulse.py        # Standalone training script
│   ├── generate_pulse2pulse.py     # Sample generation script
│   └── test_models_only.py         # Model architecture tests
├── configs/
│   └── experiments/
│       └── pulse2pulse_mimic.yaml  # Training configuration
└── docs/
    └── pulse2pulse_training.md     # Detailed training guide
```

## Key Components

### 1. Model Architecture

#### Generator (`WaveGANGenerator`)
- **Input**: Random noise (B, 8, 5000)
- **Architecture**: U-Net style with skip connections
  - Encoder: 6 convolutional layers
  - Decoder: 6 transposed convolutional layers
- **Output**: Generated ECG (B, 8, 5000)
- **Parameters**: ~10.5M

#### Discriminator (`WaveGANDiscriminator`)
- **Input**: ECG signal (B, 8, 5000)
- **Architecture**: 7 convolutional layers with phase shuffling
- **Output**: Scalar score (B, 1)
- **Parameters**: ~334M

### 2. Training Algorithm

Uses WGAN-GP (Wasserstein GAN with Gradient Penalty):
- Discriminator updates: Every step
- Generator updates: Every 5 steps (configurable via `n_critic`)
- Gradient penalty coefficient: λ=10.0

### 3. Data Module

`Pulse2PulseMIMICDataModule` handles:
- Loading MIMIC-IV-ECG dataset
- Filtering to 8 leads
- Train/val/test splits
- Data normalization

### 4. Callbacks

#### ECGVisualizationCallback
- Generates real vs. fake comparison plots
- Frequency: Every 10 epochs (default)
- Output: `samples/sample_epoch_XXXX.png`

#### GeneratedSamplesCallback
- Saves generated samples as tensors and plots
- Frequency: Every 25 epochs (default)
- Output: `samples/generated_epoch_XXXX.{pt,png}`

## Configuration

### Model Parameters

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      model_size: 50          # Base channel multiplier
      num_channels: 8         # Number of ECG leads
      seq_length: 5000        # ECG sequence length
      lr: 0.0001             # Learning rate
      b1: 0.5                # Adam beta1
      b2: 0.9                # Adam beta2
      lmbda: 10.0            # Gradient penalty coefficient
      n_critic: 5            # Discriminator updates per generator update
```

### Data Parameters

```yaml
data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      data_dir: /path/to/MIMIC-IV-ECG
      batch_size: 128
      num_workers: 4
      max_samples: null      # Use all samples
      skip_missing_check: true
      num_channels: 8
      seq_length: 5000
```

### Training Parameters

```yaml
trainer:
  max_epochs: 300
  accelerator: "gpu"
  devices: [0]
  log_every_n_steps: 50
  check_val_every_n_epoch: 5
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb
```

### Logged Metrics

- `train/d_loss`: Discriminator loss
- `train/d_wasserstein`: Wasserstein distance (higher is better)
- `train/g_loss`: Generator loss
- `val/d_wasserstein`: Validation Wasserstein distance
- `val_loss`: Validation loss (for checkpointing)

## Output Structure

```
runs/pulse2pulse_mimic/seed_42/
├── checkpoints/
│   ├── last.ckpt                    # Latest checkpoint
│   ├── epoch000-step000123.ckpt     # Top-K checkpoints
│   └── ...
├── samples/
│   ├── sample_epoch_0010.png        # Real vs. fake comparisons
│   ├── generated_epoch_0025.png     # Generated samples (plot)
│   ├── generated_epoch_0025.pt      # Generated samples (tensor)
│   └── ...
├── tb/                              # TensorBoard logs
├── config_resolved.yaml             # Resolved configuration
└── metadata.json                    # Run metadata
```

## Differences from Original pulse2pulse_simple.py

### Improvements

1. **PyTorch Lightning Integration**
   - Automatic GPU handling
   - Built-in checkpointing
   - TensorBoard logging
   - Cleaner training loop

2. **Modular Design**
   - Separate model, data, training, and callback modules
   - Config-based training
   - Reusable components

3. **Better Visualization**
   - Automatic sample generation during training
   - Real vs. fake comparison plots
   - Saved sample tensors for analysis

4. **Improved Data Handling**
   - LightningDataModule for data loading
   - Automatic train/val/test splits
   - Better error handling

5. **Enhanced Checkpointing**
   - Save top-K checkpoints based on validation metric
   - Automatic last checkpoint saving
   - Easy resume from checkpoint

### Preserved Features

- Same model architecture (WaveGANGenerator, WaveGANDiscriminator)
- Same training algorithm (WGAN-GP)
- Same hyperparameters (by default)
- Compatible with MIMIC-IV-ECG dataset

## Troubleshooting

### Issue: Import errors with markupsafe/jinja2

**Solution**: Upgrade jinja2
```bash
pip install --upgrade jinja2
```

### Issue: Shape mismatch in generator

**Solution**: Ensure `post_proc_filt_len=0` in generator config (default)

### Issue: Out of memory

**Solutions**:
- Reduce batch size
- Reduce model_size parameter
- Use gradient accumulation

### Issue: Training is unstable

**Solutions**:
- Reduce learning rate
- Increase gradient penalty coefficient (lmbda)
- Increase n_critic

## References

1. **Original Implementation**: `/work/vajira/DL2025/deepfakeECGLDM/deepfake/pulse2pulse_simple.py`
2. **WaveGAN Paper**: Donahue et al., "Adversarial Audio Synthesis" (2019)
3. **WGAN-GP Paper**: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
4. **MIMIC-IV-ECG**: Goldberger et al., PhysioNet (2000)

## Next Steps

1. **Train the model**: Start with default hyperparameters
2. **Monitor training**: Use TensorBoard to track metrics
3. **Evaluate samples**: Check generated ECGs every 10-25 epochs
4. **Tune hyperparameters**: Adjust based on training dynamics
5. **Generate samples**: Use trained model to generate new ECGs

## Support

For detailed training instructions, see `docs/pulse2pulse_training.md`.

For issues or questions, check the troubleshooting section or review the test scripts.
