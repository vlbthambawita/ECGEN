# Pulse2Pulse Overview

Pulse2Pulse is a WaveGAN-based generative model for synthesizing realistic 8-lead ECG signals, integrated into the ECGEN framework with PyTorch Lightning support.

## What is Pulse2Pulse?

Pulse2Pulse uses a Generative Adversarial Network (GAN) architecture specifically designed for ECG generation:

- **Generator**: Creates synthetic 8-lead ECG signals from random noise
- **Discriminator**: Distinguishes between real and generated ECGs
- **Training**: Uses WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training

## Model Architecture

### Generator: WaveGANGenerator

The generator uses a U-Net style architecture with skip connections:

```
Input: Random noise (B, 8, 5000)
    ↓
Encoder: 6 convolutional layers
    ↓
Decoder: 6 transposed convolutional layers with skip connections
    ↓
Output: Generated ECG (B, 8, 5000)
```

**Key Features**:
- **Parameters**: ~10.5M
- **Architecture**: U-Net with skip connections
- **Input**: Random noise matching ECG dimensions
- **Output**: 8-lead ECG signal (5000 samples per lead)

### Discriminator: WaveGANDiscriminator

The discriminator uses a convolutional architecture with phase shuffling:

```
Input: ECG signal (B, 8, 5000)
    ↓
7 convolutional layers with phase shuffling
    ↓
Output: Scalar score (B, 1)
```

**Key Features**:
- **Parameters**: ~334M
- **Architecture**: 7-layer CNN
- **Regularization**: Phase shuffling to prevent artifacts
- **Output**: Wasserstein distance estimate

## Training Algorithm

Pulse2Pulse uses **WGAN-GP** (Wasserstein GAN with Gradient Penalty):

1. **Discriminator Update**: Train discriminator to distinguish real from fake
2. **Gradient Penalty**: Add penalty to enforce Lipschitz constraint
3. **Generator Update**: Train generator to fool discriminator (every N steps)

**Hyperparameters**:
- Discriminator updates per generator update: 5 (configurable via `n_critic`)
- Gradient penalty coefficient: λ=10.0
- Learning rate: 0.0001
- Optimizer: Adam (β₁=0.5, β₂=0.9)

## Key Features

### 1. PyTorch Lightning Integration

- ✅ Automatic GPU handling
- ✅ Built-in checkpointing
- ✅ TensorBoard logging
- ✅ Cleaner training loop
- ✅ Easy distributed training

### 2. Modular Design

- ✅ Separate model, data, and training modules
- ✅ Config-based training
- ✅ Reusable components
- ✅ Type hints throughout

### 3. Visualization

- ✅ Real vs. fake comparison plots
- ✅ Automatic sample generation during training
- ✅ Saved sample tensors for analysis
- ✅ TensorBoard integration

### 4. Data Handling

- ✅ LightningDataModule for data loading
- ✅ Automatic train/val/test splits
- ✅ MIMIC-IV-ECG dataset support
- ✅ 8-lead extraction from 12-lead data

## Model Components

### Generator Architecture Details

```python
WaveGANGenerator(
    model_size=50,           # Base channel multiplier
    num_channels=8,          # Number of ECG leads
    seq_length=5000,         # Samples per lead
    use_batch_norm=False,    # Batch normalization
    num_layers_enc=6,        # Encoder depth
    num_layers_dec=6,        # Decoder depth
)
```

**Layer Structure**:
```
Encoder:
  Conv1d(8, 50) → Conv1d(50, 100) → Conv1d(100, 200) → 
  Conv1d(200, 400) → Conv1d(400, 800) → Conv1d(800, 1600)

Decoder (with skip connections):
  TransposeConv1d(1600, 800) → TransposeConv1d(1600, 400) →
  TransposeConv1d(800, 200) → TransposeConv1d(400, 100) →
  TransposeConv1d(200, 50) → TransposeConv1d(100, 8)
```

### Discriminator Architecture Details

```python
WaveGANDiscriminator(
    model_size=50,           # Base channel multiplier
    num_channels=8,          # Number of ECG leads
    seq_length=5000,         # Samples per lead
    use_batch_norm=False,    # Batch normalization
    phaseshuffle_rad=2,      # Phase shuffle radius
)
```

**Layer Structure**:
```
Conv1d(8, 50) → PhaseShuffles → Conv1d(50, 100) → PhaseShuffles →
Conv1d(100, 200) → PhaseShuffles → Conv1d(200, 400) → PhaseShuffles →
Conv1d(400, 800) → PhaseShuffles → Conv1d(800, 1600) → PhaseShuffles →
Conv1d(1600, 1) → Global Average Pool
```

## Training Metrics

### Logged Metrics

- `train/d_loss`: Discriminator loss (lower is better)
- `train/d_wasserstein`: Wasserstein distance (higher is better)
- `train/g_loss`: Generator loss (lower is better)
- `val/d_wasserstein`: Validation Wasserstein distance
- `val_loss`: Validation loss (for checkpointing)

### Interpreting Metrics

**Discriminator Loss**: Should decrease over time
- Negative values are normal with WGAN
- Stable values indicate convergence

**Wasserstein Distance**: Measures distribution difference
- Higher values = discriminator better at distinguishing
- Should increase initially, then stabilize

**Generator Loss**: Should decrease over time
- Indicates how well generator fools discriminator
- Oscillations are normal in GAN training

## Use Cases

Pulse2Pulse is designed for:

1. **Synthetic ECG Generation**: Create realistic ECG samples for data augmentation
2. **Privacy-Preserving Research**: Generate synthetic data instead of sharing real patient data
3. **Rare Condition Modeling**: Generate samples of rare ECG patterns
4. **Algorithm Testing**: Create controlled test cases for ECG analysis algorithms
5. **Educational Tools**: Generate example ECGs for teaching

## Limitations

- **8-Lead Only**: Currently supports 8-lead ECGs (can be extended)
- **Fixed Length**: 5000 samples per lead (10 seconds at 500 Hz)
- **MIMIC-IV Format**: Designed for MIMIC-IV-ECG dataset format
- **Computational Cost**: Large discriminator requires significant GPU memory

## Comparison with Original

### Improvements

✅ PyTorch Lightning integration  
✅ Better validation and checkpointing  
✅ Improved visualization  
✅ Modular, reusable code  
✅ Type hints and documentation  

### Preserved Features

✅ Same model architecture  
✅ Same training algorithm (WGAN-GP)  
✅ Same hyperparameters  
✅ Compatible with MIMIC-IV-ECG  

## Next Steps

- [Training Guide](training.md) - Learn how to train Pulse2Pulse
- [Configuration](configuration.md) - Customize your experiments
- [API Reference](../../reference/) - Explore the code

## References

1. **WaveGAN Paper**: Donahue et al., "Adversarial Audio Synthesis" (2019)
2. **WGAN-GP Paper**: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
3. **MIMIC-IV-ECG**: Goldberger et al., PhysioNet (2000)
4. **Original Implementation**: Based on pulse2pulse_simple.py
