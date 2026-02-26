# Configuration Guide

This guide explains how to configure Pulse2Pulse training experiments using YAML configuration files.

## Configuration File Structure

ECGEN uses YAML files for experiment configuration. Here's the structure:

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      # Model hyperparameters

data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      # Data loading parameters

trainer:
  # PyTorch Lightning trainer parameters

wandb:
  # Weights & Biases configuration (optional)
```

## Model Configuration

### Basic Parameters

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      model_size: 50          # Base channel multiplier
      num_channels: 8         # Number of ECG leads
      seq_length: 5000        # Samples per lead (10s at 500Hz)
      lr: 0.0001             # Learning rate
      b1: 0.5                # Adam beta1
      b2: 0.9                # Adam beta2
      lmbda: 10.0            # Gradient penalty coefficient
      n_critic: 5            # Discriminator updates per generator update
```

### Parameter Details

#### `model_size`
- **Type**: Integer
- **Default**: 50
- **Description**: Base channel multiplier for all layers
- **Effect**: Higher values = more parameters = better quality but slower training
- **Typical Range**: 25-100

#### `num_channels`
- **Type**: Integer
- **Default**: 8
- **Description**: Number of ECG leads
- **Options**: 8 (for 8-lead) or 12 (for 12-lead, requires code changes)

#### `seq_length`
- **Type**: Integer
- **Default**: 5000
- **Description**: Number of samples per lead
- **Note**: 5000 samples = 10 seconds at 500 Hz sampling rate

#### `lr` (Learning Rate)
- **Type**: Float
- **Default**: 0.0001
- **Description**: Learning rate for both generator and discriminator
- **Typical Range**: 1e-5 to 5e-4
- **Tips**: 
  - Start with 1e-4
  - Reduce if training is unstable
  - Increase if training is too slow

#### `b1`, `b2` (Adam Betas)
- **Type**: Float
- **Default**: 0.5, 0.9
- **Description**: Adam optimizer momentum parameters
- **Note**: β₁=0.5 is common for GANs (lower than typical 0.9)

#### `lmbda` (Gradient Penalty)
- **Type**: Float
- **Default**: 10.0
- **Description**: Coefficient for WGAN-GP gradient penalty
- **Typical Range**: 5.0-20.0
- **Effect**: Higher values = stronger Lipschitz constraint

#### `n_critic`
- **Type**: Integer
- **Default**: 5
- **Description**: Number of discriminator updates per generator update
- **Typical Range**: 3-10
- **Effect**: Higher values = more stable training but slower

## Data Configuration

```yaml
data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      data_dir: /path/to/MIMIC-IV-ECG
      batch_size: 128
      num_workers: 4
      max_samples: null
      skip_missing_check: true
      num_channels: 8
      seq_length: 5000
```

### Parameter Details

#### `data_dir`
- **Type**: String (path)
- **Required**: Yes
- **Description**: Path to MIMIC-IV-ECG dataset root directory
- **Example**: `/work/vajira/DATA/MIMIC_IV_ECG/mimic-iv-ecg-1.0`

#### `batch_size`
- **Type**: Integer
- **Default**: 128
- **Description**: Number of samples per batch
- **Typical Range**: 32-256
- **Memory Impact**: Higher = more GPU memory required
- **Recommendations**:
  - 16 GB GPU: 32-64
  - 24 GB GPU: 64-128
  - 32+ GB GPU: 128-256

#### `num_workers`
- **Type**: Integer
- **Default**: 4
- **Description**: Number of parallel data loading processes
- **Typical Range**: 0-8
- **Tips**:
  - 0 = single-threaded loading
  - 4-8 = good for most systems
  - Too high can cause memory issues

#### `max_samples`
- **Type**: Integer or null
- **Default**: null (use all data)
- **Description**: Limit dataset size for debugging
- **Use Cases**:
  - Quick testing: 1000
  - Medium testing: 10000
  - Full training: null

#### `skip_missing_check`
- **Type**: Boolean
- **Default**: true
- **Description**: Skip checking if all files exist (faster startup)
- **Note**: Set to false if you suspect data issues

## Trainer Configuration

```yaml
trainer:
  max_epochs: 300
  accelerator: "gpu"
  devices: [0]
  precision: 32
  log_every_n_steps: 50
  check_val_every_n_epoch: 5
  enable_checkpointing: true
  enable_progress_bar: true
  enable_model_summary: true
```

### Parameter Details

#### `max_epochs`
- **Type**: Integer
- **Default**: 300
- **Description**: Maximum number of training epochs
- **Recommendations**:
  - Quick test: 10
  - Medium run: 100
  - Full training: 300-500

#### `accelerator`
- **Type**: String
- **Options**: "gpu", "cpu", "tpu"
- **Default**: "gpu"
- **Note**: CPU training is very slow, not recommended

#### `devices`
- **Type**: List of integers
- **Default**: [0]
- **Description**: GPU device IDs to use
- **Examples**:
  - Single GPU: [0]
  - Multi-GPU: [0, 1, 2, 3]
  - Specific GPU: [2]

#### `precision`
- **Type**: Integer
- **Options**: 32, 16, "bf16"
- **Default**: 32
- **Description**: Floating point precision
- **Note**: 16-bit can be faster but may affect quality

#### `log_every_n_steps`
- **Type**: Integer
- **Default**: 50
- **Description**: Log metrics every N training steps
- **Effect**: Lower = more frequent logging = larger log files

#### `check_val_every_n_epoch`
- **Type**: Integer
- **Default**: 5
- **Description**: Run validation every N epochs
- **Effect**: Lower = more frequent validation = slower training

## Weights & Biases Configuration

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
  entity: your-username
  run_name: null
  tags:
    - pulse2pulse
    - wavegan
    - ecg-generation
  notes: "Experiment description"
  log_model: true
  log_freq: 50
```

### Parameter Details

#### `enabled`
- **Type**: Boolean
- **Default**: false
- **Description**: Enable W&B logging

#### `project`
- **Type**: String
- **Required**: Yes (if enabled)
- **Description**: W&B project name

#### `entity`
- **Type**: String
- **Optional**: Yes
- **Description**: W&B username or team name

#### `tags`
- **Type**: List of strings
- **Description**: Tags for organizing experiments

#### `log_model`
- **Type**: Boolean
- **Default**: true
- **Description**: Upload model checkpoints to W&B

## Example Configurations

### Quick Test Configuration

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      model_size: 25          # Smaller model
      lr: 0.0002             # Higher learning rate

data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      data_dir: /path/to/MIMIC-IV-ECG
      batch_size: 32
      max_samples: 1000      # Small dataset

trainer:
  max_epochs: 10             # Few epochs
  devices: [0]
```

### High-Quality Configuration

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      model_size: 75          # Larger model
      lr: 0.00005            # Lower learning rate
      n_critic: 10           # More discriminator updates

data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      data_dir: /path/to/MIMIC-IV-ECG
      batch_size: 64
      max_samples: null      # Full dataset

trainer:
  max_epochs: 500            # More epochs
  devices: [0, 1]            # Multi-GPU
```

### Memory-Constrained Configuration

```yaml
model:
  target: ecgen.models.pulse2pulse.Pulse2PulseGAN
  params:
    config:
      model_size: 35          # Smaller model

data:
  target: ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule
  params:
    config:
      data_dir: /path/to/MIMIC-IV-ECG
      batch_size: 16          # Small batch size
      num_workers: 2

trainer:
  max_epochs: 300
  devices: [0]
  precision: 16              # Half precision
```

## Command-Line Overrides

You can override config values from the command line:

```bash
python -m ecgen.training.train \
    --config configs/experiments/pulse2pulse_mimic.yaml \
    --override "data.params.config.batch_size=64" \
    --override "trainer.max_epochs=100"
```

## Environment Variables

Some parameters can be set via environment variables:

```bash
# Set data directory
export DATA_DIR=/path/to/MIMIC-IV-ECG

# Set batch size
export BATCH_SIZE=64

# Set GPU device
export GPU_ID=1

# Run training
./scripts/run_train_pulse2pulse_mimic.sh
```

## Best Practices

1. **Start with defaults**: Use the provided config as a baseline
2. **Test small first**: Use `max_samples=1000` and `max_epochs=10` for testing
3. **Monitor early**: Check samples after 10-25 epochs
4. **Adjust gradually**: Change one parameter at a time
5. **Save configs**: Keep a copy of successful configurations
6. **Use version control**: Track config changes with git
7. **Document experiments**: Use W&B tags and notes

## Troubleshooting

### Training is unstable

Try:
- Reduce learning rate: `lr: 0.00005`
- Increase gradient penalty: `lmbda: 15.0`
- Increase discriminator updates: `n_critic: 10`

### Out of memory

Try:
- Reduce batch size: `batch_size: 32`
- Reduce model size: `model_size: 35`
- Use half precision: `precision: 16`

### Training is too slow

Try:
- Increase batch size: `batch_size: 256`
- Use multiple GPUs: `devices: [0, 1]`
- Reduce validation frequency: `check_val_every_n_epoch: 10`

## Next Steps

- [Training Guide](training.md) - Learn training best practices
- [W&B Integration](../wandb.md) - Set up experiment tracking
- [API Reference](../../reference/) - Explore the codebase
