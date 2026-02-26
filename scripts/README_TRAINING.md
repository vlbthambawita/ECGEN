# Training Scripts Guide

This directory contains scripts for training the Pulse2Pulse model.

## Quick Start

### Option 1: Config-Based Training (Recommended)

Uses the YAML configuration file for all settings:

```bash
cd /work/vajira/DL2026/ECGEN
./scripts/run_train_pulse2pulse_mimic.sh
```

Or with a custom config:

```bash
./scripts/run_train_pulse2pulse_mimic.sh configs/experiments/my_custom_config.yaml
```

### Option 2: Standalone Training (No Config File)

Uses command-line arguments and environment variables:

```bash
cd /work/vajira/DL2026/ECGEN
./scripts/run_train_pulse2pulse_standalone.sh
```

Customize with environment variables:

```bash
DATA_DIR=/path/to/MIMIC-IV-ECG \
BATCH_SIZE=64 \
MAX_EPOCHS=500 \
GPU_ID=1 \
./scripts/run_train_pulse2pulse_standalone.sh
```

## Available Scripts

### 1. `run_train_pulse2pulse_mimic.sh`

**Purpose**: Train using YAML configuration  
**Usage**: `./scripts/run_train_pulse2pulse_mimic.sh [config_path]`  
**Default Config**: `configs/experiments/pulse2pulse_mimic.yaml`

**Features**:
- Uses YAML config for all settings
- Easy to version control experiments
- Supports callbacks and advanced features
- Best for reproducible experiments

**Example**:
```bash
# Use default config
./scripts/run_train_pulse2pulse_mimic.sh

# Use custom config
./scripts/run_train_pulse2pulse_mimic.sh configs/experiments/pulse2pulse_custom.yaml
```

### 2. `run_train_pulse2pulse_standalone.sh`

**Purpose**: Train without config file using environment variables  
**Usage**: `./scripts/run_train_pulse2pulse_standalone.sh`

**Environment Variables**:
- `DATA_DIR`: Path to MIMIC-IV-ECG dataset (required)
- `EXP_NAME`: Experiment name (default: pulse2pulse_mimic)
- `BATCH_SIZE`: Batch size (default: 128)
- `MAX_EPOCHS`: Maximum epochs (default: 300)
- `GPU_ID`: GPU device ID (default: 0)
- `NUM_WORKERS`: Data loading workers (default: 4)

**Features**:
- No config file needed
- Quick experimentation
- Easy to customize via environment variables
- Good for quick tests

**Examples**:
```bash
# Use defaults
./scripts/run_train_pulse2pulse_standalone.sh

# Custom batch size and epochs
BATCH_SIZE=64 MAX_EPOCHS=500 ./scripts/run_train_pulse2pulse_standalone.sh

# Different GPU
GPU_ID=1 ./scripts/run_train_pulse2pulse_standalone.sh

# Custom data directory
DATA_DIR=/path/to/data ./scripts/run_train_pulse2pulse_standalone.sh
```

### 3. `train_pulse2pulse.py`

**Purpose**: Direct Python training script  
**Usage**: `python scripts/train_pulse2pulse.py [args]`

**Common Arguments**:
```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --exp-name my_experiment \
    --batch-size 128 \
    --max-epochs 300 \
    --accelerator gpu \
    --devices 0 \
    --lr 1e-4 \
    --model-size 50 \
    --num-channels 8 \
    --seq-length 5000
```

**All Arguments**:
- `--data-dir`: Path to MIMIC-IV-ECG dataset (required)
- `--exp-name`: Experiment name
- `--seed`: Random seed (default: 42)
- `--batch-size`: Batch size (default: 128)
- `--num-workers`: Data loading workers (default: 4)
- `--max-samples`: Max samples for debugging (default: None)
- `--skip-missing-check`: Skip file existence check
- `--model-size`: Model size parameter (default: 50)
- `--num-channels`: Number of ECG leads (default: 8)
- `--seq-length`: ECG sequence length (default: 5000)
- `--lr`: Learning rate (default: 1e-4)
- `--b1`: Adam beta1 (default: 0.5)
- `--b2`: Adam beta2 (default: 0.9)
- `--lmbda`: Gradient penalty coefficient (default: 10.0)
- `--n-critic`: Discriminator updates per generator update (default: 5)
- `--max-epochs`: Maximum epochs (default: 300)
- `--accelerator`: Accelerator type (default: gpu)
- `--devices`: GPU device IDs (default: [0])
- `--log-every-n-steps`: Logging frequency (default: 50)
- `--check-val-every-n-epoch`: Validation frequency (default: 5)
- `--viz-every-n-epochs`: Visualization frequency (default: 10)
- `--save-samples-every-n-epochs`: Sample saving frequency (default: 25)
- `--resume`: Path to checkpoint to resume from

### 4. `generate_pulse2pulse.py`

**Purpose**: Generate ECG samples from trained model  
**Usage**: `python scripts/generate_pulse2pulse.py [args]`

**Example**:
```bash
python scripts/generate_pulse2pulse.py \
    --checkpoint runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt \
    --output-dir generated_samples \
    --n-samples 16 \
    --device cuda:0
```

### 5. `test_models_only.py`

**Purpose**: Test model architecture without training  
**Usage**: `python scripts/test_models_only.py`

**Example**:
```bash
python scripts/test_models_only.py
```

## Output Structure

All training runs save outputs to:

```
runs/<experiment_name>/seed_<N>/
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
│   └── version_0/
├── config_resolved.yaml             # Resolved configuration
└── metadata.json                    # Run metadata
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb
```

Then open http://localhost:6006 in your browser.

### Check Samples

Generated samples are saved periodically:

```bash
# View comparison plots
ls runs/pulse2pulse_mimic/seed_42/samples/sample_epoch_*.png

# View generated samples
ls runs/pulse2pulse_mimic/seed_42/samples/generated_epoch_*.png
```

### Check Checkpoints

```bash
ls runs/pulse2pulse_mimic/seed_42/checkpoints/
```

## Resume Training

### From Config Script

```bash
# Edit config to add checkpoint path, or use --resume flag
python -m ecgen.training.train \
    --config configs/experiments/pulse2pulse_mimic.yaml \
    --resume runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt
```

### From Standalone Script

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --resume runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt
```

## Common Issues

### Issue: Data directory not found

**Error**: `FileNotFoundError: machine_measurements.csv not found`

**Solution**: Update the data directory path in the config or script:

```bash
# For standalone script
DATA_DIR=/correct/path/to/MIMIC-IV-ECG ./scripts/run_train_pulse2pulse_standalone.sh

# For config-based
# Edit configs/experiments/pulse2pulse_mimic.yaml
# Update data.params.config.data_dir
```

### Issue: Out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size:

```bash
# For standalone script
BATCH_SIZE=64 ./scripts/run_train_pulse2pulse_standalone.sh

# For direct Python script
python scripts/train_pulse2pulse.py --data-dir /path/to/data --batch-size 64
```

### Issue: Wrong GPU

**Solution**: Specify GPU device:

```bash
# For standalone script
GPU_ID=1 ./scripts/run_train_pulse2pulse_standalone.sh

# For direct Python script
python scripts/train_pulse2pulse.py --data-dir /path/to/data --devices 1
```

## Tips

1. **Start with small dataset**: Use `--max-samples 1000` for quick testing
2. **Monitor early**: Check samples after 10-20 epochs
3. **Use TensorBoard**: Monitor training metrics in real-time
4. **Save checkpoints**: Keep top-K checkpoints for best models
5. **Validate regularly**: Use `--check-val-every-n-epoch 5` to catch issues early

## Examples

### Quick Test Run (Small Dataset)

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --exp-name test_run \
    --batch-size 32 \
    --max-epochs 10 \
    --max-samples 1000 \
    --skip-missing-check
```

### Full Training Run

```bash
./scripts/run_train_pulse2pulse_mimic.sh
```

### Custom Hyperparameters

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --exp-name custom_hp \
    --batch-size 256 \
    --max-epochs 500 \
    --lr 2e-4 \
    --model-size 100 \
    --n-critic 10
```

### Multi-GPU Training

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --accelerator gpu \
    --devices 0 1 2 3
```

## Next Steps

After training:

1. **Generate samples**: Use `generate_pulse2pulse.py`
2. **Evaluate quality**: Visual inspection and metrics
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Scale up**: Increase epochs or model size

For more details, see:
- `../README_PULSE2PULSE.md`: Main integration guide
- `../docs/pulse2pulse_training.md`: Detailed training guide
