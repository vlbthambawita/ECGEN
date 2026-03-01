# Shell Scripts - Fixed and Ready to Use

All shell scripts in `scripts/shell/` have been updated to work with the new repository structure.

## Fixed Issues

1. ✅ **Path navigation**: Scripts now correctly navigate to ECGEN root (`cd "$(dirname "$0")/../.."`)
2. ✅ **Script paths**: Updated to use `scripts/train/` instead of `scripts/`
3. ✅ **Config paths**: Correctly reference `configs/experiments/`

## Available Scripts

### 1. Train VAE with Config

```bash
./scripts/shell/run_train_vae_mimic_config.sh
```

Or with custom config:
```bash
./scripts/shell/run_train_vae_mimic_config.sh configs/experiments/my_vae_config.yaml
```

**What it does:**
- Loads config from `configs/experiments/vae_mimic.yaml`
- Runs `scripts/train/train_vae_mimic.py`
- Saves outputs to `outputs/vae_mimic/`

### 2. Train Pulse2Pulse with Config

```bash
./scripts/shell/run_train_pulse2pulse_mimic.sh
```

Or with custom config:
```bash
./scripts/shell/run_train_pulse2pulse_mimic.sh configs/experiments/my_pulse2pulse_config.yaml
```

**What it does:**
- Loads config from `configs/experiments/pulse2pulse_mimic.yaml`
- Runs `scripts/train/train_pulse2pulse.py`
- Saves outputs to `outputs/pulse2pulse_mimic/`

### 3. Train Pulse2Pulse Standalone (No Config)

```bash
./scripts/shell/run_train_pulse2pulse_standalone.sh
```

With environment variables:
```bash
DATA_DIR=/path/to/data \
BATCH_SIZE=64 \
MAX_EPOCHS=100 \
./scripts/shell/run_train_pulse2pulse_standalone.sh
```

**Environment Variables:**
- `DATA_DIR`: Path to MIMIC-IV-ECG data
- `EXP_NAME`: Experiment name (default: pulse2pulse_mimic)
- `BATCH_SIZE`: Batch size (default: 128)
- `MAX_EPOCHS`: Maximum epochs (default: 300)
- `GPU_ID`: GPU device ID (default: 0)
- `NUM_WORKERS`: Data loader workers (default: 4)

### 4. Run Training Sweep

```bash
./scripts/shell/run_sweep.sh
```

**What it does:**
- Trains all configs in `configs/experiments/`
- Automatically detects VAE vs Pulse2Pulse configs
- Runs appropriate training script for each

## Usage Examples

### Quick Start - Train VAE

```bash
cd /work/vajira/DL2026/ECGEN
./scripts/shell/run_train_vae_mimic_config.sh
```

### Train with Custom Settings

Edit the config first:
```bash
nano configs/experiments/vae_mimic.yaml
```

Then train:
```bash
./scripts/shell/run_train_vae_mimic_config.sh
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/vae_mimic/

# Check outputs
ls -la outputs/vae_mimic/seed_42/
```

## Troubleshooting

### "Config file not found"

Make sure you're running from ECGEN root or the script will handle it:
```bash
cd /work/vajira/DL2026/ECGEN
./scripts/shell/run_train_vae_mimic_config.sh
```

### "Python module not found"

Install the package:
```bash
pip install -e .
```

Or run the fix script:
```bash
./fix_installation.sh
```

### "Data directory not found"

Update the data path in your config:
```yaml
# configs/experiments/vae_mimic.yaml
dataset:
  data_dir: /path/to/your/MIMIC-IV-ECG
```

## Script Details

All scripts now:
- ✅ Navigate to ECGEN root automatically
- ✅ Use correct paths for new structure
- ✅ Show clear progress messages
- ✅ Validate inputs before running
- ✅ Display working directory and Python path

## Next Steps

1. **Test a script**: `./scripts/shell/run_train_vae_mimic_config.sh --help` (if supported)
2. **Check config**: `cat configs/experiments/vae_mimic.yaml`
3. **Start training**: Run one of the scripts above
4. **Monitor**: Use TensorBoard or check `outputs/` directory

All scripts are ready to use! 🚀
