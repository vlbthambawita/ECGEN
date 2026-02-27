# VAE Training with Config File - Fixed!

## Problem Solved ✅

The `--data-dir` is now **optional** when using `--config`. The script will load the data path from the config file.

## Usage

### Method 1: Config File Only (Recommended)

```bash
# Use default config
bash scripts/run_train_vae_mimic_config.sh

# Use custom config
bash scripts/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
```

**Or directly with Python:**
```bash
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml
```

### Method 2: Command-Line Arguments Only

```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name my_vae \
    --batch-size 32 \
    --max-epochs 100
```

### Method 3: Config + Override

```bash
# Load config but override specific settings
python scripts/train_vae_mimic.py \
    --config configs/experiments/vae_mimic.yaml \
    --max-epochs 200 \
    --batch-size 64
```

## Config File Structure

**File**: `configs/experiments/vae_mimic.yaml`

```yaml
experiment:
  name: vae_mimic
  seed: 42
  runs_root: runs

model:
  params:
    config:
      in_channels: 12
      base_channels: 64
      latent_channels: 8
      lr: 0.0001
      kl_weight: 0.0001

data:
  params:
    mimic_path: /path/to/mimic-iv-ecg  # ← Data path here!
    batch_size: 32
    max_samples: 10000

trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: [0]
```

## Quick Test

```bash
# Test that config loads correctly
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml --help
```

Should show help without requiring `--data-dir`!

## Error Messages

### If neither config nor data-dir provided:
```
Error: Either --config or --data-dir must be provided
```

**Solution**: Provide one of them:
```bash
# Option 1: Use config
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml

# Option 2: Use data-dir
python scripts/train_vae_mimic.py --data-dir /path/to/data
```

## Shell Script Comparison

| Script | Uses Config? | Command |
|--------|--------------|---------|
| `run_train_vae_mimic_config.sh` | ✅ YES | `bash scripts/run_train_vae_mimic_config.sh` |
| `run_train_vae_mimic.sh` | ❌ No | `bash scripts/run_train_vae_mimic.sh /path/to/data` |
| `run_train_vae_mimic_quick.sh` | ❌ No | `bash scripts/run_train_vae_mimic_quick.sh /path/to/data` |

## Creating Custom Configs

### Step 1: Copy template
```bash
cp configs/experiments/vae_mimic.yaml configs/experiments/my_experiment.yaml
```

### Step 2: Edit your config
```bash
vim configs/experiments/my_experiment.yaml
```

Change settings like:
- `experiment.name` - Your experiment name
- `data.params.mimic_path` - Your data path
- `data.params.max_samples` - Number of samples
- `model.params.config.base_channels` - Model size
- `trainer.max_epochs` - Training duration

### Step 3: Run with your config
```bash
bash scripts/run_train_vae_mimic_config.sh configs/experiments/my_experiment.yaml
```

## Full Example

```bash
# 1. Create custom config
cat > configs/experiments/vae_large.yaml << 'EOF'
experiment:
  name: vae_large
  seed: 42

model:
  params:
    config:
      in_channels: 12
      base_channels: 128  # Larger model
      latent_channels: 16  # More latent dims
      lr: 0.00005

data:
  params:
    mimic_path: /work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0
    batch_size: 16  # Smaller batch for large model
    max_samples: null  # Use full dataset

trainer:
  max_epochs: 200
  accelerator: "gpu"
  devices: [0]
EOF

# 2. Run training
bash scripts/run_train_vae_mimic_config.sh configs/experiments/vae_large.yaml
```

## Summary

✅ **Fixed**: `--data-dir` is now optional when using `--config`  
✅ **Config file**: Loads all settings from YAML  
✅ **Override**: Can override config settings with command-line args  
✅ **Same pattern**: Works exactly like Pulse2Pulse training  

**Recommended workflow**: Use config files for reproducible experiments!
