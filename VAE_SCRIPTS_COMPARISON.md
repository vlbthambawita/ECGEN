# VAE Training Scripts Comparison

## Overview

There are now **multiple VAE training scripts** with different approaches:

## Scripts Summary

| Script | Uses Config? | Style | Best For |
|--------|--------------|-------|----------|
| `run_train_vae_mimic_config.sh` | ✅ YES | Pulse2Pulse style | Production (config-based) |
| `run_train_vae_mimic.sh` | ❌ No | Command-line args | Quick production runs |
| `run_train_vae_mimic_quick.sh` | ❌ No | Command-line args | Fast testing |
| `train_vae.sh` | ❌ No | Command-line args | Standalone training |
| `train_vae_quick.sh` | ❌ No | Command-line args | Quick tests |

## Detailed Comparison

### 1. `run_train_vae_mimic_config.sh` ✅ **USES CONFIG**

**Pattern**: Exactly like `run_train_pulse2pulse_mimic.sh`

**Usage**:
```bash
# Use default config
bash scripts/run_train_vae_mimic_config.sh

# Use custom config
bash scripts/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
```

**Config File**: `configs/experiments/vae_mimic.yaml`

**Features**:
- ✅ Uses YAML config file
- ✅ Same pattern as Pulse2Pulse
- ✅ Easy to version control configurations
- ✅ Clean separation of code and config
- ✅ Best for production training

**Example Config** (`configs/experiments/vae_mimic.yaml`):
```yaml
experiment:
  name: vae_mimic
  seed: 42

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
    mimic_path: /path/to/mimic-iv-ecg
    batch_size: 32
    max_samples: 10000

trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: [0]
```

---

### 2. `run_train_vae_mimic.sh` ❌ No Config

**Pattern**: Direct command-line arguments

**Usage**:
```bash
bash scripts/run_train_vae_mimic.sh /path/to/data
```

**Features**:
- ❌ No config file
- ✅ Quick to run
- ✅ All settings in one script
- ✅ Good for simple runs
- ⚠️ Harder to version control settings

---

### 3. `run_train_vae_mimic_quick.sh` ❌ No Config

**Pattern**: Fast testing with hardcoded settings

**Usage**:
```bash
bash scripts/run_train_vae_mimic_quick.sh /path/to/data
```

**Features**:
- ❌ No config file
- ✅ Fast testing (1000 samples, 10 epochs)
- ✅ Smaller model for speed
- ✅ Perfect for debugging

---

### 4. Other VAE Scripts

**`train_vae.sh`** and **`train_vae_quick.sh`**:
- ❌ No config files
- Standalone scripts with full command-line control
- More verbose but flexible

## Recommendation

### Use `run_train_vae_mimic_config.sh` if:
- ✅ You want the same workflow as Pulse2Pulse
- ✅ You need to version control your configurations
- ✅ You're doing production training
- ✅ You want clean separation of code and config
- ✅ You have multiple experiment configurations

### Use `run_train_vae_mimic.sh` if:
- ✅ You want a quick one-liner
- ✅ You don't need to save configurations
- ✅ You're doing ad-hoc experiments

### Use `run_train_vae_mimic_quick.sh` if:
- ✅ You're testing the pipeline
- ✅ You want fast feedback (5 minutes)
- ✅ You're debugging

## Usage Examples

### Config-Based (Recommended for Production)

```bash
# 1. Edit config file
vim configs/experiments/vae_mimic.yaml

# 2. Run with config
bash scripts/run_train_vae_mimic_config.sh

# Or with custom config
bash scripts/run_train_vae_mimic_config.sh configs/experiments/my_custom_vae.yaml
```

### Command-Line Based (Quick Runs)

```bash
# Quick test
bash scripts/run_train_vae_mimic_quick.sh /path/to/data

# Full training
bash scripts/run_train_vae_mimic.sh /path/to/data
```

### Python Script Direct (Full Control)

```bash
# With config file
python scripts/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml

# With command-line args
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --exp-name my_vae \
    --batch-size 32 \
    --max-epochs 100
```

## Creating Custom Configs

### Step 1: Copy Template

```bash
cp configs/experiments/vae_mimic.yaml configs/experiments/my_vae.yaml
```

### Step 2: Edit Config

```bash
vim configs/experiments/my_vae.yaml
```

### Step 3: Run with Config

```bash
bash scripts/run_train_vae_mimic_config.sh configs/experiments/my_vae.yaml
```

## Config vs Command-Line Priority

When using `--config`, you can still override settings:

```bash
python scripts/train_vae_mimic.py \
    --config configs/experiments/vae_mimic.yaml \
    --max-epochs 200 \
    --batch-size 64
```

Config values are loaded first, then command-line args override them.

## File Locations

```
ECGEN/
├── scripts/
│   ├── run_train_vae_mimic_config.sh  ← Uses config ✅
│   ├── run_train_vae_mimic.sh         ← No config ❌
│   ├── run_train_vae_mimic_quick.sh   ← No config ❌
│   ├── train_vae_mimic.py             ← Supports both!
│   ├── train_vae.sh                   ← No config ❌
│   └── train_vae_quick.sh             ← No config ❌
│
└── configs/experiments/
    ├── vae_mimic.yaml                 ← VAE config file
    └── pulse2pulse_mimic.yaml         ← Pulse2Pulse config
```

## Summary

**Answer to "which vae sh file use config?"**

➡️ **`run_train_vae_mimic_config.sh`** is the ONLY shell script that uses a config file.

It follows the exact same pattern as `run_train_pulse2pulse_mimic.sh`:
- Takes config file as argument (default: `configs/experiments/vae_mimic.yaml`)
- Passes it to the Python script with `--config`
- Clean, production-ready approach

All other scripts use command-line arguments directly.
