# Weights & Biases Integration Guide

## Overview

The standalone VQ-VAE training script now supports **Weights & Biases (wandb)** for experiment tracking, in addition to TensorBoard.

## Installation

```bash
pip install wandb
```

## Setup

### 1. Login to wandb (first time only)

```bash
wandb login
```

This will prompt you to enter your API key from https://wandb.ai/authorize

### 2. Set your wandb entity (optional)

```bash
export WANDB_ENTITY=your-username-or-team
```

## Usage

### Using Shell Script

Enable wandb by setting the `WANDB_ENABLED` environment variable:

```bash
# Basic usage
WANDB_ENABLED=true ./run_train_vqvae.sh both

# With custom project name
WANDB_ENABLED=true \
WANDB_PROJECT=my-ecg-project \
./run_train_vqvae.sh both

# Full configuration
WANDB_ENABLED=true \
WANDB_PROJECT=ecg-vqvae-experiments \
WANDB_ENTITY=my-team \
WANDB_RUN_NAME=vqvae_large_model \
WANDB_TAGS="vqvae ecg production" \
./run_train_vqvae.sh both
```

### Using Python Directly

```bash
# Stage 1 with wandb
python train_vqvae_standalone.py \
    --stage 1 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name vqvae_exp \
    --wandb \
    --wandb-project ecg-vqvae \
    --wandb-entity your-username \
    --wandb-run-name vqvae_test_run \
    --wandb-tags vqvae ecg test

# Stage 2 with wandb
python train_vqvae_standalone.py \
    --stage 2 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name prior_exp \
    --vqvae-checkpoint runs/vqvae_exp/seed_42/checkpoints/last.ckpt \
    --wandb \
    --wandb-project ecg-vqvae \
    --wandb-entity your-username
```

## Configuration Options

### Shell Script Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_ENABLED` | `false` | Enable/disable wandb logging |
| `WANDB_PROJECT` | `ecg-vqvae` | Wandb project name |
| `WANDB_ENTITY` | `` | Wandb entity (username/team) |
| `WANDB_RUN_NAME` | `` | Custom run name (auto-generated if empty) |
| `WANDB_TAGS` | `` | Space-separated tags |

### Python Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb` | `False` | Enable wandb logging (flag) |
| `--wandb-project` | `ecg-vqvae` | Wandb project name |
| `--wandb-entity` | `None` | Wandb entity (username/team) |
| `--wandb-run-name` | `None` | Custom run name |
| `--wandb-tags` | `None` | List of tags |

## What Gets Logged

### Stage 1 (VQ-VAE)

**Metrics:**
- `train/total_loss` - Total training loss
- `train/recon_loss` - Reconstruction loss (MSE)
- `train/vq_loss` - Vector quantization loss
- `train/unique_codes` - Number of unique codes used
- `train/codebook_usage` - Fraction of codebook used
- `val/total_loss` - Validation total loss
- `val/recon_loss` - Validation reconstruction loss
- `val/vq_loss` - Validation VQ loss
- `val/unique_codes` - Validation unique codes
- `val/codebook_usage` - Validation codebook usage

**Config:**
- Model architecture (channels, layers, etc.)
- Training hyperparameters (lr, batch size, etc.)
- Dataset settings
- Random seed

**Tags (auto-added):**
- `vqvae`
- `ecg`
- `stage1`

### Stage 2 (Prior)

**Metrics:**
- `train/loss` - Cross-entropy loss for autoregressive prediction
- `val/loss` - Validation cross-entropy loss

**Config:**
- Prior model architecture
- Training hyperparameters
- VQ-VAE checkpoint path
- Random seed

**Tags (auto-added):**
- `vqvae`
- `prior`
- `pixelcnn`
- `stage2`

## Viewing Results

### Wandb Dashboard

After starting training with wandb enabled, you'll see:

```
✓ Weights & Biases logging enabled: ecg-vqvae/vqvae_mimic_standalone_seed42
```

Visit your wandb dashboard at:
```
https://wandb.ai/your-username/ecg-vqvae
```

### TensorBoard (Always Available)

TensorBoard logging is always enabled regardless of wandb:

```bash
tensorboard --logdir=runs
```

## Examples

### Quick Test with Wandb

```bash
WANDB_ENABLED=true \
MAX_SAMPLES=100 \
MAX_EPOCHS_STAGE1=10 \
MAX_EPOCHS_STAGE2=10 \
./run_train_vqvae.sh both
```

### Production Training with Wandb

```bash
WANDB_ENABLED=true \
WANDB_PROJECT=ecg-production \
WANDB_ENTITY=my-team \
WANDB_RUN_NAME=vqvae_full_dataset \
WANDB_TAGS="production full-dataset v1" \
MAX_SAMPLES=null \
MAX_EPOCHS_STAGE1=200 \
MAX_EPOCHS_STAGE2=200 \
./run_train_vqvae.sh both
```

### Hyperparameter Sweep

```bash
# Run multiple experiments with different settings
for NUM_EMB in 256 512 1024; do
    WANDB_ENABLED=true \
    WANDB_RUN_NAME="vqvae_emb${NUM_EMB}" \
    WANDB_TAGS="sweep codebook-size" \
    NUM_EMBEDDINGS=$NUM_EMB \
    ./run_train_vqvae.sh 1
done
```

## Troubleshooting

### Issue: "wandb not installed"

**Solution:**
```bash
pip install wandb
```

### Issue: "wandb login required"

**Solution:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Issue: Wandb fails but training continues

This is expected behavior! If wandb fails to initialize, the script will:
1. Print a warning message
2. Continue training with TensorBoard only
3. Not crash your training run

Example output:
```
⚠ Warning: Failed to initialize wandb: <error message>
  Continuing with TensorBoard only
```

### Issue: Want to disable wandb temporarily

**Solution:**
```bash
# Just don't set WANDB_ENABLED or set it to false
WANDB_ENABLED=false ./run_train_vqvae.sh both

# Or omit --wandb flag when using Python directly
python train_vqvae_standalone.py --stage 1 --data-dir /path/to/data
```

## Advanced Features

### Custom Tags

Add custom tags to organize your experiments:

```bash
WANDB_ENABLED=true \
WANDB_TAGS="experiment-v2 large-model gpu-a100" \
./run_train_vqvae.sh both
```

### Resume Runs

Wandb automatically handles run resumption when you resume training from a checkpoint.

### Offline Mode

Train without internet connection and sync later:

```bash
export WANDB_MODE=offline
WANDB_ENABLED=true ./run_train_vqvae.sh both

# Later, sync the results
wandb sync runs/vqvae_mimic_standalone/seed_42/wandb/
```

## Comparison: TensorBoard vs Wandb

| Feature | TensorBoard | Wandb |
|---------|-------------|-------|
| Installation | Included with PyTorch Lightning | Requires `pip install wandb` |
| Setup | None | Login required |
| Local viewing | ✓ | ✓ (via web) |
| Cloud storage | ✗ | ✓ |
| Collaboration | Limited | ✓ |
| Experiment comparison | Basic | Advanced |
| Model versioning | ✗ | ✓ |
| Hyperparameter tracking | ✓ | ✓ |
| Always available | ✓ | Only if installed |

## Best Practices

1. **Use both**: TensorBoard for quick local checks, Wandb for long-term tracking
2. **Consistent naming**: Use descriptive run names and tags
3. **Tag everything**: Makes filtering and searching easier later
4. **Project organization**: Use separate projects for different experiments
5. **Document configs**: Wandb automatically logs all hyperparameters

## Summary

Wandb integration is:
- ✅ **Optional** - Training works without it
- ✅ **Easy to enable** - Just set `WANDB_ENABLED=true`
- ✅ **Graceful fallback** - Continues with TensorBoard if wandb fails
- ✅ **Fully configured** - Logs all metrics and hyperparameters
- ✅ **Production ready** - Handles errors without crashing

Enjoy tracking your experiments! 🚀
