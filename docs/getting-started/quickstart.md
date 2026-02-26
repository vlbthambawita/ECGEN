# Quick Start

Get started with ECGEN in just a few minutes! This guide will walk you through training your first Pulse2Pulse model.

## Prerequisites

Make sure you have:

- ✅ Installed ECGEN (see [Installation Guide](installation.md))
- ✅ Access to MIMIC-IV-ECG dataset
- ✅ GPU with CUDA support (recommended)

## 3-Step Quick Start

### Step 1: Verify Installation

Test that everything is set up correctly:

```bash
cd /work/vajira/DL2026/ECGEN
python scripts/test_models_only.py
```

Expected output:
```
✓ ALL TESTS PASSED!
```

### Step 2: Choose Your Training Method

=== "Config-Based (Recommended)"

    Use a YAML configuration file for reproducible experiments:

    ```bash
    ./scripts/run_train_pulse2pulse_mimic.sh
    ```

    This uses the config at `configs/experiments/pulse2pulse_mimic.yaml`.

=== "Standalone Script"

    Direct Python script with command-line arguments:

    ```bash
    ./scripts/run_train_pulse2pulse_standalone.sh
    ```

=== "Direct Python"

    Maximum flexibility with direct Python invocation:

    ```bash
    python scripts/train_pulse2pulse.py \
        --data-dir /path/to/MIMIC-IV-ECG \
        --batch-size 128 \
        --max-epochs 300 \
        --accelerator gpu \
        --devices 0
    ```

### Step 3: Monitor Training

#### TensorBoard

```bash
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb
```

Open http://localhost:6006 in your browser.

#### Weights & Biases (Optional)

```bash
# Install and login
pip install wandb
wandb login

# Add --wandb flag
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse
```

## Output Structure

All outputs are saved to `runs/pulse2pulse_mimic/seed_42/`:

```
runs/pulse2pulse_mimic/seed_42/
├── checkpoints/          # Model checkpoints
│   ├── last.ckpt        # Latest checkpoint
│   └── epoch*.ckpt      # Top-K checkpoints
├── samples/              # Generated ECG samples
│   ├── sample_epoch_*.png
│   └── generated_epoch_*.png
├── tb/                   # TensorBoard logs
├── config_resolved.yaml  # Full configuration
└── metadata.json         # Run metadata
```

## Quick Customization

### Change Batch Size

```bash
# Environment variable
BATCH_SIZE=64 ./scripts/run_train_pulse2pulse_standalone.sh

# Or command line
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --batch-size 64
```

### Use Different GPU

```bash
# Environment variable
GPU_ID=1 ./scripts/run_train_pulse2pulse_standalone.sh

# Or command line
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --devices 1
```

### Quick Test Run

Train on a small subset for testing:

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --max-samples 1000 \
    --max-epochs 10 \
    --batch-size 32
```

Expected time: ~5-10 minutes

## Generate Samples

After training, generate new ECG samples:

```bash
python scripts/generate_pulse2pulse.py \
    --checkpoint runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt \
    --n-samples 16 \
    --output-dir generated_samples
```

## Resume Training

Continue from a checkpoint:

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --resume runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt
```

## What You Get

After training, you'll have:

- ✅ **Trained Model**: WaveGAN generator + discriminator (~344M params)
- ✅ **Checkpoints**: Automatic saving of best models
- ✅ **Samples**: Generated ECG visualizations
- ✅ **Metrics**: Training and validation metrics
- ✅ **Logs**: TensorBoard logs for analysis

## Training Time Estimates

| Configuration | Time (Single GPU) |
|--------------|-------------------|
| Quick test (10 epochs, 1K samples) | ~5-10 minutes |
| Medium run (100 epochs, 10K samples) | ~4-8 hours |
| Full training (300 epochs, full dataset) | ~24-48 hours |

*Times based on NVIDIA V100 GPU*

## Common Issues

### Out of Memory

**Solution**: Reduce batch size
```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --batch-size 32  # or 64
```

### Data Not Found

**Solution**: Update data directory path
```bash
# Check your data path
ls /path/to/MIMIC-IV-ECG/files

# Update in script or config
python scripts/train_pulse2pulse.py \
    --data-dir /correct/path/to/MIMIC-IV-ECG
```

### Wrong GPU Selected

**Solution**: Specify GPU device
```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --devices 0  # or 1, 2, etc.
```

### Import Errors

**Solution**: Upgrade dependencies
```bash
pip install --upgrade jinja2
pip install -e .
```

## Next Steps

Now that you've trained your first model:

- 📖 [Pulse2Pulse Training Guide](../user-guide/pulse2pulse/training.md) - Detailed training instructions
- ⚙️ [Configuration Guide](../user-guide/pulse2pulse/configuration.md) - Customize your experiments
- 📊 [W&B Integration](../user-guide/wandb.md) - Advanced experiment tracking
- 🔍 [API Reference](../reference/index.md) - Explore the codebase

## Tips for Success

1. **Start Small**: Test with `--max-samples 1000` first
2. **Monitor Early**: Check samples after 10-25 epochs
3. **Save Often**: Default checkpointing is automatic
4. **Track Experiments**: Use W&B for easy comparison
5. **Read Logs**: TensorBoard shows training dynamics

---

**Questions?** Check the [User Guide](../user-guide/pulse2pulse/overview.md) or [API Reference](../reference/).
