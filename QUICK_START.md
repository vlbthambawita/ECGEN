# Quick Start Guide

Get started with ECGEN in 5 minutes!

## Step 1: Fix Installation (if needed)

If you encountered the PyYAML error:

```bash
cd /work/vajira/DL2026/ECGEN
./fix_installation.sh
```

Or manually:
```bash
pip install --ignore-installed PyYAML
pip install -e .
```

## Step 2: Verify Installation

```bash
python verify_reorganization.py
```

Expected output: `✅ All imports successful!`

## Step 3: Train Your First Model

### Option A: Train VAE (Recommended for first time)

```bash
./scripts/shell/run_train_vae_mimic_config.sh
```

This will:
- Load config from `configs/experiments/vae_mimic.yaml`
- Train VAE on MIMIC-IV-ECG dataset
- Save outputs to `outputs/vae_mimic/`

### Option B: Train Pulse2Pulse GAN

```bash
./scripts/shell/run_train_pulse2pulse_mimic.sh
```

## Step 4: Monitor Training

### TensorBoard

```bash
tensorboard --logdir outputs/vae_mimic/
```

Then open http://localhost:6006

### Check Outputs

```bash
ls -la outputs/vae_mimic/seed_42/
```

You'll see:
- `checkpoints/` - Model checkpoints
- `samples/` - Visualization samples
- `logs/` - Training logs

## Step 5: Generate Samples

After training completes:

```python
from models.vae import VAELightning

# Load trained model
model = VAELightning.load_from_checkpoint(
    "outputs/vae_mimic/seed_42/checkpoints/best.ckpt"
)

# Generate samples
samples = model.sample(n_samples=16, seq_length=5000)
print(f"Generated shape: {samples.shape}")  # (16, 12, 5000)
```

## Common Commands

### Update Config

```bash
nano configs/experiments/vae_mimic.yaml
```

### Check Structure

```bash
ls -la
```

### View Documentation

```bash
cat docs/README.md
```

### Run Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### Import Errors

```bash
# Verify installation
pip list | grep ecgen

# Reinstall if needed
pip install -e .
```

### Config Not Found

Make sure you're in ECGEN root:
```bash
cd /work/vajira/DL2026/ECGEN
pwd  # Should show .../DL2026/ECGEN
```

### Data Path Issues

Update data path in config:
```yaml
dataset:
  data_dir: /your/path/to/MIMIC-IV-ECG
```

## What's Next?

1. **Explore configs**: Check `configs/` for different model configurations
2. **Read docs**: See `docs/README.md` for comprehensive documentation
3. **Try different models**: Train GAN, or prepare for Diffusion/SSM models
4. **Customize**: Modify configs, add new datasets, create new models

## Key Files

- **Main README**: [README.md](README.md)
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Shell Scripts**: [SHELL_SCRIPTS_FIXED.md](SHELL_SCRIPTS_FIXED.md)
- **Installation**: [INSTALLATION_AND_VERIFICATION.md](INSTALLATION_AND_VERIFICATION.md)

## Getting Help

1. Check documentation in `docs/`
2. Review examples in `scripts/train/`
3. See [REORGANIZATION_COMPLETE.md](REORGANIZATION_COMPLETE.md) for what changed

Happy training! 🚀
