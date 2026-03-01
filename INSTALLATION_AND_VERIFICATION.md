# Installation and Verification Guide

## Current Issue: PyYAML Installation

You encountered a PyYAML installation conflict. This is resolved below.

## Quick Fix

Run the fix script:

```bash
cd /work/vajira/DL2026/ECGEN
./fix_installation.sh
```

This will:
1. Reinstall PyYAML properly
2. Install the ECGEN package
3. Verify all imports work

## Manual Fix (if script doesn't work)

### Option 1: Force Reinstall
```bash
pip install --ignore-installed PyYAML
pip install -e .
```

### Option 2: Use Conda
```bash
conda install pyyaml
pip install -e .
```

### Option 3: Manual Removal
```bash
# Remove old PyYAML
rm -rf ~/anaconda3/lib/python3.8/site-packages/PyYAML-5.3.1*
rm -rf ~/anaconda3/lib/python3.8/site-packages/yaml

# Install fresh
pip install -e .
```

## Verification

After installation, verify everything works:

```bash
python verify_reorganization.py
```

Expected output:
```
Testing imports...
============================================================
✓ Models - VAE imports successful
✓ Models - Components imports successful
✓ Data - MIMIC dataset import successful
✓ Training - Callbacks imports successful
✓ Training - VAE losses import successful
✓ Evaluation - Metrics imports successful
✓ Visualization - Imports successful
✓ Utils - Imports successful
============================================================

Results: 8/8 tests passed
✅ All imports successful! Reorganization verified.
```

## Testing the New Structure

### 1. Test Model Import

```python
from models.vae import VAE1D, VAEConfig

config = VAEConfig(in_channels=12, latent_channels=8)
model = VAE1D(
    in_channels=config.in_channels,
    latent_channels=config.latent_channels,
)
print("✓ VAE model created successfully")
```

### 2. Test Training Script

```bash
python scripts/train/train_vae_mimic.py --help
```

### 3. Check Directory Structure

```bash
ls -la
```

You should see:
- `data/` - Data handling
- `models/` - Model architectures (vae, gan, diffusion, ssm)
- `training/` - Training utilities
- `evaluation/` - Evaluation metrics
- `visualization/` - Visualization tools
- `scripts/` - Executable scripts
- `configs/` - Configuration files
- `docs/` - Documentation

## New Import Patterns

### Before (Old)
```python
from ecgen.models.vae import VAE1D
from ecgen.data.mimic_dataset import MIMICIVECGDataset
from ecgen.training.callbacks import VAEVisualizationCallback
```

### After (New)
```python
from models.vae import VAE1D
from data.datasets.mimic.dataset import MIMICIVECGDataset
from training.callbacks.visualization import VAEVisualizationCallback
```

## Quick Start After Installation

### Train VAE on MIMIC

```bash
# 1. Configure (optional - defaults are set)
nano configs/experiments/vae_mimic.yaml

# 2. Train
./scripts/shell/run_train_vae_mimic_config.sh

# 3. Monitor
tensorboard --logdir outputs/vae_mimic/
```

### Generate Samples

```python
from models.vae import VAELightning

# Load model
model = VAELightning.load_from_checkpoint("outputs/vae_mimic/checkpoints/best.ckpt")

# Generate
samples = model.sample(n_samples=16, seq_length=5000)
```

## Documentation

- **Main README**: [README.md](README.md)
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Project Structure**: [docs/project_structure.md](docs/project_structure.md)
- **Training Guide**: [docs/training/guide.md](docs/training/guide.md)
- **API Docs**: [docs/README.md](docs/README.md)

## Troubleshooting

### Import Errors

If you get import errors:
1. Make sure you're in the project root
2. Check `sys.path` includes project root
3. Verify package is installed: `pip list | grep ecgen`

### Training Script Errors

If training scripts fail:
1. Check imports are updated (see MIGRATION_GUIDE.md)
2. Verify data paths in configs
3. Check Python version >= 3.8

### Missing Dependencies

```bash
pip install torch pytorch-lightning pyyaml pandas scikit-learn wfdb matplotlib scipy numpy
```

## Getting Help

1. Check [docs/README.md](docs/README.md) for documentation
2. Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for import changes
3. See [REORGANIZATION_COMPLETE.md](REORGANIZATION_COMPLETE.md) for what changed

## Success Checklist

- [ ] PyYAML installed without errors
- [ ] `pip install -e .` completed successfully
- [ ] `verify_reorganization.py` shows all tests passed
- [ ] Can import models: `from models.vae import VAE1D`
- [ ] Training script runs: `python scripts/train/train_vae_mimic.py --help`

Once all checked, you're ready to use the reorganized repository! 🚀
