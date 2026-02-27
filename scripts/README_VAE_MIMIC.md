# VAE Training Scripts for MIMIC-IV-ECG

Training scripts for VAE model on MIMIC-IV-ECG dataset, following the same pattern as Pulse2Pulse training.

## Files

- `train_vae_mimic.py` - Main Python training script
- `run_train_vae_mimic.sh` - Shell wrapper for production training
- `run_train_vae_mimic_quick.sh` - Quick test script for debugging
- `../configs/experiments/vae_mimic.yaml` - YAML configuration file

## Quick Start

### 1. Quick Test (5 minutes)

Test the training pipeline with limited samples:

```bash
# From ECGEN root directory
bash scripts/run_train_vae_mimic_quick.sh /path/to/mimic-iv-ecg
```

This will:
- Use 1000 samples
- Train for 10 epochs
- Use smaller model (base_channels=32)
- Complete in ~5 minutes

### 2. Full Training

Run full training with default settings:

```bash
bash scripts/run_train_vae_mimic.sh /path/to/mimic-iv-ecg
```

Or use the default path in the script:

```bash
# Edit the script to set your data path
vim scripts/run_train_vae_mimic.sh
# Then run
bash scripts/run_train_vae_mimic.sh
```

### 3. Custom Training

Use the Python script directly for full control:

```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name my_vae_experiment \
    --batch-size 32 \
    --max-epochs 100 \
    --in-channels 12 \
    --base-channels 64 \
    --latent-channels 8
```

## Script Comparison with Pulse2Pulse

These scripts follow the same structure as `run_train_pulse2pulse_mimic.sh`:

| Feature | Pulse2Pulse | VAE |
|---------|-------------|-----|
| Shell wrapper | ✓ | ✓ |
| Python script | ✓ | ✓ |
| YAML config | ✓ | ✓ |
| Quick test script | ✗ | ✓ |
| W&B logging | ✓ | ✓ |
| Checkpoint management | ✓ | ✓ |
| Early stopping | ✗ | ✓ |

## Configuration

### Default Settings (run_train_vae_mimic.sh)

```bash
Experiment: vae_mimic
Seed: 42
Batch size: 32
Max epochs: 100
Learning rate: 0.0001
KL weight: 0.0001

Model:
  Input channels: 12
  Base channels: 64
  Latent channels: 8
  Residual blocks: 2
  Sequence length: 5000
```

### Quick Test Settings (run_train_vae_mimic_quick.sh)

```bash
Experiment: vae_mimic_quick
Max samples: 1000
Batch size: 16
Max epochs: 10
Base channels: 32 (smaller model)
Residual blocks: 1
```

## Command-Line Arguments

The Python script supports all these arguments:

### Experiment
- `--exp-name` - Experiment name (default: vae_mimic)
- `--seed` - Random seed (default: 42)
- `--runs-root` - Root directory for runs (default: runs)

### Data
- `--data-dir` - Path to MIMIC-IV-ECG dataset (required)
- `--batch-size` - Batch size (default: 32)
- `--num-workers` - Data loading workers (default: 4)
- `--max-samples` - Limit samples for debugging (optional)
- `--skip-missing-check` - Skip missing file check
- `--val-split` - Validation split ratio (default: 0.1)
- `--test-split` - Test split ratio (default: 0.1)

### Model
- `--in-channels` - Number of ECG leads (default: 12)
- `--base-channels` - Base channels (default: 64)
- `--latent-channels` - Latent channels (default: 8)
- `--num-res-blocks` - Residual blocks (default: 2)
- `--seq-length` - Sequence length (default: 5000)
- `--lr` - Learning rate (default: 0.0001)
- `--kl-weight` - KL divergence weight (default: 0.0001)
- `--b1` - Adam beta1 (default: 0.9)
- `--b2` - Adam beta2 (default: 0.999)

### Training
- `--max-epochs` - Maximum epochs (default: 100)
- `--accelerator` - Accelerator type (default: gpu)
- `--devices` - Device IDs (default: [0])
- `--log-every-n-steps` - Log frequency (default: 50)
- `--check-val-every-n-epoch` - Validation frequency (default: 1)
- `--gradient-clip` - Gradient clipping (default: 1.0)
- `--patience` - Early stopping patience (default: 10)
- `--save-top-k` - Save top k checkpoints (default: 3)

### Weights & Biases
- `--wandb` - Enable W&B logging
- `--wandb-project` - W&B project (default: ecg-vae)
- `--wandb-entity` - W&B entity
- `--wandb-run-name` - W&B run name
- `--wandb-tags` - W&B tags

### Resume
- `--resume` - Path to checkpoint to resume from

## Usage Examples

### Example 1: Quick Test

```bash
bash scripts/run_train_vae_mimic_quick.sh /data/mimic-iv-ecg
```

### Example 2: Full Training with Custom Settings

```bash
python scripts/train_vae_mimic.py \
    --data-dir /data/mimic-iv-ecg \
    --exp-name vae_large \
    --batch-size 16 \
    --max-epochs 200 \
    --base-channels 128 \
    --num-res-blocks 3 \
    --lr 0.00005
```

### Example 3: Multi-GPU Training

```bash
python scripts/train_vae_mimic.py \
    --data-dir /data/mimic-iv-ecg \
    --exp-name vae_multigpu \
    --batch-size 64 \
    --devices 0 1
```

### Example 4: With Weights & Biases

```bash
python scripts/train_vae_mimic.py \
    --data-dir /data/mimic-iv-ecg \
    --exp-name vae_wandb \
    --wandb \
    --wandb-project my-ecg-project \
    --wandb-tags vae mimic-iv experiment-1
```

### Example 5: Resume Training

```bash
python scripts/train_vae_mimic.py \
    --data-dir /data/mimic-iv-ecg \
    --exp-name vae_resumed \
    --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt \
    --max-epochs 150
```

## Output Structure

```
runs/
└── vae_mimic/
    └── seed_42/
        ├── checkpoints/
        │   ├── epoch000-step000000.ckpt
        │   ├── epoch010-step001000.ckpt
        │   ├── last.ckpt
        │   └── ...
        ├── samples/
        └── tb/
            └── version_0/
                ├── events.out.tfevents...
                └── hparams.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

Open http://localhost:6006

### Check Progress

```bash
# View latest checkpoint
ls -lh runs/vae_mimic/seed_42/checkpoints/

# Monitor training
tail -f runs/vae_mimic/seed_42/tb/version_0/events.out.tfevents*
```

## Comparison with Other Training Scripts

### Similar to Pulse2Pulse
- Same directory structure
- Same argument naming convention
- Same logging setup
- Same checkpoint management

### Differences from train_vae_full.py
- Follows ECGEN project structure
- Uses same patterns as Pulse2Pulse
- Integrated with project conventions
- Simpler shell wrapper

## Integration with ECGEN

These scripts are fully integrated with the ECGEN framework:

```python
# Import paths work correctly
from ecgen.models.vae import VAELightning, VAEConfig
from ecgen.data.mimic_dataset import MIMICIVECGDataset
from ecgen.utils.seed import set_global_seed
```

## Troubleshooting

### Issue: Data not found

```bash
Error: Data directory not found: /path/to/data
```

**Solution**: Update the data path in the script or pass it as argument:
```bash
bash scripts/run_train_vae_mimic.sh /correct/path/to/mimic-iv-ecg
```

### Issue: Out of memory

**Solution**: Reduce batch size or model size:
```bash
python scripts/train_vae_mimic.py \
    --data-dir /data/mimic-iv-ecg \
    --batch-size 16 \
    --base-channels 32
```

### Issue: Import errors

**Solution**: The script automatically adds src to path, but if issues persist:
```bash
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH
```

## Performance

### Training Time (V100 GPU)

- Quick test (1k samples, 10 epochs): ~5 minutes
- Default (10k samples, 100 epochs): ~3-4 hours
- Full dataset (100k+ samples, 100 epochs): ~1-2 days

### Model Size

- Small (base=32): ~1.5M parameters
- Default (base=64): ~6.2M parameters
- Large (base=128): ~25M parameters

### Expected Losses

- Reconstruction loss: 0.1 - 0.3
- KL loss: 0.01 - 0.05
- Total loss: 0.1 - 0.35

## Next Steps

After training:

1. **Load checkpoint**:
```python
from ecgen.models.vae import VAELightning
model = VAELightning.load_from_checkpoint(
    "runs/vae_mimic/seed_42/checkpoints/last.ckpt"
)
```

2. **Generate samples**:
```python
samples = model.sample(n_samples=16, seq_length=5000)
```

3. **Evaluate reconstruction**:
```python
reconstructed = model.reconstruct(ecg_signals)
```

## Additional Resources

- Full VAE documentation: `docs/VAE_MODEL.md`
- Quick start guide: `docs/VAE_QUICKSTART.md`
- Training guide: `TRAINING_GUIDE.md`
- Config file: `configs/experiments/vae_mimic.yaml`
