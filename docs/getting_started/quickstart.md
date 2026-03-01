# VAE Training - Quick Reference

## One-Line Quick Start

```bash
# Quick test (5 minutes)
bash scripts/run_train_vae_mimic_quick.sh /path/to/mimic-iv-ecg

# Full training
bash scripts/run_train_vae_mimic.sh /path/to/mimic-iv-ecg
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/train_vae_mimic.py` | Main Python training script |
| `scripts/run_train_vae_mimic.sh` | Shell wrapper (production) |
| `scripts/run_train_vae_mimic_quick.sh` | Quick test script |
| `configs/experiments/vae_mimic.yaml` | YAML configuration |
| `scripts/README_VAE_MIMIC.md` | Full documentation |

## Three Ways to Train

### 1. Quick Test (Recommended First)
```bash
bash scripts/run_train_vae_mimic_quick.sh /path/to/data
```
- 1000 samples
- 10 epochs
- ~5 minutes

### 2. Shell Wrapper (Easy)
```bash
bash scripts/run_train_vae_mimic.sh /path/to/data
```
- 10k samples
- 100 epochs
- Default settings

### 3. Python Script (Full Control)
```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --exp-name my_experiment \
    --batch-size 32 \
    --max-epochs 100
```

## Common Commands

### Monitor Training
```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

### Check Checkpoints
```bash
ls -lh runs/vae_mimic/seed_42/checkpoints/
```

### Load Trained Model
```python
from ecgen.models.vae import VAELightning
model = VAELightning.load_from_checkpoint("runs/vae_mimic/seed_42/checkpoints/last.ckpt")
```

### Generate Samples
```python
samples = model.sample(n_samples=16, seq_length=5000)
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to MIMIC-IV-ECG |
| `--exp-name` | vae_mimic | Experiment name |
| `--batch-size` | 32 | Batch size |
| `--max-epochs` | 100 | Training epochs |
| `--base-channels` | 64 | Model size |
| `--latent-channels` | 8 | Latent dimensions |
| `--lr` | 0.0001 | Learning rate |
| `--kl-weight` | 0.0001 | KL weight |

## Quick Examples

### Small Model (Fast)
```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --base-channels 32 \
    --batch-size 64 \
    --max-epochs 50
```

### Large Model (Quality)
```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --base-channels 128 \
    --batch-size 16 \
    --max-epochs 200
```

### With W&B
```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --wandb \
    --wandb-project my-project
```

### Resume
```bash
python scripts/train_vae_mimic.py \
    --data-dir /path/to/data \
    --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
```

## Output Structure

```
runs/vae_mimic/seed_42/
├── checkpoints/
│   ├── epoch000-step000000.ckpt
│   └── last.ckpt
├── samples/
└── tb/
    └── version_0/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Data not found | Update path in script or pass as argument |
| Out of memory | Reduce `--batch-size` or `--base-channels` |
| Import error | Script auto-handles, or set PYTHONPATH |
| Slow training | Increase `--num-workers` or use multi-GPU |

## Performance

| Configuration | Time (V100) | Parameters |
|---------------|-------------|------------|
| Quick test | ~5 min | ~1.5M |
| Default | ~3-4 hours | ~6.2M |
| Full dataset | ~1-2 days | ~6.2M |
| Large model | ~2-3 days | ~25M |

## Expected Losses

- Reconstruction: 0.1 - 0.3
- KL divergence: 0.01 - 0.05
- Total: 0.1 - 0.35

## Documentation

- Full guide: `VAE_MIMIC_TRAINING.md`
- Detailed docs: `scripts/README_VAE_MIMIC.md`
- Model docs: `docs/VAE_MODEL.md`
- Config: `configs/experiments/vae_mimic.yaml`

## Similar to Pulse2Pulse

```bash
# Pulse2Pulse
bash scripts/run_train_pulse2pulse_mimic.sh

# VAE (same style!)
bash scripts/run_train_vae_mimic.sh
```

Same pattern, same simplicity!
