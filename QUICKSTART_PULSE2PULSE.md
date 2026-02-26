# Pulse2Pulse Quick Start

## 🚀 Start Training in 3 Steps

### Step 1: Navigate to ECGEN directory
```bash
cd /work/vajira/DL2026/ECGEN
```

### Step 2: Choose your training method

#### Method A: Config-based (Recommended)
```bash
./scripts/run_train_pulse2pulse_mimic.sh
```

#### Method B: Standalone (No config file)
```bash
./scripts/run_train_pulse2pulse_standalone.sh
```

#### Method C: Direct Python (Most flexible)
```bash
python scripts/train_pulse2pulse.py \
    --data-dir /work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --batch-size 128 \
    --max-epochs 300
```

### Step 3: Monitor training
```bash
# Open TensorBoard
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb

# Or use Weights & Biases (optional)
# Add --wandb flag or enable in config
# View at: https://wandb.ai

# Check samples
ls runs/pulse2pulse_mimic/seed_42/samples/
```

## 📊 Output Location

All outputs are saved to:
```
runs/pulse2pulse_mimic/seed_42/
├── checkpoints/          # Model checkpoints
├── samples/              # Generated ECG samples
└── tb/                   # TensorBoard logs
```

## 🎯 Generate Samples After Training

```bash
python scripts/generate_pulse2pulse.py \
    --checkpoint runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt \
    --n-samples 16
```

## ⚙️ Quick Customization

### Change batch size
```bash
BATCH_SIZE=64 ./scripts/run_train_pulse2pulse_standalone.sh
```

### Use different GPU
```bash
GPU_ID=1 ./scripts/run_train_pulse2pulse_standalone.sh
```

### Quick test (small dataset)
```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --max-samples 1000 \
    --max-epochs 10
```

## 🔍 Test Setup First

```bash
python scripts/test_models_only.py
```

Expected: ✓ ALL TESTS PASSED!

## 🔬 Use Weights & Biases (Optional)

Track experiments in the cloud:

```bash
# Install and login
pip install wandb
wandb login

# Enable in config or add --wandb flag
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse
```

See `WANDB_QUICKSTART.md` for details.

## 📚 More Information

- **Full guide**: `README_PULSE2PULSE.md`
- **Training details**: `docs/pulse2pulse_training.md`
- **Script reference**: `scripts/README_TRAINING.md`
- **Wandb setup**: `WANDB_QUICKSTART.md` or `docs/wandb_setup.md`
- **Changelog**: `CHANGELOG_PULSE2PULSE.md`

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size: `BATCH_SIZE=64` |
| Data not found | Update `DATA_DIR` in config or script |
| Wrong GPU | Set `GPU_ID=1` or `--devices 1` |
| Import errors | `pip install --upgrade jinja2` |

## ✅ What You Get

- **Model**: WaveGAN generator + discriminator (~344M params)
- **Training**: WGAN-GP with automatic checkpointing
- **Monitoring**: TensorBoard + sample visualization
- **Outputs**: Checkpoints, samples, plots, metrics

## 🎓 Training Time

- **Quick test** (10 epochs, 1K samples): ~5-10 minutes
- **Full training** (300 epochs, full dataset): ~24-48 hours on single GPU

## 🔄 Resume Training

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --resume runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt
```

---

**Ready to start?** Run `./scripts/run_train_pulse2pulse_mimic.sh` now! 🚀
