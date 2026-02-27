# Train VAE Right Now - Step by Step

This is a simple, step-by-step guide to train the VAE model immediately.

## Step 1: Verify Everything Works (30 seconds)

```bash
cd /work/vajira/DL2026/ECGEN
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH
python tests/test_vae.py
```

Expected output:
```
✓ VAE forward pass test successful
✓ VAE Lightning module test successful
✓ All tests passed!
```

## Step 2: Quick Test with Your Data (5 minutes)

Replace `/path/to/mimic-iv-ecg` with your actual data path:

```bash
bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg 1000
```

This will:
- Use only 1000 samples
- Train for 10 epochs
- Save to `runs/vae/vae_quick_test/`
- Complete in ~5 minutes

## Step 3: Full Training (hours to days)

### Option A: Simple (Edit the script)

1. Edit the script:
```bash
vim scripts/train_vae.sh
```

2. Change line 10:
```bash
DATA_DIR="/path/to/mimic-iv-ecg"  # CHANGE THIS
```

3. Run:
```bash
bash scripts/train_vae.sh
```

### Option B: Command Line (No editing needed)

```bash
bash scripts/train_vae.sh --data_dir /path/to/mimic-iv-ecg
```

### Option C: Custom Configuration

```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/mimic-iv-ecg \
    --exp_name my_vae_experiment \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 0.0001 \
    --gpus 1
```

## Step 4: Monitor Training

### Open TensorBoard

In a new terminal:
```bash
cd /work/vajira/DL2026/ECGEN
tensorboard --logdir runs/vae/
```

Then open: http://localhost:6006

### Watch Progress

```bash
# Check latest checkpoint
ls -lh runs/vae/*/checkpoints/

# View logs
tail -f runs/vae/*/logs/version_0/events.out.tfevents*
```

## Step 5: Use Trained Model

```python
from ecgen.models import VAELightning
import torch

# Load trained model
model = VAELightning.load_from_checkpoint(
    "runs/vae/your_exp_name/checkpoints/best.ckpt"
)
model.eval()

# Generate samples
samples = model.sample(n_samples=16, seq_length=5000)
print(f"Generated: {samples.shape}")  # [16, 12, 5000]

# Reconstruct ECGs
ecg = torch.randn(4, 12, 5000)  # Your real ECG data
recon = model.reconstruct(ecg)
print(f"Reconstructed: {recon.shape}")  # [4, 12, 5000]
```

## Common Issues and Solutions

### Issue 1: Data not found
```
FileNotFoundError: machine_measurements.csv not found
```

**Solution**: Check your data path has this structure:
```
mimic-iv-ecg/
├── machine_measurements.csv
└── files/
```

### Issue 2: Out of memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size:
```bash
bash scripts/train_vae.sh --data_dir /path/to/data --batch_size 16
```

### Issue 3: Import error
```
ModuleNotFoundError: No module named 'ecgen'
```

**Solution**: Set PYTHONPATH:
```bash
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH
```

## Training Configurations

### Fast (for testing)
```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/data \
    --base_channels 32 \
    --num_res_blocks 1 \
    --batch_size 64 \
    --max_epochs 50
```

### Default (recommended)
```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/data \
    --batch_size 32 \
    --max_epochs 100
```

### High Quality
```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/data \
    --base_channels 128 \
    --num_res_blocks 3 \
    --batch_size 16 \
    --max_epochs 200
```

## Expected Results

After training completes, you should see:

```
============================================================================
Training completed!
Best checkpoint: runs/vae/your_exp_name/checkpoints/vae-epoch=XX-val_loss=0.XXXX.ckpt
Best validation loss: 0.XXXX
============================================================================
```

Typical loss values:
- Reconstruction loss: 0.1 - 0.3
- KL loss: 0.01 - 0.05
- Total loss: 0.1 - 0.35

## What's Next?

1. **Evaluate**: Test reconstruction quality
2. **Visualize**: Plot original vs reconstructed ECGs
3. **Generate**: Create new synthetic ECGs
4. **Integrate**: Use with diffusion models

## Need Help?

- **Quick Start**: `docs/VAE_QUICKSTART.md`
- **Full Guide**: `TRAINING_GUIDE.md`
- **Documentation**: `docs/VAE_MODEL.md`
- **Scripts Help**: `scripts/README_VAE_TRAINING.md`

## Summary

```bash
# 1. Test (30 seconds)
python tests/test_vae.py

# 2. Quick test (5 minutes)
bash scripts/train_vae_quick.sh /path/to/data 1000

# 3. Full training (hours)
bash scripts/train_vae.sh --data_dir /path/to/data

# 4. Monitor
tensorboard --logdir runs/vae/

# 5. Use model
python -c "from ecgen.models import VAELightning; model = VAELightning.load_from_checkpoint('runs/vae/.../best.ckpt')"
```

That's it! You're ready to train the VAE model.
