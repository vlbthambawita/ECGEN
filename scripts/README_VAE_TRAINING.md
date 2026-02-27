# VAE Training Scripts

This directory contains scripts for training the VAE model on MIMIC-IV-ECG data.

## Files

- `train_vae_full.py` - Complete Python training script with full configuration
- `train_vae.sh` - Main bash script for production training
- `train_vae_quick.sh` - Quick test script for debugging with limited samples

## Quick Start

### 1. Quick Test (Recommended First)

Test the training pipeline with a small subset of data:

```bash
bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg 1000
```

This will:
- Use only 1000 samples
- Train for 10 epochs
- Use smaller model (base_channels=32)
- Complete in a few minutes

### 2. Full Training

Once the quick test works, run full training:

```bash
# Edit the DATA_DIR in the script first
bash scripts/train_vae.sh
```

Or with command-line arguments:

```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/mimic-iv-ecg \
    --exp_name my_vae_experiment \
    --batch_size 32 \
    --max_epochs 100 \
    --gpus 1
```

### 3. Direct Python Script

For more control, use the Python script directly:

```bash
cd /work/vajira/DL2026/ECGEN
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH

python scripts/train_vae_full.py \
    --data_dir /path/to/mimic-iv-ecg \
    --output_dir runs/vae \
    --exp_name vae_experiment \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 0.0001 \
    --kl_weight 0.0001 \
    --gpus 1
```

## Configuration Options

### Data Settings

- `--data_dir` - Path to MIMIC-IV-ECG data directory (required)
- `--output_dir` - Output directory for checkpoints and logs (default: `runs/vae`)
- `--exp_name` - Experiment name (default: auto-generated with timestamp)
- `--max_samples` - Limit number of samples for debugging (optional)

### Training Hyperparameters

- `--batch_size` - Batch size (default: 32)
- `--num_workers` - Number of data workers (default: 4)
- `--max_epochs` - Maximum training epochs (default: 100)
- `--learning_rate` - Learning rate (default: 0.0001)
- `--kl_weight` - KL divergence weight (default: 0.0001)

### Model Architecture

- `--in_channels` - Number of ECG leads (default: 12)
- `--base_channels` - Base number of channels (default: 64)
- `--latent_channels` - Latent space channels (default: 8)
- `--num_res_blocks` - Residual blocks per stage (default: 2)

### Training Settings

- `--gpus` - Number of GPUs (default: 1)
- `--seed` - Random seed (default: 42)
- `--gradient_clip` - Gradient clipping value (default: 1.0)
- `--patience` - Early stopping patience (default: 10)
- `--save_top_k` - Save top k checkpoints (default: 3)

### Data Splits

- `--val_split` - Validation split ratio (default: 0.1)
- `--test_split` - Test split ratio (default: 0.1)

### Resume Training

- `--resume` - Path to checkpoint to resume from

## Examples

### Example 1: Quick Debug Run

```bash
bash scripts/train_vae_quick.sh /data/mimic-iv-ecg 500
```

### Example 2: Small Model for Fast Training

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_small \
    --base_channels 32 \
    --num_res_blocks 1 \
    --batch_size 64 \
    --max_epochs 50
```

### Example 3: Large Model for Best Quality

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_large \
    --base_channels 128 \
    --num_res_blocks 3 \
    --batch_size 16 \
    --max_epochs 200
```

### Example 4: Resume from Checkpoint

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_resumed \
    --resume runs/vae/vae_experiment/checkpoints/last.ckpt \
    --max_epochs 150
```

### Example 5: Multi-GPU Training

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_multigpu \
    --gpus 2 \
    --batch_size 64
```

## Output Structure

After training, you'll find:

```
runs/vae/
└── your_exp_name/
    ├── checkpoints/
    │   ├── vae-epoch=XX-val_loss=X.XXXX.ckpt
    │   ├── last.ckpt
    │   └── ...
    └── logs/
        └── version_0/
            ├── events.out.tfevents...
            └── hparams.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/vae/your_exp_name/logs
```

Then open http://localhost:6006 in your browser.

### Check Progress

```bash
# View latest logs
tail -f runs/vae/your_exp_name/logs/version_0/events.out.tfevents*

# Check checkpoints
ls -lh runs/vae/your_exp_name/checkpoints/
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH
```

### Issue: Out of Memory

Reduce batch size or model size:
```bash
bash scripts/train_vae.sh \
    --batch_size 16 \
    --base_channels 32
```

### Issue: Data Not Found

Ensure your data directory has this structure:
```
mimic-iv-ecg/
├── machine_measurements.csv
└── files/
    └── p0000/
        └── p10000032/
            └── s50000001/
                ├── 50000001.hea
                └── 50000001.dat
```

### Issue: Training Too Slow

- Increase `num_workers`
- Use multiple GPUs
- Reduce model size
- Use `--max_samples` for debugging

### Issue: Poor Reconstruction

- Decrease `kl_weight` (e.g., 0.00001)
- Increase `base_channels` (e.g., 128)
- Train for more epochs
- Increase `num_res_blocks`

## Hardware Requirements

### Minimum

- 1 GPU with 8GB VRAM
- 16GB RAM
- 50GB disk space

### Recommended

- 1 GPU with 16GB+ VRAM
- 32GB+ RAM
- 100GB+ disk space

## Expected Training Time

With default settings on a single V100 GPU:

- Quick test (1000 samples, 10 epochs): ~5 minutes
- Small dataset (10k samples, 100 epochs): ~2-3 hours
- Full MIMIC-IV-ECG (100k+ samples, 100 epochs): ~1-2 days

## Model Performance

Expected validation loss after training:
- Reconstruction loss: 0.1 - 0.3
- KL loss: 0.01 - 0.05
- Total loss: 0.1 - 0.35

Lower is better, but very low KL loss might indicate posterior collapse.

## Next Steps

After training:

1. Load the best checkpoint
2. Generate samples
3. Evaluate reconstruction quality
4. Use for latent diffusion

See `docs/VAE_MODEL.md` for usage examples.

## Support

For issues:
1. Check this README
2. Review `docs/VAE_MODEL.md`
3. Check `docs/VAE_QUICKSTART.md`
4. Examine the training script code
