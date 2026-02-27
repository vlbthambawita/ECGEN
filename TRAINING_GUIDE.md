# VAE Training Guide

Complete guide for training the VAE model in ECGEN.

## Prerequisites

1. **Data**: MIMIC-IV-ECG dataset downloaded and extracted
2. **Environment**: Python 3.8+, PyTorch, PyTorch Lightning installed
3. **Hardware**: At least 1 GPU with 8GB+ VRAM recommended

## Quick Start (3 Steps)

### Step 1: Verify Installation

```bash
cd /work/vajira/DL2026/ECGEN
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH

# Test imports
python -c "from ecgen.models import VAELightning; print('✓ VAE model ready')"
```

### Step 2: Quick Test Run

Test with a small subset of data (takes ~5 minutes):

```bash
bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg 1000
```

### Step 3: Full Training

Once the quick test works, run full training:

```bash
# Option A: Edit DATA_DIR in the script
vim scripts/train_vae.sh  # Change DATA_DIR="/path/to/mimic-iv-ecg"
bash scripts/train_vae.sh

# Option B: Use command-line arguments
bash scripts/train_vae.sh --data_dir /path/to/mimic-iv-ecg
```

## Training Scripts

### 1. Quick Test Script (`train_vae_quick.sh`)

**Purpose**: Fast testing with limited samples

**Usage**:
```bash
bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg [max_samples]
```

**Example**:
```bash
bash scripts/train_vae_quick.sh /data/mimic-iv-ecg 1000
```

**Settings**:
- 1000 samples (default)
- 10 epochs
- Small model (base_channels=32)
- Batch size 16

### 2. Full Training Script (`train_vae.sh`)

**Purpose**: Production training with full configuration

**Usage**:
```bash
bash scripts/train_vae.sh [options]
```

**Common Options**:
```bash
bash scripts/train_vae.sh \
    --data_dir /path/to/mimic-iv-ecg \
    --exp_name my_experiment \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 0.0001 \
    --gpus 1
```

**All Options**:
- `--data_dir PATH` - Data directory (required)
- `--output_dir PATH` - Output directory
- `--exp_name NAME` - Experiment name
- `--batch_size N` - Batch size
- `--max_epochs N` - Maximum epochs
- `--learning_rate LR` - Learning rate
- `--gpus N` - Number of GPUs
- `--seed N` - Random seed
- `--resume PATH` - Resume from checkpoint
- `--max_samples N` - Limit samples (debugging)

### 3. Python Script (`train_vae_full.py`)

**Purpose**: Direct Python execution with full control

**Usage**:
```bash
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH

python scripts/train_vae_full.py \
    --data_dir /path/to/mimic-iv-ecg \
    --output_dir runs/vae \
    --exp_name vae_experiment \
    --batch_size 32 \
    --max_epochs 100
```

## Configuration Examples

### Example 1: Default Settings

```bash
bash scripts/train_vae.sh --data_dir /data/mimic-iv-ecg
```

Settings:
- Batch size: 32
- Epochs: 100
- Learning rate: 0.0001
- Base channels: 64
- Latent channels: 8

### Example 2: Fast Training (Small Model)

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_fast \
    --base_channels 32 \
    --num_res_blocks 1 \
    --batch_size 64 \
    --max_epochs 50
```

### Example 3: High Quality (Large Model)

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_large \
    --base_channels 128 \
    --num_res_blocks 3 \
    --batch_size 16 \
    --max_epochs 200 \
    --learning_rate 0.00005
```

### Example 4: Multi-GPU Training

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_multigpu \
    --gpus 2 \
    --batch_size 64
```

### Example 5: Resume Training

```bash
bash scripts/train_vae.sh \
    --data_dir /data/mimic-iv-ecg \
    --exp_name vae_resumed \
    --resume runs/vae/vae_experiment/checkpoints/last.ckpt \
    --max_epochs 150
```

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir runs/vae/your_exp_name/logs
```

Open http://localhost:6006 in your browser.

**Metrics to Watch**:
- `train/recon_loss` - Reconstruction loss (should decrease)
- `train/kl_loss` - KL divergence (should stabilize)
- `val/total_loss` - Validation loss (for early stopping)

### Command Line

```bash
# Watch training progress
tail -f runs/vae/your_exp_name/logs/version_0/events.out.tfevents*

# Check saved checkpoints
ls -lh runs/vae/your_exp_name/checkpoints/

# View best checkpoint
cat runs/vae/your_exp_name/checkpoints/best.ckpt
```

## Output Structure

```
runs/vae/
└── your_exp_name/
    ├── checkpoints/
    │   ├── vae-epoch=XX-val_loss=X.XXXX.ckpt  # Top-k checkpoints
    │   ├── last.ckpt                           # Latest checkpoint
    │   └── best.ckpt -> vae-epoch=...          # Best checkpoint (symlink)
    └── logs/
        └── version_0/
            ├── events.out.tfevents...          # TensorBoard logs
            └── hparams.yaml                    # Hyperparameters
```

## Using Trained Model

### Load Checkpoint

```python
from ecgen.models import VAELightning

# Load trained model
model = VAELightning.load_from_checkpoint(
    "runs/vae/your_exp_name/checkpoints/best.ckpt"
)
model.eval()
```

### Generate Samples

```python
import torch

# Generate ECG samples
samples = model.sample(n_samples=16, seq_length=5000)
print(f"Generated samples shape: {samples.shape}")  # [16, 12, 5000]
```

### Reconstruct ECGs

```python
# Load some ECG data
ecg_signals = torch.randn(4, 12, 5000)  # Replace with real data

# Reconstruct
with torch.no_grad():
    reconstructed = model.reconstruct(ecg_signals)
```

### Extract Latent Representations

```python
# Encode to latent space
with torch.no_grad():
    latent = model.vae.encode_to_latent(ecg_signals)
    print(f"Latent shape: {latent.shape}")  # [4, 8, 625]
```

## Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'ecgen'
```

**Solution**:
```bash
export PYTHONPATH=/work/vajira/DL2026/ECGEN/src:$PYTHONPATH
```

### Problem: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `--batch_size 16`
2. Reduce model size: `--base_channels 32`
3. Use gradient checkpointing
4. Use mixed precision training

### Problem: Data Not Found

```
FileNotFoundError: machine_measurements.csv not found
```

**Solution**: Ensure data directory structure:
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

### Problem: Training Too Slow

**Solutions**:
1. Increase workers: `--num_workers 8`
2. Use multiple GPUs: `--gpus 2`
3. Increase batch size: `--batch_size 64`
4. Use smaller model for testing

### Problem: Poor Reconstruction Quality

**Solutions**:
1. Reduce KL weight: `--kl_weight 0.00001`
2. Increase model capacity: `--base_channels 128`
3. Train longer: `--max_epochs 200`
4. Add more residual blocks: `--num_res_blocks 3`

### Problem: Posterior Collapse (KL → 0)

**Solutions**:
1. Increase KL weight: `--kl_weight 0.001`
2. Use KL annealing (modify training script)
3. Reduce model capacity

## Performance Expectations

### Training Time (Single V100 GPU)

- Quick test (1k samples, 10 epochs): ~5 minutes
- Small (10k samples, 100 epochs): ~2-3 hours
- Full MIMIC-IV (100k+ samples, 100 epochs): ~1-2 days

### Model Size

- Small (base_channels=32): ~1.5M parameters
- Default (base_channels=64): ~6.2M parameters
- Large (base_channels=128): ~25M parameters

### Expected Losses

After training with default settings:
- Reconstruction loss: 0.1 - 0.3
- KL loss: 0.01 - 0.05
- Total loss: 0.1 - 0.35

## Hardware Requirements

### Minimum

- 1 GPU with 8GB VRAM
- 16GB RAM
- 50GB disk space

### Recommended

- 1 GPU with 16GB+ VRAM (e.g., V100, A100)
- 32GB+ RAM
- 100GB+ disk space
- SSD for faster data loading

## Best Practices

1. **Start Small**: Always run quick test first
2. **Monitor Early**: Watch TensorBoard from the start
3. **Save Often**: Use `--save_top_k 3` to keep multiple checkpoints
4. **Use Early Stopping**: Default patience=10 works well
5. **Tune KL Weight**: This is the most important hyperparameter
6. **Validate Often**: Check reconstruction quality periodically

## Next Steps

After successful training:

1. **Evaluate**: Test reconstruction quality on held-out data
2. **Visualize**: Plot original vs reconstructed ECGs
3. **Analyze**: Explore latent space structure
4. **Use**: Integrate with diffusion models or other applications

## Additional Resources

- Full documentation: `docs/VAE_MODEL.md`
- Quick start guide: `docs/VAE_QUICKSTART.md`
- Training scripts README: `scripts/README_VAE_TRAINING.md`
- Test suite: `tests/test_vae.py`

## Support

For issues or questions:
1. Check this guide
2. Review the documentation
3. Run the quick test script
4. Examine training script code
5. Check TensorBoard logs
