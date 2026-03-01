# VQ-VAE Standalone Training Scripts

This directory contains a self-contained implementation for training VQ-VAE models on ECG data in two stages.

## Overview

The VQ-VAE (Vector Quantized Variational Autoencoder) model learns discrete representations of ECG signals through a two-stage training process:

1. **Stage 1 (VQ-VAE Training)**: Train the encoder, vector quantizer, and decoder to learn discrete latent representations
2. **Stage 2 (Prior Training)**: Train a PixelCNN autoregressive prior to model the distribution of discrete codes

## Files

- `train_vqvae_standalone.py` - Single Python file containing all model components and training logic
- `run_train_vqvae.sh` - Shell script to easily run training with configurable parameters
- `README.md` - This file

## Features

### Single File Implementation
All necessary components are included in `train_vqvae_standalone.py`:
- Model architectures (Encoder, Decoder, Vector Quantizer, PixelCNN)
- Dataset loading (MIMIC-IV-ECG)
- Training loops (PyTorch Lightning)
- Loss functions
- Visualization callbacks
- Utilities

### Two-Stage Training
The script supports training both stages independently or sequentially:
- Stage 1: VQ-VAE reconstruction training
- Stage 2: Prior distribution learning

## Requirements

```bash
pip install torch pytorch-lightning matplotlib numpy pandas scikit-learn wfdb
```

## Usage

### Quick Start (Both Stages)

```bash
# Train both stages sequentially
./run_train_vqvae.sh both

# Or specify data directory
DATA_DIR=/path/to/mimic-iv-ecg ./run_train_vqvae.sh both
```

### Stage-by-Stage Training

```bash
# Stage 1: Train VQ-VAE only
./run_train_vqvae.sh 1

# Stage 2: Train Prior only (requires VQ-VAE checkpoint)
VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/last.ckpt ./run_train_vqvae.sh 2
```

### Using Python Directly

```bash
# Stage 1: Train VQ-VAE
python train_vqvae_standalone.py \
    --stage 1 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name vqvae_exp \
    --batch-size 32 \
    --max-epochs 100 \
    --num-embeddings 512

# Stage 2: Train Prior
python train_vqvae_standalone.py \
    --stage 2 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name prior_exp \
    --vqvae-checkpoint runs/vqvae_exp/seed_42/checkpoints/best.ckpt \
    --batch-size 32 \
    --max-epochs 100
```

## Configuration

### Environment Variables (Shell Script)

The shell script supports configuration through environment variables:

#### Data Settings
- `DATA_DIR` - Path to MIMIC-IV-ECG dataset (required)
- `BATCH_SIZE` - Batch size (default: 32)
- `NUM_WORKERS` - Number of data loading workers (default: 4)
- `MAX_SAMPLES` - Maximum samples to use, null for full dataset (default: 1000)
- `VAL_SPLIT` - Validation split ratio (default: 0.1)
- `TEST_SPLIT` - Test split ratio (default: 0.1)

#### Experiment Settings
- `EXP_NAME_STAGE1` - Experiment name for Stage 1 (default: vqvae_mimic_standalone)
- `EXP_NAME_STAGE2` - Experiment name for Stage 2 (default: prior_mimic_standalone)
- `SEED` - Random seed (default: 42)
- `RUNS_ROOT` - Root directory for runs (default: runs)

#### Model Settings (Stage 1)
- `IN_CHANNELS` - Number of ECG leads (default: 12)
- `BASE_CHANNELS` - Base number of channels (default: 64)
- `LATENT_CHANNELS` - Latent channels (default: 64)
- `NUM_RES_BLOCKS` - Number of residual blocks (default: 2)
- `NUM_EMBEDDINGS` - Codebook size (default: 512)
- `COMMITMENT_COST` - Commitment loss weight (default: 0.25)
- `SEQ_LENGTH` - ECG sequence length (default: 5000)

#### Model Settings (Stage 2)
- `HIDDEN_DIM` - Hidden dimension for PixelCNN (default: 128)
- `NUM_LAYERS` - Number of gated conv layers (default: 3)
- `VQVAE_CHECKPOINT` - Path to VQ-VAE checkpoint (required for Stage 2)

#### Training Settings
- `LR_STAGE1` - Learning rate for Stage 1 (default: 0.0001)
- `LR_STAGE2` - Learning rate for Stage 2 (default: 0.001)
- `MAX_EPOCHS_STAGE1` - Max epochs for Stage 1 (default: 100)
- `MAX_EPOCHS_STAGE2` - Max epochs for Stage 2 (default: 100)
- `ACCELERATOR` - Accelerator type: gpu/cpu (default: gpu)
- `DEVICES` - Device IDs (default: 0)
- `GRADIENT_CLIP` - Gradient clipping value (default: 1.0)
- `PATIENCE` - Early stopping patience (default: 10)
- `SAVE_TOP_K` - Save top k checkpoints (default: 3)

#### Visualization Settings (Stage 1)
- `VIZ_EVERY_N_EPOCHS` - Generate visualizations every N epochs (default: 5)
- `VIZ_NUM_SAMPLES` - Number of samples to visualize (default: 4)

### Example Configurations

#### Quick Testing (Small Dataset)
```bash
MAX_SAMPLES=100 \
MAX_EPOCHS_STAGE1=10 \
MAX_EPOCHS_STAGE2=10 \
./run_train_vqvae.sh both
```

#### Full Training (Production)
```bash
DATA_DIR=/path/to/mimic-iv-ecg \
MAX_SAMPLES=null \
MAX_EPOCHS_STAGE1=200 \
MAX_EPOCHS_STAGE2=200 \
BATCH_SIZE=64 \
NUM_EMBEDDINGS=1024 \
./run_train_vqvae.sh both
```

#### Custom Model Architecture
```bash
BASE_CHANNELS=128 \
LATENT_CHANNELS=128 \
NUM_EMBEDDINGS=2048 \
HIDDEN_DIM=256 \
NUM_LAYERS=5 \
./run_train_vqvae.sh both
```

## Command-Line Arguments (Python Script)

### Common Arguments
- `--stage` - Training stage: 1 (VQ-VAE) or 2 (Prior) [required]
- `--exp-name` - Experiment name
- `--seed` - Random seed
- `--runs-root` - Root directory for runs
- `--data-dir` - Path to MIMIC-IV-ECG dataset [required]
- `--batch-size` - Batch size
- `--num-workers` - Number of data loading workers
- `--max-samples` - Max samples to use (for debugging)
- `--skip-missing-check` - Skip missing file check
- `--val-split` - Validation split ratio
- `--test-split` - Test split ratio

### Stage 1 Arguments
- `--in-channels` - Number of ECG leads
- `--base-channels` - Base number of channels
- `--latent-channels` - Latent channels
- `--num-res-blocks` - Number of residual blocks
- `--num-embeddings` - Codebook size
- `--commitment-cost` - Commitment loss weight
- `--seq-length` - ECG sequence length
- `--viz-every-n-epochs` - Generate visualizations every N epochs
- `--viz-num-samples` - Number of samples to visualize

### Stage 2 Arguments
- `--hidden-dim` - Hidden dimension for PixelCNN
- `--num-layers` - Number of gated conv layers
- `--vqvae-checkpoint` - Path to VQ-VAE checkpoint [required for Stage 2]

### Training Arguments
- `--lr` - Learning rate
- `--b1` - Adam beta1
- `--b2` - Adam beta2
- `--max-epochs` - Maximum number of epochs
- `--accelerator` - Accelerator type (gpu/cpu)
- `--devices` - Device IDs
- `--log-every-n-steps` - Log every N steps
- `--check-val-every-n-epoch` - Validate every N epochs
- `--gradient-clip` - Gradient clipping value
- `--patience` - Early stopping patience
- `--save-top-k` - Save top k checkpoints

## Output Structure

```
runs/
├── vqvae_mimic_standalone/
│   └── seed_42/
│       ├── checkpoints/
│       │   ├── epoch000-step000100.ckpt
│       │   ├── epoch001-step000200.ckpt
│       │   └── last.ckpt
│       ├── samples/
│       │   ├── epoch_0000.png
│       │   ├── epoch_0005.png
│       │   └── ...
│       └── tb/
│           └── version_0/
│               └── events.out.tfevents...
└── prior_mimic_standalone/
    └── seed_42/
        ├── checkpoints/
        │   ├── epoch000-step000100.ckpt
        │   ├── epoch001-step000200.ckpt
        │   └── last.ckpt
        └── tb/
            └── version_0/
                └── events.out.tfevents...
```

## Monitoring Training

### TensorBoard

```bash
# View all experiments
tensorboard --logdir=runs

# View specific experiment
tensorboard --logdir=runs/vqvae_mimic_standalone/seed_42/tb
```

### Logged Metrics

**Stage 1 (VQ-VAE):**
- `train/total_loss` - Total loss (reconstruction + VQ)
- `train/recon_loss` - Reconstruction loss (MSE)
- `train/vq_loss` - Vector quantization loss
- `train/unique_codes` - Number of unique codes used
- `train/codebook_usage` - Fraction of codebook used
- `val/total_loss` - Validation total loss
- `val/recon_loss` - Validation reconstruction loss
- `val/vq_loss` - Validation VQ loss
- `val/unique_codes` - Validation unique codes
- `val/codebook_usage` - Validation codebook usage

**Stage 2 (Prior):**
- `train/loss` - Cross-entropy loss for autoregressive prediction
- `val/loss` - Validation cross-entropy loss

## Model Architecture

### Stage 1: VQ-VAE

```
Input ECG [B, 12, 5000]
    ↓
Encoder (Conv1D + ResBlocks + Downsampling)
    ↓
Latent [B, 64, 312]
    ↓
Vector Quantizer (Codebook size: 512)
    ↓
Quantized Latent [B, 64, 312]
Discrete Indices [B, 312]
    ↓
Decoder (Conv1D + ResBlocks + Upsampling)
    ↓
Reconstructed ECG [B, 12, 5000]
```

### Stage 2: PixelCNN Prior

```
Discrete Indices [B, 312]
    ↓
Embedding Layer
    ↓
Gated Masked Conv1D Layers (Autoregressive)
    ↓
Logits [B, 512, 312]
    ↓
Cross-Entropy Loss
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `BATCH_SIZE=16 ./run_train_vqvae.sh`
   - Reduce model size: `BASE_CHANNELS=32 LATENT_CHANNELS=32 ./run_train_vqvae.sh`

2. **Data Loading Errors**
   - Check data path: `DATA_DIR=/correct/path ./run_train_vqvae.sh`
   - Skip missing file check: `python train_vqvae_standalone.py --skip-missing-check ...`

3. **Low Codebook Usage**
   - Increase commitment cost: `COMMITMENT_COST=0.5 ./run_train_vqvae.sh`
   - Reduce codebook size: `NUM_EMBEDDINGS=256 ./run_train_vqvae.sh`

4. **Training Not Converging**
   - Adjust learning rate: `LR_STAGE1=0.0005 ./run_train_vqvae.sh`
   - Increase gradient clipping: `GRADIENT_CLIP=0.5 ./run_train_vqvae.sh`

## Advanced Usage

### Resume Training

```bash
# Resume Stage 1 from checkpoint
python train_vqvae_standalone.py \
    --stage 1 \
    --data-dir /path/to/mimic \
    --exp-name vqvae_exp \
    --resume runs/vqvae_exp/seed_42/checkpoints/last.ckpt
```

### Generate Samples (After Training)

```python
import torch
from train_vqvae_standalone import PriorLightning

# Load trained prior
model = PriorLightning.load_from_checkpoint("runs/prior_exp/seed_42/checkpoints/best.ckpt")
model.eval()
model = model.cuda()

# Generate samples
samples = model.sample(n_samples=16, seq_length=5000, temperature=1.0)
print(f"Generated samples shape: {samples.shape}")  # [16, 12, 5000]

# Save samples
torch.save(samples.cpu(), "generated_ecg_samples.pt")
```

### Hyperparameter Search

```bash
# Grid search over codebook sizes
for NUM_EMB in 256 512 1024 2048; do
    NUM_EMBEDDINGS=$NUM_EMB \
    EXP_NAME_STAGE1="vqvae_emb${NUM_EMB}" \
    ./run_train_vqvae.sh 1
done
```

## Citation

If you use this code, please cite the VQ-VAE paper:

```bibtex
@inproceedings{oord2017neural,
  title={Neural discrete representation learning},
  author={Oord, Aaron van den and Vinyals, Oriol and Kavukcuoglu, Koray},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6306--6315},
  year={2017}
}
```

## License

This code is provided as-is for research purposes.

## Contact

For questions or issues, please refer to the main ECGEN repository documentation.
