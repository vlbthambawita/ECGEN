# Quick Start Guide - VQ-VAE Standalone Training

## TL;DR - Get Started in 3 Steps

### 1. Set Your Data Path

```bash
export DATA_DIR=/path/to/your/mimic-iv-ecg
```

### 2. Run Training (Both Stages)

```bash
cd /work/vajira/DL2026/ECGEN/single_file
./run_train_vqvae.sh both
```

### 3. Monitor Progress

```bash
tensorboard --logdir=runs
```

---

## Common Use Cases

### Quick Test (Small Dataset, Fast)

```bash
# Train on 100 samples for 10 epochs each stage
MAX_SAMPLES=100 \
MAX_EPOCHS_STAGE1=10 \
MAX_EPOCHS_STAGE2=10 \
./run_train_vqvae.sh both
```

**Expected time:** ~10-20 minutes on GPU

### Production Training (Full Dataset)

```bash
# Train on full dataset
DATA_DIR=/path/to/mimic-iv-ecg \
MAX_SAMPLES=null \
MAX_EPOCHS_STAGE1=200 \
MAX_EPOCHS_STAGE2=200 \
BATCH_SIZE=64 \
./run_train_vqvae.sh both
```

**Expected time:** Several hours to days depending on dataset size

### Train Only Stage 1 (VQ-VAE)

```bash
./run_train_vqvae.sh 1
```

### Train Only Stage 2 (Prior)

```bash
# Requires VQ-VAE checkpoint from Stage 1
VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/last.ckpt \
./run_train_vqvae.sh 2
```

---

## File Overview

- **`train_vqvae_standalone.py`** - Single Python file with all code (~1400 lines)
  - All model architectures
  - Dataset loading
  - Training loops
  - Loss functions
  - Everything needed to train

- **`run_train_vqvae.sh`** - Shell script for easy training
  - Configurable via environment variables
  - Runs both stages automatically
  - Handles checkpoint management

- **`README.md`** - Complete documentation
  - Detailed usage instructions
  - All configuration options
  - Troubleshooting guide

---

## Key Configuration Options

### Most Important Settings

```bash
# Data
DATA_DIR=/path/to/mimic-iv-ecg    # Your data path
MAX_SAMPLES=1000                   # Number of samples (null for all)

# Training
BATCH_SIZE=32                      # Batch size
MAX_EPOCHS_STAGE1=100             # Epochs for Stage 1
MAX_EPOCHS_STAGE2=100             # Epochs for Stage 2

# Model
NUM_EMBEDDINGS=512                # Codebook size
LATENT_CHANNELS=64                # Latent dimension
```

### Full List

See `README.md` for complete list of ~30+ configuration options.

---

## Output Files

After training, you'll find:

```
runs/
├── vqvae_mimic_standalone/seed_42/
│   ├── checkpoints/          # Model checkpoints
│   ├── samples/              # Reconstruction visualizations
│   └── tb/                   # TensorBoard logs
└── prior_mimic_standalone/seed_42/
    ├── checkpoints/          # Prior model checkpoints
    └── tb/                   # TensorBoard logs
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

### Key Metrics to Watch

**Stage 1 (VQ-VAE):**
- `val/recon_loss` - Should decrease (reconstruction quality)
- `val/codebook_usage` - Should be > 0.5 (using enough codes)

**Stage 2 (Prior):**
- `val/loss` - Should decrease (prior learning)

---

## Troubleshooting

### Problem: CUDA Out of Memory

**Solution:** Reduce batch size
```bash
BATCH_SIZE=16 ./run_train_vqvae.sh both
```

### Problem: Data not found

**Solution:** Check data path
```bash
DATA_DIR=/correct/path/to/mimic-iv-ecg ./run_train_vqvae.sh both
```

### Problem: Training too slow

**Solution:** Use fewer samples for testing
```bash
MAX_SAMPLES=100 ./run_train_vqvae.sh both
```

### Problem: Low codebook usage (< 0.3)

**Solution:** Increase commitment cost
```bash
COMMITMENT_COST=0.5 ./run_train_vqvae.sh both
```

---

## Generate Samples After Training

```python
import torch
from train_vqvae_standalone import PriorLightning

# Load trained model
model = PriorLightning.load_from_checkpoint(
    "runs/prior_mimic_standalone/seed_42/checkpoints/last.ckpt"
)
model.eval()
model = model.cuda()

# Generate 16 ECG samples
samples = model.sample(n_samples=16, seq_length=5000, temperature=1.0)
print(f"Generated shape: {samples.shape}")  # [16, 12, 5000]

# Save
torch.save(samples.cpu(), "generated_ecgs.pt")
```

---

## Python Direct Usage (Without Shell Script)

### Stage 1

```bash
python train_vqvae_standalone.py \
    --stage 1 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name my_vqvae \
    --batch-size 32 \
    --max-epochs 100
```

### Stage 2

```bash
python train_vqvae_standalone.py \
    --stage 2 \
    --data-dir /path/to/mimic-iv-ecg \
    --exp-name my_prior \
    --vqvae-checkpoint runs/my_vqvae/seed_42/checkpoints/last.ckpt \
    --batch-size 32 \
    --max-epochs 100
```

---

## What's Happening Under the Hood?

### Stage 1: VQ-VAE Training

1. **Encoder** compresses ECG [12, 5000] → latent [64, 312]
2. **Vector Quantizer** discretizes latent → 312 discrete codes from codebook of 512
3. **Decoder** reconstructs ECG from discrete codes → [12, 5000]
4. **Loss** = reconstruction error + vector quantization loss

### Stage 2: Prior Training

1. **VQ-VAE Encoder** (frozen) converts ECG → discrete codes [312]
2. **PixelCNN** learns to predict next code autoregressively
3. **Loss** = cross-entropy for predicting next code
4. **Generation** = sample codes from PixelCNN, decode with VQ-VAE

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Look at TensorBoard logs for training progress
3. Verify data path is correct
4. Try with smaller dataset first (`MAX_SAMPLES=100`)

---

## Example: Complete Workflow

```bash
# 1. Navigate to directory
cd /work/vajira/DL2026/ECGEN/single_file

# 2. Set data path
export DATA_DIR=/path/to/mimic-iv-ecg

# 3. Quick test (100 samples, 10 epochs)
MAX_SAMPLES=100 MAX_EPOCHS_STAGE1=10 MAX_EPOCHS_STAGE2=10 ./run_train_vqvae.sh both

# 4. Monitor training
tensorboard --logdir=runs

# 5. Check results
ls runs/vqvae_mimic_standalone/seed_42/samples/

# 6. Generate samples (in Python)
python -c "
from train_vqvae_standalone import PriorLightning
import torch
model = PriorLightning.load_from_checkpoint('runs/prior_mimic_standalone/seed_42/checkpoints/last.ckpt')
model.eval()
model = model.cuda()
samples = model.sample(n_samples=4, seq_length=5000)
torch.save(samples.cpu(), 'generated_ecgs.pt')
print('Generated 4 ECG samples!')
"
```

That's it! You're ready to train VQ-VAE models on ECG data.
