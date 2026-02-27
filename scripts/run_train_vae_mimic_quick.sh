#!/usr/bin/env bash
# Quick VAE training test on MIMIC-IV-ECG dataset (for debugging)
set -euo pipefail

# Change to script directory's parent (ECGEN root)
cd "$(dirname "$0")/.."

# Default data directory (CHANGE THIS to your MIMIC-IV-ECG path)
DATA_DIR="${1:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    echo "Usage: $0 [data_dir]"
    exit 1
fi

echo "=========================================="
echo "VAE Quick Training Test (Debug Mode)"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Max samples: 1000"
echo "Max epochs: 10"
echo "=========================================="
echo ""

# Run quick training test
python scripts/train_vae_mimic.py \
    --data-dir "${DATA_DIR}" \
    --exp-name vae_mimic_quick \
    --seed 42 \
    --batch-size 16 \
    --num-workers 2 \
    --max-samples 1000 \
    --in-channels 12 \
    --base-channels 32 \
    --latent-channels 8 \
    --num-res-blocks 1 \
    --seq-length 5000 \
    --lr 0.0001 \
    --kl-weight 0.0001 \
    --max-epochs 10 \
    --accelerator gpu \
    --devices 0 \
    --log-every-n-steps 10 \
    --check-val-every-n-epoch 1 \
    --patience 5 \
    --save-top-k 1

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
