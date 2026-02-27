#!/usr/bin/env bash
# Train VAE on MIMIC-IV-ECG dataset
set -euo pipefail

# Change to script directory's parent (ECGEN root)
cd "$(dirname "$0")/.."

# Default data directory (CHANGE THIS to your MIMIC-IV-ECG path)
DATA_DIR="${1:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    echo "Usage: $0 [data_dir]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/mimic-iv-ecg"
    exit 1
fi

# Check if machine_measurements.csv exists
if [ ! -f "${DATA_DIR}/machine_measurements.csv" ]; then
    echo "Error: machine_measurements.csv not found in ${DATA_DIR}"
    echo "Please ensure you have the complete MIMIC-IV-ECG dataset"
    exit 1
fi

echo "=========================================="
echo "VAE Training on MIMIC-IV-ECG"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Run training
python scripts/train_vae_mimic.py \
    --data-dir "${DATA_DIR}" \
    --exp-name vae_mimic \
    --seed 42 \
    --batch-size 32 \
    --num-workers 4 \
    --max-samples 10000 \
    --in-channels 12 \
    --base-channels 64 \
    --latent-channels 8 \
    --num-res-blocks 2 \
    --seq-length 5000 \
    --lr 0.0001 \
    --kl-weight 0.0001 \
    --max-epochs 100 \
    --accelerator gpu \
    --devices 0 \
    --log-every-n-steps 50 \
    --check-val-every-n-epoch 1 \
    --gradient-clip 1.0 \
    --patience 10 \
    --save-top-k 3

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
