#!/usr/bin/env bash
# Train Pulse2Pulse WaveGAN using standalone script (no config file needed)
set -euo pipefail

# Change to script directory's parent (ECGEN root)
cd "$(dirname "$0")/.."

# Configuration
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"
EXP_NAME="${EXP_NAME:-pulse2pulse_mimic}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=========================================="
echo "Pulse2Pulse Training (Standalone)"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Experiment name: ${EXP_NAME}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "GPU: ${GPU_ID}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Warning: Data directory not found: ${DATA_DIR}"
    echo "Please set DATA_DIR environment variable or edit this script"
    echo ""
    echo "Example:"
    echo "  DATA_DIR=/path/to/MIMIC-IV-ECG $0"
    echo ""
fi

# Run training
python scripts/train_pulse2pulse.py \
    --data-dir "${DATA_DIR}" \
    --exp-name "${EXP_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --accelerator gpu \
    --devices "${GPU_ID}" \
    --skip-missing-check \
    --viz-every-n-epochs 10 \
    --save-samples-every-n-epochs 25 \
    --check-val-every-n-epoch 5 \
    --log-every-n-steps 50

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check outputs in: runs/${EXP_NAME}/seed_42/"
echo "=========================================="
