#!/bin/bash

# Quick VAE Training Script - Minimal configuration for fast testing
# Usage: bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/train_vae_quick.sh /path/to/mimic-iv-ecg [max_samples]"
    echo ""
    echo "Example:"
    echo "  bash scripts/train_vae_quick.sh /data/mimic-iv-ecg 1000"
    exit 1
fi

DATA_DIR="$1"
MAX_SAMPLES="${2:-1000}"  # Default to 1000 samples for quick testing

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "============================================================================"
echo "Quick VAE Training (Debug Mode)"
echo "============================================================================"
echo "Data directory: $DATA_DIR"
echo "Max samples:    $MAX_SAMPLES"
echo "Output:         runs/vae/vae_quick_test"
echo "============================================================================"
echo ""

python "$SCRIPT_DIR/train_vae_full.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "runs/vae" \
    --exp_name "vae_quick_test" \
    --batch_size 16 \
    --num_workers 2 \
    --max_epochs 10 \
    --learning_rate 0.0001 \
    --kl_weight 0.0001 \
    --in_channels 12 \
    --base_channels 32 \
    --latent_channels 8 \
    --num_res_blocks 1 \
    --gpus 1 \
    --seed 42 \
    --max_samples "$MAX_SAMPLES" \
    --patience 5

echo ""
echo "============================================================================"
echo "Quick test completed!"
echo "============================================================================"
