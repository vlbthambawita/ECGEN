#!/usr/bin/env bash
# Train PixelCNN Prior (Stage 2) on MIMIC-IV-ECG dataset
# Requires a trained VQ-VAE checkpoint from Stage 1
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

# Check arguments
if [ $# -lt 1 ]; then
    echo "Error: VQ-VAE checkpoint path is required"
    echo "Usage: $0 <vqvae_checkpoint> [config_path]"
    echo ""
    echo "Example:"
    echo "  $0 runs/vqvae_mimic/seed_42/checkpoints/best.ckpt"
    echo "  $0 runs/vqvae_mimic/seed_42/checkpoints/best.ckpt configs/experiments/prior_mimic.yaml"
    exit 1
fi

VQVAE_CHECKPOINT="$1"
CONFIG="${2:-configs/experiments/prior_mimic.yaml}"

# Check if VQ-VAE checkpoint exists
if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    echo "Error: VQ-VAE checkpoint not found: ${VQVAE_CHECKPOINT}"
    echo "Please train VQ-VAE first using run_train_vqvae_mimic.sh"
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    exit 1
fi

echo "=========================================="
echo "Prior Training (Stage 2)"
echo "=========================================="
echo "VQ-VAE checkpoint: ${VQVAE_CHECKPOINT}"
echo "Config: ${CONFIG}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Run training using the config file and VQ-VAE checkpoint
python scripts/train/train_prior_mimic.py \
    --config "${CONFIG}" \
    --vqvae-checkpoint "${VQVAE_CHECKPOINT}"

echo ""
echo "=========================================="
echo "Prior training (Stage 2) completed!"
echo "=========================================="
