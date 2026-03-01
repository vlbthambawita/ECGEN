#!/usr/bin/env bash
# Train VQ-VAE (Stage 1) on MIMIC-IV-ECG dataset using config file
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

# Default config
CONFIG="${1:-configs/experiments/vqvae_mimic.yaml}"

# Check if config exists
if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    echo "Usage: $0 [config_path]"
    exit 1
fi

echo "=========================================="
echo "VQ-VAE Training (Stage 1)"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Run training using the config file
python scripts/train/train_vqvae_mimic.py --config "${CONFIG}"

echo ""
echo "=========================================="
echo "VQ-VAE training (Stage 1) completed!"
echo "=========================================="
