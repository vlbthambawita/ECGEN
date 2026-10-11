#!/usr/bin/env bash
# Train the PixelCNN prior (stage 2) on a frozen VQ-VAE.
# Usage: run_train_vqvae_prior_mimic.sh <vqvae_checkpoint> [config] [key=value ...]
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <vqvae_checkpoint> [config] [key=value ...]"
    echo "  e.g. $0 runs/vqvae_mimic/seed_42/checkpoints/last.ckpt"
    exit 1
fi
VQVAE_CHECKPOINT="$1"
CONFIG="${2:-configs/experiments/prior_mimic.yaml}"
shift; shift || true

if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
    echo "Error: VQ-VAE checkpoint not found: ${VQVAE_CHECKPOINT}"
    exit 1
fi

python -m ecgen train "${CONFIG}" "model.params.vqvae_checkpoint=${VQVAE_CHECKPOINT}" "$@"
