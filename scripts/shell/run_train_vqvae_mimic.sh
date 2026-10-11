#!/usr/bin/env bash
# Train VQ-VAE (stage 1) on MIMIC-IV-ECG.  Usage: run_train_vqvae_mimic.sh [config] [key=value ...]
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/experiments/vqvae_mimic.yaml}"
shift || true

python -m ecgen train "${CONFIG}" "$@"

echo ""
echo "Next: train the prior (stage 2) with"
echo "  ./scripts/shell/run_train_vqvae_prior_mimic.sh runs/<exp>/seed_<seed>/checkpoints/last.ckpt"
