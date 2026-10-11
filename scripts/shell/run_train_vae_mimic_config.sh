#!/usr/bin/env bash
# Train VAE on MIMIC-IV-ECG.  Usage: run_train_vae_mimic_config.sh [config] [key=value ...]
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/experiments/vae_mimic.yaml}"
shift || true

python -m ecgen train "${CONFIG}" "$@"
