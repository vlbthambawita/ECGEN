#!/usr/bin/env bash
# Run training sweep across multiple experiment configs
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

CONFIG_DIR="configs/experiments"

echo "=========================================="
echo "Training Sweep"
echo "=========================================="
echo "Config directory: ${CONFIG_DIR}"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

for cfg in "${CONFIG_DIR}"/*.yaml; do
  echo "Launching training for ${cfg}"
  
  # Determine which training script to use based on config name
  if [[ "${cfg}" == *"vae"* ]]; then
    python scripts/train/train_vae_mimic.py --config "${cfg}"
  elif [[ "${cfg}" == *"pulse2pulse"* ]]; then
    python scripts/train/train_pulse2pulse.py --config "${cfg}"
  else
    echo "Warning: Unknown config type for ${cfg}, skipping..."
  fi
  
  echo ""
done

echo "=========================================="
echo "Sweep completed!"
echo "=========================================="
