#!/usr/bin/env bash
# Train Pulse2Pulse WaveGAN on MIMIC-IV-ECG dataset
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

# Default config
CONFIG="${1:-configs/experiments/pulse2pulse_mimic.yaml}"

# Check if config exists
if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    echo "Usage: $0 [config_path]"
    exit 1
fi

echo "=========================================="
echo "Pulse2Pulse Training"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Run training
python scripts/train/train_pulse2pulse.py --config "${CONFIG}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
