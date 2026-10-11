#!/usr/bin/env bash
# Train Pulse2Pulse (WaveGAN) on MIMIC-IV-ECG.  Usage: run_train_pulse2pulse_mimic.sh [config] [key=value ...]
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/experiments/pulse2pulse_mimic.yaml}"
shift || true

python -m ecgen train "${CONFIG}" "$@"
