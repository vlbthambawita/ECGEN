#!/usr/bin/env bash
# Train Pulse2Pulse with settings from environment variables (no config editing needed).
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"
EXP_NAME="${EXP_NAME:-pulse2pulse_mimic}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python -m ecgen train configs/experiments/pulse2pulse_mimic.yaml \
    "experiment.name=${EXP_NAME}" \
    "data.dataset.params.root=${DATA_DIR}" \
    "data.dataset.params.max_samples=null" \
    "data.dataset.params.skip_missing_check=true" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "trainer.max_epochs=${MAX_EPOCHS}" \
    "trainer.devices=[${GPU_ID}]" \
    "wandb.enabled=false" \
    "$@"
