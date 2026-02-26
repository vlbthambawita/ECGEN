#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/exp01_imn_ptbxl_100hz.yaml}"
SPLIT="${2:-test}"
OUTPUT="${3:-preds_${SPLIT}.parquet}"

python -m my_project.evaluation.eval_utils \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}"
