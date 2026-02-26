#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/exp01_imn_ptbxl_100hz.yaml}"

python -m my_project.training.train --config "${CONFIG}"
