#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/pulse2pulse_mimic.yaml}"

python -m ecgen.training.train --config "${CONFIG}"
