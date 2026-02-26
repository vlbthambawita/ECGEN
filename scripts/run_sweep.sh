#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs/experiments"

for cfg in "${CONFIG_DIR}"/*.yaml; do
  echo "Launching training for ${cfg}"
  python -m my_project.training.train --config "${cfg}"
done
