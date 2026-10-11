#!/usr/bin/env bash
# Train every experiment config in configs/experiments/ in turn.
# The prior needs a stage-1 checkpoint, so it runs after vqvae_mimic.yaml.
set -euo pipefail

# Change to ECGEN root (two levels up from scripts/shell/)
cd "$(dirname "$0")/../.."

for cfg in configs/experiments/*.yaml; do
  case "${cfg}" in
    *prior*) continue ;;
  esac
  echo "Launching training for ${cfg}"
  python -m ecgen train "${cfg}" "$@"
  echo ""
done

if [ -f runs/vqvae_mimic/seed_42/checkpoints/last.ckpt ]; then
  python -m ecgen train configs/experiments/prior_mimic.yaml \
    model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt "$@"
fi
