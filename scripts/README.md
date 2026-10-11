# Scripts

Executable scripts for training, generation, evaluation, and data processing.

## Structure

- `train/` - `train_template.py` (notes on adding a model; training itself is `ecgen train <config>`)
- `generate/` - Generation scripts
  - `generate_pulse2pulse.py` - Generate samples with Pulse2Pulse
- `evaluate/` - Evaluation scripts (future)
- `data/` - Data processing scripts (future)
- `utils/` - Utility scripts
  - `test_models_only.py` - Test model instantiation
  - `test_pulse2pulse_setup.py` - Test Pulse2Pulse setup
- `shell/` - Shell wrappers around `python -m ecgen train <config> [overrides]`
  - `run_train_vae_mimic_config.sh`
  - `run_train_vqvae_mimic.sh`
  - `run_train_vqvae_prior_mimic.sh <vqvae_ckpt>`
  - `run_train_pulse2pulse_mimic.sh`
  - `run_train_pulse2pulse_standalone.sh` (settings from env vars)
  - `run_sweep.sh`

## Usage

### Training
```bash
# Train VAE
./scripts/shell/run_train_vae_mimic_config.sh

# Train Pulse2Pulse
./scripts/shell/run_train_pulse2pulse_mimic.sh
```

### Generation
```bash
# Generate samples
python scripts/generate/generate_pulse2pulse.py --checkpoint path/to/checkpoint
```
