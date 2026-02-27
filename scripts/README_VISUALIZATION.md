# VAE Visualization Quick Reference

## Quick Start

```bash
# Train with default visualization settings (every 5 epochs, 4 samples)
./scripts/run_train_vae_mimic_config.sh

# View visualizations in TensorBoard
tensorboard --logdir runs/vae_mimic/seed_42/tb
# Then open: http://localhost:6006 → IMAGES tab

# View saved images
ls runs/vae_mimic/seed_42/samples/
```

## Customize Visualization

### Option 1: Edit Config File

Edit `configs/experiments/vae_mimic.yaml`:

```yaml
visualization:
  every_n_epochs: 10     # Visualize every 10 epochs
  num_samples: 8         # Show 8 samples
  plot_all_leads: false  # Use overlay mode
```

### Option 2: Command Line

```bash
./scripts/run_train_vae_mimic_config.sh \
    --viz-every-n-epochs 10 \
    --viz-num-samples 8 \
    --viz-log-to-wandb
```

## Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--viz-every-n-epochs` | 5 | Visualize every N epochs |
| `--viz-num-samples` | 4 | Number of samples to plot |
| `--viz-plot-all-leads` | false | Plot each lead separately |
| `--viz-log-to-tensorboard` | true | Log to TensorBoard |
| `--viz-log-to-wandb` | false | Log to W&B |

## Output Locations

- **Disk**: `runs/{exp_name}/seed_{seed}/samples/epoch_XXXX.png`
- **TensorBoard**: IMAGES tab
- **W&B**: Media panel (if enabled)

## Visualization Modes

### Separate Leads (Medical Style)
```bash
--viz-plot-all-leads
```
- Each lead in its own subplot
- 2 rows per sample (real + reconstructed)
- 12 columns (one per lead)

### Overlay (Compact)
```bash
# Default - no flag needed
```
- All leads overlaid with colors
- 2 columns (real vs reconstructed)
- Compact view

## Common Use Cases

### Quick Testing
```bash
# Visualize every epoch with fewer samples
./scripts/run_train_vae_mimic_config.sh \
    --viz-every-n-epochs 1 \
    --viz-num-samples 2
```

### Production Training
```bash
# Less frequent, more samples
./scripts/run_train_vae_mimic_config.sh \
    --viz-every-n-epochs 10 \
    --viz-num-samples 8
```

### With W&B
```bash
# Enable W&B logging for visualizations
./scripts/run_train_vae_mimic_config.sh \
    --wandb \
    --viz-log-to-wandb
```

## Files

- **Callback**: `scripts/callbacks/vae_visualization.py`
- **Training Script**: `scripts/train_vae_mimic.py`
- **Config**: `configs/experiments/vae_mimic.yaml`
- **Documentation**: `docs/VAE_VISUALIZATION.md`
