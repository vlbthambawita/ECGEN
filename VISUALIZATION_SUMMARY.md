# VAE Visualization Feature - Summary

## What Was Added

A comprehensive visualization system to monitor VAE reconstruction quality during training.

## Files Created/Modified

### New Files Created

1. **`scripts/callbacks/vae_visualization.py`**
   - Custom PyTorch Lightning callback for VAE visualization
   - Two visualization modes: separate leads and overlaid leads
   - TensorBoard and Weights & Biases integration
   - ~270 lines of well-documented code

2. **`scripts/callbacks/__init__.py`**
   - Module initialization for callbacks package

3. **`docs/VAE_VISUALIZATION.md`**
   - Comprehensive documentation
   - Usage guide, configuration options, troubleshooting
   - Examples and best practices

### Modified Files

1. **`scripts/train_vae_mimic.py`**
   - Added import for visualization callback
   - Added command-line arguments for visualization settings
   - Added YAML config parsing for visualization section
   - Integrated callback into training pipeline

2. **`configs/experiments/vae_mimic.yaml`**
   - Added `visualization` section with default settings
   - Configured to visualize every 5 epochs by default

## Features

✅ **Automatic Visualization**
- Generates comparison plots during training
- Configurable frequency (every N epochs)
- Multiple samples per visualization

✅ **Two Visualization Styles**
- Separate leads: Each lead in its own subplot (medical-style)
- Overlaid leads: All leads overlaid in one plot (compact)

✅ **Multiple Output Formats**
- Saved as PNG files to disk
- Logged to TensorBoard (optional)
- Logged to Weights & Biases (optional)

✅ **Flexible Configuration**
- YAML config file support
- Command-line argument overrides
- Programmatic API

## Usage

### Quick Start

Just run your training script as usual - visualization is enabled by default:

```bash
./scripts/run_train_vae_mimic_config.sh
```

Visualizations will be saved to: `runs/vae_mimic/seed_42/samples/`

### View in TensorBoard

```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

Navigate to the "IMAGES" tab to see visualizations.

### Customize Settings

Edit `configs/experiments/vae_mimic.yaml`:

```yaml
visualization:
  every_n_epochs: 5      # How often to visualize
  num_samples: 4         # Number of samples
  plot_all_leads: true   # Separate (true) or overlay (false)
  log_to_tensorboard: true
  log_to_wandb: false
```

Or use command-line arguments:

```bash
./scripts/run_train_vae_mimic_config.sh \
    --viz-every-n-epochs 10 \
    --viz-num-samples 8 \
    --viz-plot-all-leads
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `every_n_epochs` | 5 | Generate visualizations every N epochs |
| `num_samples` | 4 | Number of ECG samples to visualize |
| `plot_all_leads` | true | Plot leads separately (true) or overlaid (false) |
| `log_to_tensorboard` | true | Log images to TensorBoard |
| `log_to_wandb` | false | Log images to Weights & Biases |

## Output Examples

### Separate Leads Mode (`plot_all_leads: true`)

Creates a grid showing:
- **Rows**: 2 per sample (real + reconstructed)
- **Columns**: 12 (one per ECG lead: I, II, III, aVR, aVL, aVF, V1-V6)
- **Layout**: Medical-style, easy to compare each lead

### Overlay Mode (`plot_all_leads: false`)

Creates a compact view:
- **Columns**: 2 (real vs reconstructed)
- **Rows**: Number of samples
- **Layout**: All leads overlaid with different colors

## Benefits

1. **Early Problem Detection**: Spot training issues immediately
2. **Quality Assessment**: Visual confirmation of reconstruction quality
3. **Hyperparameter Tuning**: Compare different configurations visually
4. **Progress Tracking**: See improvement over epochs
5. **Publication Ready**: High-quality plots for papers/presentations

## Performance Impact

- **Minimal**: Visualization runs only during validation
- **Configurable**: Adjust frequency to balance monitoring vs speed
- **Default settings**: ~1-2 seconds per visualization every 5 epochs

## Next Steps

1. **Start training** and monitor visualizations
2. **Check TensorBoard** to view images during training
3. **Adjust settings** based on your needs
4. **Compare experiments** by examining visualization outputs

## Documentation

For detailed information, see: `docs/VAE_VISUALIZATION.md`

## Technical Details

- **Framework**: PyTorch Lightning Callback
- **Plotting**: Matplotlib
- **Image Format**: PNG (150 DPI)
- **Dependencies**: No additional packages required (uses existing dependencies)
