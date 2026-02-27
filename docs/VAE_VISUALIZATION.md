# VAE Visualization Guide

This document explains how to use the VAE visualization feature to monitor reconstruction quality during training.

## Overview

The VAE visualization callback automatically generates comparison plots of real vs reconstructed ECG signals during training. This helps you:

- **Monitor reconstruction quality** visually throughout training
- **Detect training issues** early (e.g., mode collapse, poor reconstruction)
- **Compare different model configurations** by examining visual outputs
- **Track improvement** over epochs

## Features

- ✅ **Automatic visualization** at configurable intervals (every N epochs)
- ✅ **Two plot styles**: separate leads or overlaid leads
- ✅ **Multiple samples** per visualization (configurable)
- ✅ **TensorBoard integration** for easy viewing
- ✅ **Weights & Biases integration** (optional)
- ✅ **Saved to disk** for later inspection

## Configuration

### Via YAML Config File

Edit your config file (e.g., `configs/experiments/vae_mimic.yaml`):

```yaml
visualization:
  every_n_epochs: 5        # Generate visualizations every 5 epochs
  num_samples: 4           # Visualize 4 samples per epoch
  plot_all_leads: true     # true = separate subplots, false = overlay
  log_to_tensorboard: true # Log to TensorBoard
  log_to_wandb: false      # Log to Weights & Biases (requires wandb enabled)
```

### Via Command Line

Override config settings with command-line arguments:

```bash
./scripts/run_train_vae_mimic_config.sh configs/experiments/vae_mimic.yaml \
    --viz-every-n-epochs 10 \
    --viz-num-samples 8 \
    --viz-plot-all-leads \
    --viz-log-to-wandb
```

## Visualization Styles

### 1. Separate Leads (Recommended)

Set `plot_all_leads: true` to plot each ECG lead in its own subplot.

**Advantages:**
- Clear view of each individual lead
- Easy to spot lead-specific issues
- Professional medical-style layout

**Output:** Each sample shows 2 rows × 12 columns (real + reconstructed for all 12 leads)

### 2. Overlaid Leads

Set `plot_all_leads: false` to overlay all leads in a single plot.

**Advantages:**
- Compact visualization
- Good for quick overview
- Shows overall signal shape

**Output:** Each sample shows 2 plots side-by-side (real vs reconstructed)

## Output Locations

Visualizations are saved to multiple locations:

1. **Disk**: `runs/{exp_name}/seed_{seed}/samples/epoch_{epoch:04d}.png`
2. **TensorBoard**: View in the "IMAGES" tab
3. **Weights & Biases**: View in the "Media" panel (if enabled)

## Viewing Visualizations

### During Training

**TensorBoard:**
```bash
tensorboard --logdir runs/vae_mimic/seed_42/tb
```

Then open http://localhost:6006 and navigate to the "IMAGES" tab.

**Weights & Biases:**

If W&B is enabled, visualizations appear automatically in your run's dashboard.

### After Training

Browse the saved images:

```bash
ls runs/vae_mimic/seed_42/samples/
# epoch_0000.png, epoch_0005.png, epoch_0010.png, ...
```

View with any image viewer or Python:

```python
from PIL import Image
img = Image.open("runs/vae_mimic/seed_42/samples/epoch_0050.png")
img.show()
```

## Interpreting Visualizations

### What to Look For

**Good Reconstruction:**
- Reconstructed signals closely match real signals
- All leads are preserved
- QRS complexes, P waves, and T waves are visible
- Amplitude and timing are similar

**Poor Reconstruction:**
- Blurry or smoothed signals
- Missing features (e.g., P waves)
- Wrong amplitudes
- Phase shifts or timing issues
- Artifacts or noise

### Common Issues

| Observation | Possible Cause | Solution |
|------------|----------------|----------|
| Blurry reconstructions | KL weight too high | Reduce `kl_weight` |
| Perfect reconstructions, high KL loss | KL weight too low | Increase `kl_weight` |
| Noisy reconstructions | Model underfitting | Increase `base_channels` or `num_res_blocks` |
| Missing high-frequency details | Insufficient model capacity | Increase model size |
| Reconstructions don't improve | Learning rate issues | Adjust `lr` |

## Performance Considerations

### Frequency

- **Default: Every 5 epochs** - Good balance between monitoring and performance
- **Quick testing: Every 1 epoch** - Use during debugging
- **Long training: Every 10-20 epochs** - Reduce overhead for very long runs

### Number of Samples

- **Default: 4 samples** - Sufficient for most cases
- **More samples (8-16)** - Better statistical view, but slower
- **Fewer samples (1-2)** - Faster, good for quick checks

### Disk Space

Each visualization is ~500KB-1MB depending on settings. For 100 epochs with `every_n_epochs=5`:
- Total images: 20
- Disk space: ~10-20 MB

## Advanced Usage

### Custom Visualization Callback

You can customize the visualization by using the callback directly:

```python
from ecgen.training.callbacks import VAEVisualizationCallback

# Create custom callback
custom_viz = VAEVisualizationCallback(
    save_dir="custom_output",
    log_every_n_epochs=1,
    num_samples=8,
    plot_all_leads=True,
    log_to_tensorboard=True,
    log_to_wandb=False,
)
```

### Programmatic Access

```python
import pytorch_lightning as pl
from ecgen.training.callbacks import VAEVisualizationCallback

# Add to your trainer
trainer = pl.Trainer(
    callbacks=[
        # ... other callbacks ...
        VAEVisualizationCallback(
            save_dir="visualizations",
            log_every_n_epochs=5,
        ),
    ]
)
```

## Troubleshooting

### Visualizations Not Generated

1. Check that validation is running: `check_val_every_n_epoch` in trainer config
2. Ensure `viz_every_n_epochs` is less than `max_epochs`
3. Check the samples directory exists: `runs/{exp_name}/seed_{seed}/samples/`

### TensorBoard Not Showing Images

1. Verify TensorBoard is pointing to the correct log directory
2. Refresh the browser
3. Check `viz_log_to_tensorboard: true` in config

### Out of Memory

1. Reduce `viz_num_samples`
2. Reduce `batch_size` (affects validation batch)
3. Use `plot_all_leads: false` for smaller plots

## Example Workflow

1. **Start training** with default visualization settings:
   ```bash
   ./scripts/run_train_vae_mimic_config.sh
   ```

2. **Monitor in TensorBoard**:
   ```bash
   tensorboard --logdir runs/vae_mimic/seed_42/tb
   ```

3. **Check early epochs** (0, 5, 10) to ensure training is working

4. **Adjust hyperparameters** if reconstructions are poor

5. **Compare runs** by viewing different experiment directories

## References

- Visualization callback: `src/ecgen/training/callbacks.py` (VAEVisualizationCallback)
- Training script: `scripts/train_vae_mimic.py`
- Config file: `configs/experiments/vae_mimic.yaml`
- Main training script: `scripts/run_train_vae_mimic_config.sh`
