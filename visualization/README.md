# Visualization

Visualization tools for ECG signals, latent spaces, and training progress.

## Modules

### ecg_plots.py
ECG-specific plotting utilities.

```python
from visualization.ecg_plots import plot_ecg_leads, plot_ecg_comparison

# Plot all 12 leads
fig = plot_ecg_leads(ecg_signal, title="Patient ECG")

# Compare real vs generated
fig = plot_ecg_comparison(real_ecg, generated_ecg, lead_idx=0)
```

### latent_space.py
Latent space visualization for VAE and other models.

```python
from visualization.latent_space import plot_latent_space_2d

# Visualize latent space with PCA
fig = plot_latent_space_2d(latents, method="pca")

# Visualize with t-SNE
fig = plot_latent_space_2d(latents, method="tsne")
```

### training_curves.py
Training progress visualization.

```python
from visualization.training_curves import plot_training_curves

# Plot training and validation losses
fig = plot_training_curves(train_losses, val_losses)
```

## Usage in Training

### With Callbacks

The visualization callbacks in `training/callbacks/visualization.py` use these utilities automatically.

### Manual Usage

```python
import matplotlib.pyplot as plt
from visualization.ecg_plots import plot_ecg_leads

# Load your ECG data
ecg = ...  # Shape: (12, 5000)

# Create plot
fig = plot_ecg_leads(ecg, title="My ECG")

# Save
fig.savefig("ecg_plot.png", dpi=300, bbox_inches='tight')
plt.close(fig)
```

## Customization

All plotting functions return matplotlib Figure objects, allowing further customization:

```python
fig = plot_ecg_leads(ecg)

# Customize
fig.suptitle("Custom Title", fontsize=16)
axes = fig.get_axes()
axes[0].set_ylim(-1, 1)

# Save
fig.savefig("custom_plot.png")
```
