# Training Callbacks

Guide to using callbacks in ECGEN.

## Available Callbacks

### Visualization Callbacks

Located in `training/callbacks/visualization.py`:

#### VAEVisualizationCallback

Visualizes VAE reconstructions during training.

```python
from training.callbacks.visualization import VAEVisualizationCallback

callback = VAEVisualizationCallback(
    save_dir="outputs/vae/samples",
    log_every_n_epochs=5,
    num_samples=4,
    plot_all_leads=True,
)
```

#### ECGVisualizationCallback

Visualizes GAN outputs (real vs generated).

```python
from training.callbacks.visualization import ECGVisualizationCallback

callback = ECGVisualizationCallback(
    save_dir="outputs/gan/samples",
    log_every_n_epochs=5,
)
```

#### GeneratedSamplesCallback

Saves generated samples periodically.

```python
from training.callbacks.visualization import GeneratedSamplesCallback

callback = GeneratedSamplesCallback(
    save_dir="outputs/samples",
    log_every_n_epochs=10,
)
```

## Using Callbacks

### In Training Scripts

```python
import pytorch_lightning as pl
from training.callbacks.visualization import VAEVisualizationCallback

# Create callbacks
viz_callback = VAEVisualizationCallback(save_dir="outputs/samples")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="outputs/checkpoints",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
)

# Add to trainer
trainer = pl.Trainer(
    callbacks=[viz_callback, checkpoint_callback],
    max_epochs=100,
)
```

### Configuration Files

```yaml
# configs/experiments/my_experiment.yaml
callbacks:
  - type: vae_visualization
    save_dir: outputs/samples
    log_every_n_epochs: 5
  - type: model_checkpoint
    save_top_k: 3
    monitor: val_loss
```

## Custom Callbacks

Create custom callbacks by inheriting from `pl.Callback`:

```python
import pytorch_lightning as pl

class MyCustomCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Your custom logic here
        pass
```

## Best Practices

1. **Use visualization callbacks**: Monitor training visually
2. **Save checkpoints**: Regular model saving
3. **Early stopping**: Prevent overfitting
4. **Learning rate monitoring**: Track LR changes
5. **Custom metrics**: Log domain-specific metrics
