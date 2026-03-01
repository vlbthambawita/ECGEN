# Training

Training utilities, callbacks, losses, and metrics.

## Structure

- `callbacks/` - Training callbacks
  - `visualization.py` - Visualization callbacks (ECG, VAE, GAN)
  - `checkpointing.py` - Custom checkpoint callbacks
  - `logging.py` - Custom logging callbacks
- `losses/` - Loss functions
  - `vae_losses.py` - VAE-specific losses
  - `gan_losses.py` - GAN losses
  - `common.py` - Shared loss functions
- `metrics/` - Evaluation metrics
  - `reconstruction.py` - Reconstruction metrics
  - `generation.py` - Generation quality metrics
  - `clinical.py` - Clinical metrics
- `trainers/` - Training loops
  - `base_trainer.py` - Base trainer class
  - `vae_trainer.py` - VAE training
  - `gan_trainer.py` - GAN training

## Usage

```python
from training.callbacks.visualization import VAEVisualizationCallback
from training.losses.vae_losses import vae_loss
from training.metrics.clinical import calculate_clinical_metrics
```
