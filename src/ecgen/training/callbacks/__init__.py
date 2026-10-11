"""
Training callbacks; importing this package registers them in
``ecgen.registry.CALLBACKS`` together with the common Lightning callbacks.
"""

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ecgen.registry import CALLBACKS
from ecgen.training.callbacks.visualization import (
    ECGVisualizationCallback,
    GeneratedSamplesCallback,
    VAEVisualizationCallback,
)

CALLBACKS.add("ecg_visualization", ECGVisualizationCallback)
CALLBACKS.add("generated_samples", GeneratedSamplesCallback)
CALLBACKS.add("vae_visualization", VAEVisualizationCallback)
CALLBACKS.add("early_stopping", EarlyStopping)
CALLBACKS.add("lr_monitor", LearningRateMonitor)
CALLBACKS.add("model_checkpoint", ModelCheckpoint)

__all__ = [
    "ECGVisualizationCallback",
    "GeneratedSamplesCallback",
    "VAEVisualizationCallback",
]
