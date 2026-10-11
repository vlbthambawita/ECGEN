"""
Training for ECG generation models.

``ecgen.training.runner`` is the YAML-driven entry point (``ecgen train``).
Callbacks are imported lazily because they need matplotlib.
"""

from typing import Any

__all__ = [
    "ECGVisualizationCallback",
    "GeneratedSamplesCallback",
    "VAEVisualizationCallback",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ecgen.training import callbacks

        return getattr(callbacks, name)
    raise AttributeError(f"module 'ecgen.training' has no attribute {name!r}")
