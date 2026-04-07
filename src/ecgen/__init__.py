"""
Top-level package for ECG generation experiments.
"""

__version__ = "0.1.0"

from ecgen.generate import generate
from ecgen.plot import plot_ecg

__all__ = ["data", "f", "generate", "models", "plot_ecg", "training", "evaluation", "utils"]
