"""
Evaluation of generated ECGs: measurements, distribution metrics, condition
adherence, reconstruction, lead consistency, memorisation, and a small
classifier for label adherence and TSTR.
"""

from ecgen.evaluation.classifier import (
    ResNet1dClassifier,
    fit_classifier,
    label_adherence,
    label_metrics,
    predict,
    tstr,
)
from ecgen.evaluation.features import handcrafted_features, module_features
from ecgen.evaluation.measure import adherence, detect_r_peaks, heart_rate
from ecgen.evaluation.metrics import (
    distribution_metrics,
    einthoven_residual,
    frechet_distance,
    inter_lead_correlation_error,
    memorization_ratio,
    mmd_rbf,
    nearest_neighbour_distances,
    precision_recall,
    reconstruction_metrics,
)

__all__ = [
    "ResNet1dClassifier",
    "adherence",
    "detect_r_peaks",
    "distribution_metrics",
    "einthoven_residual",
    "fit_classifier",
    "frechet_distance",
    "handcrafted_features",
    "heart_rate",
    "inter_lead_correlation_error",
    "label_adherence",
    "label_metrics",
    "memorization_ratio",
    "mmd_rbf",
    "module_features",
    "nearest_neighbour_distances",
    "precision_recall",
    "predict",
    "reconstruction_metrics",
    "tstr",
]
