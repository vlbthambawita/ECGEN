"""
Datasets returning dict samples; importing this package registers them in
``ecgen.registry.DATASETS`` (``mimic_iv_ecg``, ``synthetic``).
"""

from ecgen.data.datasets.base import BaseECGDataset, scale_signal
from ecgen.data.datasets.mimic import MIMICIVECG
from ecgen.data.datasets.synthetic import SyntheticECG

__all__ = ["BaseECGDataset", "MIMICIVECG", "SyntheticECG", "scale_signal"]
