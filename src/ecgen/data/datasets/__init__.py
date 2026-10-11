"""
Datasets returning dict samples; importing this package registers them in
``ecgen.registry.DATASETS`` (``mimic_iv_ecg``, ``ptbxl``, ``synthetic``).
"""

from ecgen.data.datasets.base import BaseECGDataset, scale_signal
from ecgen.data.datasets.mimic import MIMICIVECG
from ecgen.data.datasets.ptbxl import PTBXL
from ecgen.data.datasets.synthetic import SyntheticECG

__all__ = ["BaseECGDataset", "MIMICIVECG", "PTBXL", "SyntheticECG", "scale_signal"]
