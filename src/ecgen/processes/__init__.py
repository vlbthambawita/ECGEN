"""
Corruption processes for continuous generative models; importing this package
registers them in ``ecgen.registry.PROCESSES`` (``ddpm``, ``rectified_flow``).
"""

from ecgen.processes.base import Inpainting, Process, TrainingPair, cfg_model_fn
from ecgen.processes.ddpm import DDPMProcess
from ecgen.processes.flow import RectifiedFlowProcess, ot_pairing

__all__ = [
    "DDPMProcess",
    "Inpainting",
    "Process",
    "RectifiedFlowProcess",
    "TrainingPair",
    "cfg_model_fn",
    "ot_pairing",
]
