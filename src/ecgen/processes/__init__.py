"""
Corruption processes for continuous generative models; importing this package
registers them in ``ecgen.registry.PROCESSES`` (``ddpm``).
"""

from ecgen.processes.base import Inpainting, Process, TrainingPair, cfg_model_fn
from ecgen.processes.ddpm import DDPMProcess

__all__ = ["DDPMProcess", "Inpainting", "Process", "TrainingPair", "cfg_model_fn"]
