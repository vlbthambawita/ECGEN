"""Continuous-time generative families: diffusion (and flow matching, same module)."""

from ecgen.families.continuous.module import (
    ContinuousConfig,
    ContinuousGenerativeModule,
    DiffusionModule,
    FlowMatchingConfig,
    FlowMatchingModule,
)

__all__ = [
    "ContinuousConfig",
    "ContinuousGenerativeModule",
    "DiffusionModule",
    "FlowMatchingConfig",
    "FlowMatchingModule",
]
