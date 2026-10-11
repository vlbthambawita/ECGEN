"""Continuous-time generative families: diffusion (and flow matching, same module)."""

from ecgen.families.continuous.module import (
    ContinuousConfig,
    ContinuousGenerativeModule,
    DiffusionModule,
)

__all__ = ["ContinuousConfig", "ContinuousGenerativeModule", "DiffusionModule"]
