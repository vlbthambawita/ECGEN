"""
Corruption processes for continuous generative models (diffusion, flow matching).

A :class:`Process` defines how training pairs are made and how samples are
drawn; the family module (:class:`ecgen.families.continuous.ContinuousGenerativeModule`)
owns the network, conditioning, EMA, and optimisation, so adding a new
diffusion/flow method means adding a ``Process``, not a family.

``model_fn(x, t)`` passed to :meth:`Process.sample` returns the raw network
output at time ``t`` (``[B]``, in the process's own time units); guidance is
applied by the caller (see :func:`cfg_model_fn`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

ModelFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class TrainingPair:
    """What one training step needs: the network input, its target, and per-sample weights."""

    xt: Tensor  # corrupted signal [B, C, T]
    t: Tensor  # network time input [B]
    target: Tensor  # regression target [B, C, T]
    weight: Tensor  # per-sample loss weight [B]


@dataclass
class Inpainting:
    """
    Training-free inpainting at sampling time.

    Attributes:
        obs: Observed signal ``[B, C, T]``.
        mask: 1 = observed.
        mode: ``"replace"`` (overwrite the observed part with a correspondingly
            noised observation after every step), ``"repaint"`` (replace, plus
            ``jumps`` resampling round trips per step; Lugmayr et al., 2022), or
            ``"clean"`` (overwrite with the clean observation; for models trained
            with ``clean_observed``).
        jumps: RePaint resampling count per step.
    """

    obs: Tensor
    mask: Tensor
    mode: str = "replace"
    jumps: int = 5


class Process(nn.Module):
    """Base class; see the module docstring."""

    #: What the network predicts (``"eps"``, ``"v"``, ``"x0"``, ``"velocity"``).
    prediction: str

    def training_pair(self, x0: Tensor, noise: Tensor) -> TrainingPair:
        raise NotImplementedError

    def sample(
        self,
        model_fn: ModelFn,
        shape: tuple[int, ...],
        steps: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        inpaint: Optional[Inpainting] = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError


def cfg_model_fn(
    model_fn_cond: ModelFn, model_fn_uncond: Optional[ModelFn], guidance: float
) -> ModelFn:
    """Classifier-free guidance: ``u + w (c - u)`` on the network output."""
    if model_fn_uncond is None or guidance == 1.0:
        return model_fn_cond

    def fn(x: Tensor, t: Tensor) -> Tensor:
        c = model_fn_cond(x, t)
        u = model_fn_uncond(x, t)
        return u + guidance * (c - u)

    return fn


def randn(shape, device, generator: Optional[torch.Generator] = None) -> Tensor:
    """Gaussian noise on ``device``, drawn from ``generator`` (on its own device) if given."""
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=generator, device=generator.device).to(device)
