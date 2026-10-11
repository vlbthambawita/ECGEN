"""
Flow matching / rectified flow (Lipman et al., 2023; Liu et al., 2023).

Time runs from ``t = 0`` (noise ``x0 ~ N(0, I)``) to ``t = 1`` (data ``x1``):

    x_t = t * x1 + (1 - (1 - sigma_min) * t) * x0,   target velocity = x1 - (1 - sigma_min) * x0

and sampling integrates ``dx/dt = v(x, t)`` with a fixed-step ODE solver.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ecgen.processes.base import Inpainting, ModelFn, Process, TrainingPair, randn
from ecgen.registry import PROCESSES


@PROCESSES.register("rectified_flow")
class RectifiedFlowProcess(Process):
    """
    Args:
        t_sampling: ``"uniform"`` or ``"logit_normal"`` (Esser et al., 2024).
        logit_mean: Mean of the logit-normal time distribution.
        logit_std: Standard deviation of the logit-normal time distribution.
        sigma_min: Width of the conditional path at ``t = 1`` (0 = rectified flow).
        ot_coupling: Pair noise and data within each batch by optimal transport
            (minibatch OT-CFM, Tong et al., 2024) for straighter paths.
        time_scale: The network sees ``t * time_scale`` (matches the 0..1000 range
            the sinusoidal embedding was tuned for in DDPM).
        solver: Default ODE solver, ``"euler"`` or ``"heun"``.
    """

    prediction = "velocity"

    def __init__(
        self,
        t_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        sigma_min: float = 0.0,
        ot_coupling: bool = False,
        time_scale: float = 1000.0,
        solver: str = "euler",
    ) -> None:
        super().__init__()
        if t_sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                f"t_sampling must be 'uniform' or 'logit_normal', got {t_sampling!r}"
            )
        if solver not in ("euler", "heun"):
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}")
        self.t_sampling = t_sampling
        self.logit_mean, self.logit_std = logit_mean, logit_std
        self.sigma_min = sigma_min
        self.ot_coupling = ot_coupling
        self.time_scale = time_scale
        self.solver = solver

    # -------------------------------------------------------------- training

    def sample_t(self, b: int, device) -> Tensor:
        if self.t_sampling == "uniform":
            return torch.rand(b, device=device)
        return torch.sigmoid(
            self.logit_mean + self.logit_std * torch.randn(b, device=device)
        )

    def interpolate(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        tt = t.view(-1, *([1] * (x1.dim() - 1)))
        return tt * x1 + (1 - (1 - self.sigma_min) * tt) * x0

    def training_pair(self, x1: Tensor, noise: Tensor) -> TrainingPair:
        if self.ot_coupling and x1.shape[0] > 1:
            noise = noise[ot_pairing(x1, noise)]
        t = self.sample_t(x1.shape[0], x1.device)
        return TrainingPair(
            xt=self.interpolate(x1, noise, t),
            t=t * self.time_scale,
            target=x1 - (1 - self.sigma_min) * noise,
            weight=torch.ones_like(t),
        )

    # -------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        shape: tuple[int, ...],
        steps: int = 50,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        inpaint: Optional[Inpainting] = None,
        solver: Optional[str] = None,
        x_init: Optional[Tensor] = None,
        t_start: float = 0.0,
        **kwargs,
    ) -> Tensor:
        """
        Integrate from noise (``t = t_start``) to data (``t = 1``) in ``steps`` steps.

        Inpainting: ``"replace"`` overwrites the observed part with the point on
        the straight path from the initial noise to the observation; ``"clean"``
        with the observation itself. (RePaint is a diffusion method; use ``replace``.)
        ``x_init`` / ``t_start`` start mid-path (SDEdit-style editing).
        """
        solver = solver or self.solver
        if inpaint is not None and inpaint.mode == "repaint":
            raise ValueError("RePaint is not defined for flows; use inpaint='replace'")
        noise = randn(shape, device, generator)
        x = noise if x_init is None else x_init
        ts = torch.linspace(t_start, 1.0, steps + 1, device=device)

        def v(x_: Tensor, t: Tensor) -> Tensor:
            return model_fn(
                x_, torch.full((shape[0],), float(t) * self.time_scale, device=device)
            )

        def known(t: Tensor) -> Optional[Tensor]:
            if inpaint is None:
                return None
            if inpaint.mode == "clean":
                return inpaint.obs
            return self.interpolate(inpaint.obs, noise, t.expand(shape[0]))

        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            h = t_next - t
            if inpaint is not None:
                m = inpaint.mask.to(x.dtype)
                x = m * known(t) + (1 - m) * x
            d = v(x, t)
            if solver == "heun" and i < steps - 1:
                x_euler = x + h * d
                x = x + h * 0.5 * (d + v(x_euler, t_next))
            else:
                x = x + h * d
        if inpaint is not None:
            m = inpaint.mask.to(x.dtype)
            x = m * inpaint.obs + (1 - m) * x
        return x


def ot_pairing(x1: Tensor, x0: Tensor) -> Tensor:
    """Permutation of ``x0`` minimising the summed squared distance to ``x1`` (exact assignment)."""
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x1.flatten(1).float(), x0.flatten(1).float()) ** 2
    _, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(cols, device=x0.device)
