"""
Discrete-time Gaussian diffusion (DDPM, Ho et al., 2020) with DDIM sampling.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from ecgen.processes.base import Inpainting, ModelFn, Process, TrainingPair, randn
from ecgen.registry import PROCESSES


def _schedule(name: str, steps: int) -> Tensor:
    """``alphas_cumprod`` for a beta schedule."""
    if name == "linear":
        betas = torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
        return torch.cumprod(1 - betas, dim=0)
    if name == "cosine":  # Nichol & Dhariwal, 2021
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
        ac = f / f[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        return torch.cumprod(1 - betas, dim=0)
    raise ValueError(f"Unknown schedule {name!r}; use 'linear' or 'cosine'")


@PROCESSES.register("ddpm")
class DDPMProcess(Process):
    """
    Args:
        num_steps: Diffusion steps ``T``.
        schedule: ``"cosine"`` or ``"linear"``.
        prediction: Network target, ``"eps"``, ``"v"`` (Salimans & Ho, 2022) or ``"x0"``.
        min_snr_gamma: Min-SNR loss weighting (Hang et al., 2023); ``None`` = uniform.
        clip_x0: Clamp the predicted ``x0`` to ``[-clip_x0, clip_x0]`` while sampling
            (``None`` = no clamp; z-scored ECGs reach about ±10).
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "cosine",
        prediction: str = "v",
        min_snr_gamma: Optional[float] = 5.0,
        clip_x0: Optional[float] = None,
    ) -> None:
        super().__init__()
        if prediction not in ("eps", "v", "x0"):
            raise ValueError(f"prediction must be eps, v, or x0; got {prediction!r}")
        self.num_steps = num_steps
        self.prediction = prediction
        self.min_snr_gamma = min_snr_gamma
        self.clip_x0 = clip_x0
        self.register_buffer(
            "alphas_cumprod", _schedule(schedule, num_steps).float(), persistent=False
        )

    # ------------------------------------------------------------- helpers

    def _ac(self, t: Tensor) -> Tensor:
        return self.alphas_cumprod[t].view(-1, 1, 1)

    def to_x0_eps(self, out: Tensor, xt: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Network output at integer steps ``t`` → (predicted x0, predicted noise)."""
        ac = self._ac(t)
        a, s = ac.sqrt(), (1 - ac).sqrt()
        if self.prediction == "eps":
            eps = out
            x0 = (xt - s * eps) / a
        elif self.prediction == "x0":
            x0 = out
            eps = (xt - a * x0) / s
        else:  # v = a * eps - s * x0
            x0 = a * xt - s * out
            eps = s * xt + a * out
        if self.clip_x0 is not None:
            x0 = x0.clamp(-self.clip_x0, self.clip_x0)
            eps = (xt - a * x0) / s
        return x0, eps

    def q_sample(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        ac = self._ac(t)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    # -------------------------------------------------------------- training

    def training_pair(self, x0: Tensor, noise: Tensor) -> TrainingPair:
        b = x0.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=x0.device)
        xt = self.q_sample(x0, noise, t)
        ac = self._ac(t)
        if self.prediction == "eps":
            target = noise
        elif self.prediction == "x0":
            target = x0
        else:
            target = ac.sqrt() * noise - (1 - ac).sqrt() * x0

        weight = torch.ones(b, device=x0.device)
        if self.min_snr_gamma is not None:
            snr = (ac / (1 - ac)).view(-1)
            clipped = snr.clamp(max=self.min_snr_gamma)
            weight = {"eps": clipped / snr, "v": clipped / (snr + 1), "x0": clipped}[
                self.prediction
            ]
        return TrainingPair(xt=xt, t=t.float(), target=target, weight=weight)

    # -------------------------------------------------------------- sampling

    def timesteps(self, steps: int) -> list[int]:
        """Evenly spaced steps from ``T - 1`` down to 0."""
        steps = max(1, min(steps, self.num_steps))
        return sorted(
            {
                int(round(x))
                for x in torch.linspace(0, self.num_steps - 1, steps).tolist()
            },
            reverse=True,
        )

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        shape: tuple[int, ...],
        steps: int = 50,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        inpaint: Optional[Inpainting] = None,
        eta: float = 0.0,
        x_init: Optional[Tensor] = None,
        start_step: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        DDIM sampling (Song et al., 2021) over ``steps`` evenly spaced steps;
        ``eta=1`` is ancestral (DDPM-like) sampling, ``eta=0`` deterministic.

        ``x_init`` / ``start_step`` start from a partially noised signal (SDEdit).
        """
        ts = self.timesteps(steps)
        if start_step is not None:
            ts = [t for t in ts if t <= start_step]
        x = randn(shape, device, generator) if x_init is None else x_init

        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            jumps = (
                inpaint.jumps
                if (inpaint is not None and inpaint.mode == "repaint" and t_prev >= 0)
                else 1
            )
            for j in range(jumps):
                x_prev = self._ddim_step(model_fn, x, t, t_prev, eta, device, generator)
                if inpaint is not None:
                    x_prev = self._replace(x_prev, inpaint, t_prev, device, generator)
                if (
                    j < jumps - 1
                ):  # RePaint: diffuse back from t_prev to t and redo the step
                    x = self._renoise(x_prev, t_prev, t, device, generator)
                else:
                    x = x_prev
        return x

    def _ddim_step(self, model_fn, x, t, t_prev, eta, device, generator) -> Tensor:
        tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        x0, eps = self.to_x0_eps(model_fn(x, tt.float()), x, tt)
        if t_prev < 0:
            return x0
        ac, ac_prev = self.alphas_cumprod[t], self.alphas_cumprod[t_prev]
        sigma = (
            eta * ((1 - ac_prev) / (1 - ac) * (1 - ac / ac_prev)).clamp(min=0).sqrt()
        )
        dir_xt = (1 - ac_prev - sigma**2).clamp(min=0).sqrt() * eps
        x_prev = ac_prev.sqrt() * x0 + dir_xt
        if eta > 0:
            x_prev = x_prev + sigma * randn(x.shape, device, generator)
        return x_prev

    def _replace(self, x: Tensor, inp: Inpainting, t: int, device, generator) -> Tensor:
        """Overwrite the observed region with the observation noised to level ``t``."""
        if t < 0 or inp.mode == "clean":
            known = inp.obs
        else:
            tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            known = self.q_sample(inp.obs, randn(x.shape, device, generator), tt)
        m = inp.mask.to(x.dtype)
        return m * known + (1 - m) * x

    def _renoise(self, x: Tensor, t_from: int, t_to: int, device, generator) -> Tensor:
        """Sample ``q(x_{t_to} | x_{t_from})`` for ``t_to > t_from``."""
        ratio = self.alphas_cumprod[t_to] / self.alphas_cumprod[t_from]
        return ratio.sqrt() * x + (1 - ratio).sqrt() * randn(x.shape, device, generator)
