"""
One LightningModule for every continuous-time generative method (diffusion,
flow matching): the method is a :class:`~ecgen.processes.Process`, the network
a registered backbone, the conditions a :class:`~ecgen.conditioning.CompositeConditioner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

import ecgen.nn.backbones  # noqa: F401  (registers backbones)
import ecgen.processes  # noqa: F401  (registers processes)
from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.processes.base import Inpainting, cfg_model_fn
from ecgen.registry import BACKBONES, FAMILIES, PROCESSES
from ecgen.training.ema import EMA
from ecgen.types import ConditionOutput
from ecgen.utils.config import resolve_config


@dataclass
class ContinuousConfig:
    """
    Attributes:
        in_channels: Leads.
        process: ``{name, params}`` from ``ecgen.registry.PROCESSES`` (``ddpm``, ...).
        backbone: ``{name, params}`` from ``ecgen.registry.BACKBONES`` (``unet1d``, ...);
            channel and condition sizes are filled in.
        conditioning: See :mod:`ecgen.conditioning` (``None`` = unconditional).
        lr, weight_decay, b1, b2: AdamW.
        warmup_steps: Linear learning-rate warm-up.
        ema_decay: EMA of the weights used for sampling (0 = off).
        loss_on: ``"all"`` or ``"unobserved"`` (with ``data.masks``): loss only where
            ``obs_mask`` is 0.
        clean_observed: SSSD-style partial-ECG training: the observed entries of the
            noisy input are replaced by the clean signal (and again at every
            sampling step).
        sample_steps: Default sampling steps.
        guidance: Default classifier-free guidance scale (1 = off).
    """

    in_channels: int = 12
    process: Dict[str, Any] = field(default_factory=lambda: {"name": "ddpm"})
    backbone: Dict[str, Any] = field(default_factory=lambda: {"name": "unet1d"})
    conditioning: Optional[Dict[str, Any]] = None
    lr: float = 2e-4
    weight_decay: float = 0.0
    b1: float = 0.9
    b2: float = 0.999
    warmup_steps: int = 500
    ema_decay: float = 0.999
    loss_on: str = "all"
    clean_observed: bool = False
    sample_steps: int = 50
    guidance: float = 1.0


class ContinuousGenerativeModule(GenerativeModule):
    """
    Diffusion / flow-matching family. Registered as ``diffusion`` (default
    process ``ddpm``) and ``flow_matching`` (default process ``rectified_flow``).

    Training: draw noise, let the process build ``(x_t, t, target, weight)``,
    regress the backbone output on the target. Sampling: the process integrates
    the backbone from noise, with optional classifier-free guidance, and
    partial-ECG conditioning either learned (``masked_signal`` encoder,
    ``clean_observed``) or training-free (``inpaint="replace" | "repaint"``).
    """

    family = "continuous"
    supports = Capabilities(conditional=True, cfg=True, inpainting=True)
    accepts = frozenset({"vector", "tokens", "dense"})
    config_cls: type = ContinuousConfig

    def __init__(
        self, config: ContinuousConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        config = resolve_config(self.config_cls, config, kwargs)
        if config.loss_on not in ("all", "unobserved"):
            raise ValueError(
                f"loss_on must be 'all' or 'unobserved', got {config.loss_on!r}"
            )
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.init_conditioning(config.conditioning)
        sizes = {"cond_dim": 0, "context_dim": 0, "dense_channels": 0}
        if self.conditioner is not None:
            slots = self.conditioner.slots
            sizes["cond_dim"] = self.conditioner.vector_dim if "vector" in slots else 0
            sizes["context_dim"] = (
                self.conditioner.token_dim if "tokens" in slots else 0
            )
            sizes["dense_channels"] = self.conditioner.dense_channels
            backbone_cls = BACKBONES.resolve(config.backbone)
            extra = slots - set(getattr(backbone_cls, "accepts", self.accepts))
            if extra:
                raise ValueError(
                    f"Backbone {backbone_cls.__name__} cannot consume condition slots {sorted(extra)}"
                )
        self.process = PROCESSES.build(config.process)
        self.backbone = BACKBONES.build(
            config.backbone, in_channels=self._model_channels(config), **sizes
        )
        self.ema = (
            EMA([self.backbone, self.conditioner], config.ema_decay)
            if config.ema_decay > 0
            else None
        )

    # ------------------------------------------------- model space (hooks)

    def _model_channels(self, config: ContinuousConfig) -> int:
        """Channels of the space the process runs in (latent subclasses override)."""
        return config.in_channels

    def to_model_space(self, x: Tensor) -> Tensor:
        """Signal ``[B, C, T]`` → the space the process runs in (identity here)."""
        return x

    def from_model_space(self, z: Tensor) -> Tensor:
        """Inverse of :meth:`to_model_space`."""
        return z

    def model_space_shape(self, n: int, seq_length: int) -> tuple[int, ...]:
        return (n, self.config.in_channels, seq_length)

    def adapt_condition(
        self, c: Optional[ConditionOutput], length: int
    ) -> Optional[ConditionOutput]:
        """Resample dense conditions to the model-space length."""
        if c is None or c.dense is None or c.dense.shape[-1] == length:
            return c
        dense = torch.nn.functional.adaptive_avg_pool1d(c.dense, length)
        return ConditionOutput(c.vector, c.tokens, c.tokens_mask, dense)

    # ------------------------------------------------------------- training

    def _loss(self, batch: Any) -> Tensor:
        signal = self.get_signal(batch)
        x0 = self.to_model_space(signal)
        cond = self.adapt_condition(self.condition(batch), x0.shape[-1])
        obs_mask = getattr(batch, "obs_mask", None)
        if x0.shape != signal.shape:
            obs_mask = None  # masks live in signal space
        pair = self.process.training_pair(x0, torch.randn_like(x0))
        xt = pair.xt
        if self.config.clean_observed and obs_mask is not None:
            m = obs_mask.to(x0.dtype)
            xt = m * x0 + (1 - m) * xt
        out = self.backbone(xt, pair.t, cond)
        err = (out - pair.target) ** 2
        if self.config.loss_on == "unobserved" and obs_mask is not None:
            w = 1.0 - obs_mask.to(err.dtype)
            per_sample = (err * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(
                1.0
            )
        else:
            per_sample = err.flatten(1).mean(1)
        return (pair.weight * per_sample).mean()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update()

    def configure_optimizers(self) -> Any:
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
            weight_decay=self.config.weight_decay,
        )
        warmup = max(1, self.config.warmup_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: min(1.0, (step + 1) / warmup)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # ----------------------------------------------------------- checkpoint

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

    # ------------------------------------------------------------- sampling

    def _model_fn(self, c: Optional[ConditionOutput], length: int):
        c = self.adapt_condition(c, length)
        return lambda x, t: self.backbone(x, t, c)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        seq_length: Optional[int] = None,
        cond: Optional[Mapping[str, Any]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        guidance: Optional[float] = None,
        steps: Optional[int] = None,
        inpaint: Optional[str] = None,
        jumps: int = 5,
        paste_observed: bool = True,
        use_ema: bool = True,
        seed: Optional[int] = None,
        **process_kwargs: Any,
    ) -> Tensor:
        """
        Generate ``[n, C, T]`` ECGs.

        Args:
            n_samples: Number of ECGs.
            seq_length: Length; defaults to the training data's.
            cond: Raw condition values (see :meth:`prepare_condition`).
            obs, obs_mask: Partially observed ECG and mask (1 = observed). Used by a
                ``masked_signal`` condition, ``clean_observed`` models, and ``inpaint``.
            guidance: Classifier-free guidance scale (default ``config.guidance``).
            steps: Sampling steps (default ``config.sample_steps``).
            inpaint: Training-free inpainting for models without a learned observation
                condition: ``"replace"`` or ``"repaint"`` (needs ``obs``).
            jumps: RePaint resampling count.
            paste_observed: Copy the observed part of ``obs`` into the output.
            use_ema: Sample with the EMA weights.
            seed: Seed for the sampling noise.
            process_kwargs: Passed to the process sampler (e.g. ``eta`` for DDIM,
                ``solver`` for flows).
        """
        self.eval()
        seq_length = seq_length or self.data_info.get("seq_length")
        if seq_length is None:
            raise ValueError("seq_length is unknown; pass seq_length=")
        guidance = self.config.guidance if guidance is None else guidance
        steps = steps or self.config.sample_steps
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        if obs is not None:
            obs = obs.to(self.device)
            obs_mask = (
                torch.ones_like(obs)
                if obs_mask is None
                else obs_mask.to(self.device, obs.dtype)
            )

        shape = self.model_space_shape(n_samples, seq_length)
        latent = shape[1:] != (self.config.in_channels, seq_length)
        if latent and (inpaint is not None or self.config.clean_observed):
            raise ValueError("Training-free inpainting works in signal space only")
        inpainting = None
        if inpaint is not None:
            if obs is None:
                raise ValueError("inpaint needs obs (and obs_mask)")
            inpainting = Inpainting(obs, obs_mask, inpaint, jumps)
        elif self.config.clean_observed and obs is not None:
            inpainting = Inpainting(obs, obs_mask, "clean")

        scope = (
            self.ema.swap() if (use_ema and self.ema is not None) else _nullcontext()
        )
        # obs feeds the conditioner only when it has an observation encoder;
        # otherwise it is used for inpainting / pasting only
        c_obs, c_mask = (
            (obs, obs_mask)
            if (self.conditioner is not None and self.conditioner.uses_obs)
            else (None, None)
        )
        with scope:
            c = self.encode_condition(n_samples, cond, c_obs, c_mask)
            uncond_fn = None
            if c is not None and guidance != 1.0:
                u = self.encode_condition(n_samples, cond, c_obs, c_mask, drop="all")
                uncond_fn = self._model_fn(u, shape[-1])
            model_fn = cfg_model_fn(self._model_fn(c, shape[-1]), uncond_fn, guidance)
            x = self.process.sample(
                model_fn,
                shape,
                steps=steps,
                device=self.device,
                generator=generator,
                inpaint=inpainting,
                **process_kwargs,
            )
            x = self.from_model_space(x)
        if paste_observed and obs is not None:
            x = obs_mask * obs + (1 - obs_mask) * x
        return x


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@FAMILIES.register("diffusion")
class DiffusionModule(ContinuousGenerativeModule):
    """Gaussian diffusion (default process ``ddpm``); see :class:`ContinuousGenerativeModule`."""

    family = "diffusion"


@dataclass
class FlowMatchingConfig(ContinuousConfig):
    """:class:`ContinuousConfig` with the rectified-flow process as default."""

    process: Dict[str, Any] = field(default_factory=lambda: {"name": "rectified_flow"})


@FAMILIES.register("flow_matching")
class FlowMatchingModule(ContinuousGenerativeModule):
    """Flow matching (default process ``rectified_flow``); see :class:`ContinuousGenerativeModule`."""

    family = "flow_matching"
    config_cls = FlowMatchingConfig
