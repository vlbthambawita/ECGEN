"""
Exponential moving average of model weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class EMA:
    """
    Keeps ``shadow = decay * shadow + (1 - decay) * param`` for the parameters
    of some modules; :meth:`swap` temporarily loads the averaged weights.

    The shadow is a plain dict of tensors (not a submodule), so the owner
    stores it in checkpoints itself (:meth:`state_dict` / :meth:`load_state_dict`).
    """

    def __init__(self, modules: Sequence[nn.Module], decay: float = 0.999) -> None:
        self.decay = decay
        self.modules = [m for m in modules if m is not None]
        self.shadow: Dict[str, Tensor] = {
            name: p.detach().clone() for name, p in self._named_params()
        }

    def _named_params(self) -> Iterator[tuple[str, nn.Parameter]]:
        for i, m in enumerate(self.modules):
            for name, p in m.named_parameters():
                if p.requires_grad:
                    yield f"{i}.{name}", p

    @torch.no_grad()
    def update(self) -> None:
        for name, p in self._named_params():
            s = self.shadow[name]
            if s.device != p.device:
                s = self.shadow[name] = s.to(p.device)
            s.lerp_(p.detach(), 1.0 - self.decay)

    @contextmanager
    def swap(self) -> Iterator[None]:
        """Use the averaged weights inside the ``with`` block."""
        backup = {}
        with torch.no_grad():
            for name, p in self._named_params():
                backup[name] = p.detach().clone()
                p.copy_(self.shadow[name].to(p.device))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in self._named_params():
                    p.copy_(backup[name])

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        missing = set(self.shadow) - set(state)
        if missing:
            raise KeyError(f"EMA state is missing {sorted(missing)[:5]}...")
        self.shadow = {k: state[k].clone() for k in self.shadow}
