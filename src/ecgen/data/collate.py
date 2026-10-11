"""
Collation of dict samples into :class:`ecgen.types.ECGBatch`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ecgen.data.masks import MaskGenerator
from ecgen.types import ECGBatch


def _stack(values: List[Any]) -> Any:
    """Stack tensors; keep anything else (e.g. report strings) as a list."""
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    return list(values)


class ECGCollate:
    """
    ``collate_fn`` that stacks dict samples into an :class:`ECGBatch`.

    Tensors under ``signal``, ``cond``, ``cond_missing`` and ``obs_mask`` are
    stacked; ``meta`` values are gathered into lists. With a ``mask_generator``,
    each sample without its own ``obs_mask`` gets one drawn from it.
    """

    def __init__(
        self,
        lead_names: Sequence[str],
        fs: int,
        mask_generator: Optional[MaskGenerator] = None,
    ) -> None:
        self.lead_names = list(lead_names)
        self.fs = int(fs)
        self.mask_generator = mask_generator

    def __call__(self, samples: List[Dict[str, Any]]) -> ECGBatch:
        if self.mask_generator is not None:
            for s in samples:
                if s.get("obs_mask") is None:
                    c, t = s["signal"].shape
                    s["obs_mask"] = self.mask_generator(c, t)
        first = samples[0]
        cond = {
            k: _stack([s["cond"][k] for s in samples]) for k in first.get("cond", {})
        }
        cond_missing = {
            k: torch.stack([s["cond_missing"][k] for s in samples])
            for k in first.get("cond_missing", {})
        }
        obs_mask = (
            torch.stack([s["obs_mask"] for s in samples])
            if first.get("obs_mask") is not None
            else None
        )
        meta = {k: [s["meta"][k] for s in samples] for k in first.get("meta", {})}
        return ECGBatch(
            signal=torch.stack([s["signal"] for s in samples]),
            lead_names=list(self.lead_names),
            fs=self.fs,
            cond=cond,
            cond_missing=cond_missing,
            obs_mask=obs_mask,
            meta=meta,
        )
