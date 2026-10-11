"""
Collation of dict samples into :class:`ecgen.types.ECGBatch`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from ecgen.types import ECGBatch


class ECGCollate:
    """
    ``collate_fn`` that stacks dict samples into an :class:`ECGBatch`.

    Tensors under ``signal``, ``cond``, ``cond_missing`` and ``obs_mask`` are
    stacked; ``meta`` values are gathered into lists.
    """

    def __init__(self, lead_names: Sequence[str], fs: int) -> None:
        self.lead_names = list(lead_names)
        self.fs = int(fs)

    def __call__(self, samples: List[Dict[str, Any]]) -> ECGBatch:
        first = samples[0]
        cond = {
            k: torch.stack([s["cond"][k] for s in samples])
            for k in first.get("cond", {})
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
