"""
Observation masks for partial-ECG conditioning (lead reconstruction,
imputation, forecasting).

A mask generator is called once per sample with the signal shape and returns
an ``obs_mask`` ``[C, T]`` float tensor, 1 = observed. The data module attaches
it to the batch (``ECGBatch.obs_mask``); models condition on
``signal * obs_mask`` and (optionally) compute the loss on the unobserved part.

Config (``data.masks``)::

    masks: {name: lead_subset, params: {keep: [[I], [I, II], [II, V1]]}}
    masks:
      name: mixture
      params:
        components:
          - {weight: 0.5, name: lead_subset, params: {keep: [[I]]}}
          - {weight: 0.3, name: time_span, params: {min_frac: 0.1, max_frac: 0.5}}
          - {weight: 0.2, name: none}
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from ecgen.registry import MASKS


class MaskGenerator:
    """Base class; subclasses implement :meth:`__call__`."""

    def __init__(self, lead_names: Sequence[str]) -> None:
        self.lead_names = list(lead_names)

    def __call__(
        self, num_leads: int, length: int, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        raise NotImplementedError


def _rand(generator: Optional[torch.Generator]) -> float:
    return float(torch.rand((), generator=generator))


def _randint(low: int, high: int, generator: Optional[torch.Generator]) -> int:
    """Uniform integer in [low, high]."""
    return int(torch.randint(low, high + 1, (), generator=generator))


@MASKS.register("none")
class FullyObserved(MaskGenerator):
    """Everything observed (also: the "unconditional" branch of a mixture)."""

    def __call__(self, num_leads, length, generator=None):
        return torch.ones(num_leads, length)


@MASKS.register("unobserved")
class FullyUnobserved(MaskGenerator):
    """Nothing observed: plain generation, when mixed with partial masks."""

    def __call__(self, num_leads, length, generator=None):
        return torch.zeros(num_leads, length)


@MASKS.register("lead_subset")
class LeadSubset(MaskGenerator):
    """
    Observe whole leads.

    Args:
        keep: List of lead-name lists; one is chosen uniformly per sample
            (e.g. ``[[I], [I, II]]`` = "single lead I" or "leads I and II").
            If omitted, a random subset of ``min_leads..max_leads`` leads is kept.
        min_leads: Smallest random subset.
        max_leads: Largest random subset.
    """

    def __init__(
        self,
        lead_names: Sequence[str],
        keep: Optional[Sequence[Sequence[str]]] = None,
        min_leads: int = 1,
        max_leads: int = 3,
    ) -> None:
        super().__init__(lead_names)
        self.keep: Optional[List[List[int]]] = None
        if keep is not None:
            self.keep = []
            for names in keep:
                unknown = [n for n in names if n not in self.lead_names]
                if unknown:
                    raise ValueError(
                        f"lead_subset keeps unknown leads {unknown}; data has {self.lead_names}"
                    )
                self.keep.append([self.lead_names.index(n) for n in names])
        self.min_leads, self.max_leads = min_leads, max_leads

    def __call__(self, num_leads, length, generator=None):
        mask = torch.zeros(num_leads, length)
        if self.keep is not None:
            idx = self.keep[_randint(0, len(self.keep) - 1, generator)]
        else:
            k = _randint(self.min_leads, min(self.max_leads, num_leads), generator)
            idx = torch.randperm(num_leads, generator=generator)[:k].tolist()
        mask[idx] = 1.0
        return mask


@MASKS.register("time_span")
class TimeSpan(MaskGenerator):
    """
    Hide ``n_spans`` contiguous segments (all leads) covering ``min_frac..max_frac``
    of the record in total — imputation.
    """

    def __init__(
        self,
        lead_names: Sequence[str],
        min_frac: float = 0.1,
        max_frac: float = 0.5,
        n_spans: int = 1,
    ) -> None:
        super().__init__(lead_names)
        self.min_frac, self.max_frac, self.n_spans = min_frac, max_frac, n_spans

    def __call__(self, num_leads, length, generator=None):
        mask = torch.ones(num_leads, length)
        frac = self.min_frac + (self.max_frac - self.min_frac) * _rand(generator)
        span = max(1, int(round(frac * length / self.n_spans)))
        for _ in range(self.n_spans):
            start = _randint(0, length - span, generator)
            mask[:, start : start + span] = 0.0
        return mask


@MASKS.register("forecast")
class Forecast(MaskGenerator):
    """Observe the first ``min_frac..max_frac`` of the record; predict the rest."""

    def __init__(
        self, lead_names: Sequence[str], min_frac: float = 0.25, max_frac: float = 0.75
    ) -> None:
        super().__init__(lead_names)
        self.min_frac, self.max_frac = min_frac, max_frac

    def __call__(self, num_leads, length, generator=None):
        frac = self.min_frac + (self.max_frac - self.min_frac) * _rand(generator)
        mask = torch.zeros(num_leads, length)
        mask[:, : int(round(frac * length))] = 1.0
        return mask


@MASKS.register("random_points")
class RandomPoints(MaskGenerator):
    """Each sample point observed independently with probability ``p_observed``."""

    def __init__(self, lead_names: Sequence[str], p_observed: float = 0.5) -> None:
        super().__init__(lead_names)
        self.p = p_observed

    def __call__(self, num_leads, length, generator=None):
        return (torch.rand(num_leads, length, generator=generator) < self.p).float()


@MASKS.register("mixture")
class Mixture(MaskGenerator):
    """Pick one component per sample with probability proportional to ``weight``."""

    def __init__(
        self, lead_names: Sequence[str], components: Sequence[Mapping[str, Any]]
    ) -> None:
        super().__init__(lead_names)
        if not components:
            raise ValueError("mixture needs at least one component")
        self.components = [build_mask(c, lead_names) for c in components]
        w = torch.tensor([float(c.get("weight", 1.0)) for c in components])
        self.probs = w / w.sum()

    def __call__(self, num_leads, length, generator=None):
        i = int(torch.multinomial(self.probs, 1, generator=generator))
        return self.components[i](num_leads, length, generator)


def build_mask(
    spec: Optional[Mapping[str, Any]], lead_names: Sequence[str]
) -> Optional[MaskGenerator]:
    """Build a mask generator from ``{name | target, params}`` (``weight`` is ignored)."""
    if not spec:
        return None
    spec = {k: v for k, v in spec.items() if k != "weight"}
    return MASKS.build(spec, lead_names=lead_names)
