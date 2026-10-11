"""
Lead names and lead selection.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

# Standard clinical order of the 12 leads
STANDARD_12: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
# The 8 linearly independent leads; III, aVR, aVL, aVF follow from I and II
INDEPENDENT_8: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

LEAD_ALIASES = {
    "standard12": STANDARD_12,
    "independent8": INDEPENDENT_8,
}

LeadSpec = Union[None, str, Sequence[str]]


def resolve_leads(spec: LeadSpec, available: Sequence[str]) -> List[str]:
    """
    Turn a lead spec into a list of lead names.

    Args:
        spec: ``None`` keeps ``available`` in its stored order; a string is an
            alias from :data:`LEAD_ALIASES`; a sequence lists lead names.
        available: Leads the dataset stores, in file order.

    Returns:
        Lead names in output channel order.
    """
    if spec is None:
        return list(available)
    if isinstance(spec, str):
        if spec not in LEAD_ALIASES:
            raise ValueError(
                f"Unknown lead alias {spec!r}; known: {sorted(LEAD_ALIASES)}"
            )
        names = list(LEAD_ALIASES[spec])
    else:
        names = list(spec)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(
            f"Leads {missing} are not available; dataset has {list(available)}"
        )
    return names


def lead_indices(names: Sequence[str], available: Sequence[str]) -> List[int]:
    """Indices into ``available`` for each name in ``names``."""
    position = {n: i for i, n in enumerate(available)}
    return [position[n] for n in names]


def maybe_lead_names(n: int, names: Optional[Sequence[str]]) -> List[str]:
    """``names`` if given, else ``["lead0", "lead1", ...]``."""
    return list(names) if names is not None else [f"lead{i}" for i in range(n)]
