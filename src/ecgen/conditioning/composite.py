"""
Combining several field encoders into one :class:`~ecgen.types.ConditionOutput`,
with per-field condition dropout (for classifier-free guidance).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from ecgen.registry import CONDITION_ENCODERS
from ecgen.types import ConditionOutput

#: Field name that makes an encoder read ``obs`` / ``obs_mask`` instead of ``cond``.
OBS_FIELD = "obs"

Drop = Union[None, str, Iterable[str], Mapping[str, Tensor]]


class CompositeConditioner(nn.Module):
    """
    Encodes condition fields into the ``vector`` / ``tokens`` / ``dense`` slots.

    - vector encoders are projected to ``vector_dim`` and summed;
    - token encoders are projected to ``token_dim`` and concatenated along N;
    - dense encoders are concatenated along channels.

    Config (also stored in model hyperparameters)::

        conditioning:
          vector_dim: 256
          token_dim: 256
          encoders:
            - {field: measurements, name: numeric, params: {num_features: 9}}
            - {field: obs, name: masked_signal, params: {num_leads: 12}}
          dropout: {measurements: 0.1}      # per-field drop probability in training

    A dropped or absent field is replaced by a learned *null* embedding
    (vector: one ``[vector_dim]`` row per field; tokens: one null token;
    dense: everything unobserved). The same null is used for the
    unconditional pass of classifier-free guidance (``drop="all"``).

    Args:
        encoders: Specs ``{field, name | target, params}``.
        vector_dim: Size of the combined global vector.
        token_dim: Size of each token.
        dropout: Per-field drop probability during training.
    """

    def __init__(
        self,
        encoders: Sequence[Mapping[str, Any]],
        vector_dim: int = 256,
        token_dim: int = 256,
        dropout: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("conditioning.encoders is empty")
        self.vector_dim = vector_dim
        self.token_dim = token_dim
        self.dropout = dict(dropout or {})

        self.fields: List[str] = []
        self.encoders = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        self.null_vector = nn.ParameterDict()
        self.null_token = nn.ParameterDict()
        dense_channels = 0
        for spec in encoders:
            field = spec["field"]
            if field in self.encoders:
                raise ValueError(f"Field {field!r} has two encoders")
            enc = CONDITION_ENCODERS.build(
                {k: v for k, v in spec.items() if k != "field"}
            )
            self.fields.append(field)
            self.encoders[field] = enc
            if enc.slot == "vector":
                self.proj[field] = nn.Linear(enc.out_dim, vector_dim)
                self.null_vector[field] = nn.Parameter(torch.zeros(vector_dim))
            elif enc.slot == "tokens":
                self.proj[field] = nn.Linear(enc.out_dim, token_dim)
                self.null_token[field] = nn.Parameter(torch.randn(token_dim) * 0.02)
            elif enc.slot == "dense":
                dense_channels += enc.out_dim
            else:
                raise ValueError(f"Encoder for {field!r} has unknown slot {enc.slot!r}")
        unknown = set(self.dropout) - set(self.fields)
        if unknown:
            raise ValueError(
                f"dropout names fields without encoders: {sorted(unknown)}"
            )
        self.dense_channels = dense_channels

    # ------------------------------------------------------------ properties

    @property
    def slots(self) -> set[str]:
        return {self.encoders[f].slot for f in self.fields}

    @property
    def uses_obs(self) -> bool:
        return OBS_FIELD in self.fields

    @property
    def cond_fields(self) -> List[str]:
        """Fields read from ``cond`` (everything except ``obs``)."""
        return [f for f in self.fields if f != OBS_FIELD]

    # --------------------------------------------------------------- forward

    def _drop_masks(
        self, batch_size: int, device: torch.device, drop: Drop
    ) -> Dict[str, Tensor]:
        """Per-field ``[B]`` bool masks, True = replace with null."""
        masks = {
            f: torch.zeros(batch_size, dtype=torch.bool, device=device)
            for f in self.fields
        }
        if self.training:
            for f, p in self.dropout.items():
                if p > 0:
                    masks[f] = torch.rand(batch_size, device=device) < p
        if drop is None:
            return masks
        if isinstance(drop, Mapping):
            for f, m in drop.items():
                masks[f] = masks[f] | m.to(device).bool()
            return masks
        names = (
            self.fields
            if drop == "all"
            else ([drop] if isinstance(drop, str) else list(drop))
        )
        for f in names:
            masks[f] = torch.ones(batch_size, dtype=torch.bool, device=device)
        return masks

    def forward(
        self,
        cond: Optional[Mapping[str, Tensor]] = None,
        cond_missing: Optional[Mapping[str, Tensor]] = None,
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        drop: Drop = None,
    ) -> ConditionOutput:
        """
        Args:
            cond: Normalized condition fields; a missing key means "not given".
            cond_missing: Per-field missing masks.
            obs: Partially observed ECG ``[B, C, T]``.
            obs_mask: Mask of ``obs`` (1 = observed).
            batch_size: Needed when no field is given at all.
            drop: ``"all"``, field name(s), or ``{field: [B] bool}`` to force nulls
                (e.g. the unconditional half of classifier-free guidance).
        """
        cond = dict(cond or {})
        cond_missing = dict(cond_missing or {})
        b = batch_size or _infer_batch(cond, obs)
        device = _infer_device(self, cond, obs)
        drops = self._drop_masks(b, device, drop)

        vector: Optional[Tensor] = None
        tokens: List[Tensor] = []
        token_masks: List[Tensor] = []
        dense: List[Tensor] = []
        for field in self.fields:
            enc = self.encoders[field]
            dropped = drops[field]
            if field == OBS_FIELD:
                given = obs is not None and obs_mask is not None
            else:
                given = field in cond
            if not given:
                dropped = torch.ones_like(dropped)

            if enc.slot == "dense":
                dense.append(self._dense(field, enc, obs, obs_mask, dropped, b, device))
                continue

            null_rows = dropped[:, None]
            # Given fields are always encoded, so a fully dropped batch keeps the
            # same token count as a conditional one (CFG can batch both passes).
            out = enc(cond[field], cond_missing.get(field)) if given else None

            if enc.slot == "vector":
                null = self.null_vector[field].expand(b, -1)
                v = (
                    null
                    if out is None
                    else torch.where(null_rows, null, self.proj[field](out))
                )
                vector = v if vector is None else vector + v
            else:
                null_tok = self.null_token[field].view(1, 1, -1).expand(b, 1, -1)
                if out is None:
                    tok, tmask = (
                        null_tok,
                        torch.ones(b, 1, dtype=torch.bool, device=device),
                    )
                else:
                    raw_tok, raw_mask = out
                    tok = self.proj[field](raw_tok)
                    # dropped rows: only the null token (in slot 0) is valid
                    tok = torch.where(
                        null_rows[..., None], null_tok.expand_as(tok), tok
                    )
                    tmask = raw_mask.bool() & ~null_rows
                    tmask[:, 0] |= dropped
                tokens.append(tok)
                token_masks.append(tmask)

        return ConditionOutput(
            vector=vector,
            tokens=torch.cat(tokens, dim=1) if tokens else None,
            tokens_mask=torch.cat(token_masks, dim=1) if token_masks else None,
            dense=torch.cat(dense, dim=1) if dense else None,
        )

    def _dense(self, field, enc, obs, obs_mask, dropped, b, device) -> Tensor:
        if obs is None or obs_mask is None:
            raise ValueError(
                f"Dense encoder for {field!r} needs obs/obs_mask; configure data.masks "
                "for training or pass obs= and obs_mask= when sampling"
            )
        keep = (~dropped).to(obs.dtype).view(-1, *([1] * (obs.dim() - 1)))
        return enc((obs * keep, obs_mask.to(obs.dtype) * keep))


def _infer_device(
    module: nn.Module, cond: Mapping[str, Tensor], obs: Optional[Tensor]
) -> torch.device:
    # Not next(module.parameters()): an encoder set may have no parameters, and a
    # StopIteration inside a Lightning step silently ends the epoch.
    for v in cond.values():
        if isinstance(v, Tensor):
            return v.device
    if obs is not None:
        return obs.device
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def _infer_batch(cond: Mapping[str, Tensor], obs: Optional[Tensor]) -> int:
    for v in cond.values():
        return v.shape[0] if isinstance(v, Tensor) else len(v)
    if obs is not None:
        return obs.shape[0]
    raise ValueError("Cannot infer batch size: no condition given; pass batch_size=")


def build_conditioner(
    spec: Optional[Mapping[str, Any]],
) -> Optional[CompositeConditioner]:
    """``None`` / empty spec → ``None`` (unconditional)."""
    if not spec:
        return None
    return CompositeConditioner(**dict(spec))
