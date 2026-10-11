"""A tiny conditional family used to test the conditioning plumbing end to end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.utils.config import resolve_config


@dataclass
class ToyConfig:
    num_leads: int = 12
    seq_length: int = 256
    lr: float = 1e-2
    conditioning: Optional[dict] = None


class ToyConditional(GenerativeModule):
    """Predicts the ECG directly from the condition vector (no noise)."""

    family = "toy"
    supports = Capabilities(conditional=True, cfg=True)
    accepts = frozenset({"vector", "dense"})

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__()
        config = resolve_config(ToyConfig, config, kwargs)
        self.save_hyperparameters(config.__dict__)
        self.config = config
        self.init_conditioning(config.conditioning)
        dim = self.conditioner.vector_dim if self.conditioner else 1
        self.head = nn.Linear(dim, config.num_leads * config.seq_length)

    def _predict(self, cond) -> torch.Tensor:
        b = cond.vector.shape[0]
        return self.head(cond.vector).view(
            b, self.config.num_leads, self.config.seq_length
        )

    def training_step(self, batch, batch_idx):
        loss = F.mse_loss(self._predict(self.condition(batch)), batch.signal)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = F.mse_loss(self._predict(self.condition(batch)), batch.signal)
        self.log("val_loss", loss)
        return loss

    @torch.no_grad()
    def sample(self, n_samples: int = 4, cond=None, **kwargs):
        return self._predict(self.encode_condition(n_samples, cond))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
