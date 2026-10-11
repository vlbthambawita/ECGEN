"""
Text encoders: free-text condition fields (reports) → tokens for cross-attention
(or one pooled vector).

- ``text_hash``: hashed bag-of-words embeddings; no dependencies, trained with
  the model. Good for short, templated reports (machine statements).
- ``text``: a frozen Hugging Face encoder (``transformers``; ``ecgen[text]``).
  Its weights are *not* part of the generative model's checkpoint (only the
  model name is), and embeddings are cached in memory per distinct text.
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ecgen.conditioning.encoders import FieldEncoder
from ecgen.registry import CONDITION_ENCODERS

_WORD = re.compile(r"[a-z0-9äöüß]+")


def tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric words."""
    return _WORD.findall(text.lower())


def _pool(tokens: Tensor, mask: Tensor) -> Tensor:
    w = mask.to(tokens.dtype)[..., None]
    return (tokens * w).sum(1) / w.sum(1).clamp_min(1.0)


@CONDITION_ENCODERS.register("text_hash")
class HashTextEncoder(FieldEncoder):
    """
    Each word → ``crc32(word) mod vocab_size`` → learned embedding (+ learned
    position embedding). Output: up to ``max_tokens`` tokens, or their mean with
    ``as_tokens=False``.

    Args:
        dim: Embedding size.
        vocab_size: Hash buckets.
        max_tokens: Words kept per text.
        as_tokens: Tokens (cross-attention) or a pooled vector.
    """

    def __init__(
        self,
        dim: int = 128,
        vocab_size: int = 4096,
        max_tokens: int = 32,
        as_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.as_tokens = as_tokens
        self.slot = "tokens" if as_tokens else "vector"
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(max_tokens, dim) * 0.02)

    def ids(self, texts: Sequence[str]) -> Tuple[Tensor, Tensor]:
        ids = torch.zeros(len(texts), self.max_tokens, dtype=torch.long)
        mask = torch.zeros(len(texts), self.max_tokens, dtype=torch.bool)
        for i, text in enumerate(texts):
            words = tokenize(text)[: self.max_tokens]
            for j, w in enumerate(words):
                ids[i, j] = zlib.crc32(w.encode("utf-8")) % self.vocab_size
            mask[i, : len(words)] = True
        return ids, mask

    def forward(self, value: Sequence[str], missing: Optional[Tensor] = None):
        ids, mask = self.ids(list(value))
        device = self.embed.weight.device
        ids, mask = ids.to(device), mask.to(device)
        tokens = (self.embed(ids) + self.pos) * mask[..., None]
        if not self.as_tokens:
            return _pool(tokens, mask)
        return tokens, mask


class _Frozen:
    """Holds a module without registering it (keeps it out of state_dict / .to())."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module


@CONDITION_ENCODERS.register("text")
class PretrainedTextEncoder(FieldEncoder):
    """
    A frozen Hugging Face text encoder; tokens = its last hidden states.

    Args:
        model_name: Hugging Face model id or local path (e.g.
            ``"emilyalsentzer/Bio_ClinicalBERT"``, ``"bert-base-multilingual-cased"``
            for German PTB-XL reports).
        max_length: Tokenizer truncation length.
        as_tokens: Tokens (cross-attention) or the mean-pooled vector.
        cache_size: Distinct texts whose embeddings are kept in memory.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        as_tokens: bool = True,
        cache_size: int = 100_000,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'text' encoder needs transformers: pip install 'ecgen[text]'"
            ) from e
        self.model_name = model_name
        self.max_length = max_length
        self.as_tokens = as_tokens
        self.slot = "tokens" if as_tokens else "vector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = _Frozen(model)
        self.out_dim = int(model.config.hidden_size)
        self.cache_size = cache_size
        self._cache: Dict[str, Tuple[Tensor, int]] = {}
        self.register_buffer("_device_ref", torch.zeros(0), persistent=False)

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> List[Tuple[Tensor, int]]:
        model = self._model.module
        device = self._device_ref.device
        if next(model.parameters()).device != device:
            model.to(device)
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            enc = self.tokenizer(
                todo,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            hidden = model(
                **{k: v.to(device) for k, v in enc.items()}
            ).last_hidden_state
            for i, text in enumerate(todo):
                n = int(enc["attention_mask"][i].sum())
                if len(self._cache) < self.cache_size:
                    self._cache[text] = (hidden[i, :n].half().cpu(), n)
                else:
                    return [self._embed_uncached(t) for t in texts]
        return [self._cache[t] for t in texts]

    def _embed_uncached(self, text: str) -> Tuple[Tensor, int]:
        model = self._model.module
        enc = self.tokenizer(
            [text], truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        hidden = model(
            **{k: v.to(self._device_ref.device) for k, v in enc.items()}
        ).last_hidden_state[0]
        return hidden.half().cpu(), hidden.shape[0]

    def forward(self, value: Sequence[str], missing: Optional[Tensor] = None):
        texts = [t or "" for t in value]
        embedded = self._embed(texts)
        n = max(e[1] for e in embedded)
        device = self._device_ref.device
        tokens = torch.zeros(len(texts), n, self.out_dim, device=device)
        mask = torch.zeros(len(texts), n, dtype=torch.bool, device=device)
        for i, (h, k) in enumerate(embedded):
            if texts[i]:
                tokens[i, :k] = h.to(device, torch.float32)
                mask[i, :k] = True
        if not self.as_tokens:
            return _pool(tokens, mask)
        return tokens, mask
