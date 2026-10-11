"""
Generic ``LightningDataModule`` over any registered dataset.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ecgen.data.collate import ECGCollate
from ecgen.data.datasets.base import BaseECGDataset
from ecgen.registry import DATASETS


class ECGDataModule(pl.LightningDataModule):
    """
    Builds train/val/test splits of one dataset and yields :class:`~ecgen.types.ECGBatch`.

    Args:
        dataset: Dataset spec, ``{"name" | "target": ..., "params": {...}}``; the
            ``split`` argument is filled in per split.
        batch_size: Batch size for all splits.
        num_workers: DataLoader workers.
        pin_memory: Defaults to ``torch.cuda.is_available()``.
        split_params: Per-split overrides of ``dataset.params``, e.g.
            ``{"val": {"max_samples": 1000}}``.

    The train split's ``feature_normalizer`` (if any) is passed to the val/test
    datasets that accept one, and saved in checkpoints via :meth:`state_dict`.
    """

    def __init__(
        self,
        dataset: Mapping[str, Any],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: Optional[bool] = None,
        split_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.dataset_spec = dict(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = (
            torch.cuda.is_available() if pin_memory is None else pin_memory
        )
        self.split_params = {k: dict(v) for k, v in (split_params or {}).items()}

        self.datasets: Dict[str, BaseECGDataset] = {}

    def build_dataset(self, split: str) -> BaseECGDataset:
        cls = DATASETS.resolve(self.dataset_spec)
        params = {
            **dict(self.dataset_spec.get("params") or {}),
            **self.split_params.get(split, {}),
            "split": split,
        }
        train = self.datasets.get("train")
        if (
            split != "train"
            and "feature_normalizer" not in params
            and train is not None
            and getattr(train, "feature_normalizer", None) is not None
            and "feature_normalizer" in inspect.signature(cls).parameters
        ):
            params["feature_normalizer"] = train.feature_normalizer
        return cls(**params)

    def setup(self, stage: Optional[str] = None) -> None:
        splits = {
            "fit": ("train", "val"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("test",),
        }
        wanted = (
            splits.get(stage, ("train", "val", "test"))
            if stage
            else ("train", "val", "test")
        )
        for split in wanted:
            if split not in self.datasets:
                self.datasets[split] = self.build_dataset(split)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            self.datasets[split] = self.build_dataset(split)
        ds = self.datasets[split]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=ECGCollate(ds.lead_names, ds.fs),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    @property
    def reference_dataset(self) -> BaseECGDataset:
        """Any built split (train preferred), for lead names, fs, etc."""
        for split in ("train", "val", "test"):
            if split in self.datasets:
                return self.datasets[split]
        raise RuntimeError("No dataset built yet; call setup() first.")

    def state_dict(self) -> Dict[str, Any]:
        if not self.datasets:
            return {}
        ds = self.reference_dataset
        norm = getattr(self.datasets.get("train", ds), "feature_normalizer", None)
        return {
            "lead_names": list(ds.lead_names),
            "fs": ds.fs,
            "seq_length": ds.seq_length,
            "feature_normalizer": norm.to_dict() if norm is not None else None,
        }
