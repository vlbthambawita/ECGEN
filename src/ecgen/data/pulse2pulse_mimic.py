from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


class ECGDatasetAdapter(Dataset):
    """
    Wrap (ecg, cond) style dataset and return only ECG signals for GAN training.
    """

    def __init__(self, base_dataset: Dataset, num_leads: int = 8) -> None:
        self.base = base_dataset
        self.num_leads = num_leads

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        ecg, _ = self.base[idx]
        if ecg.shape[0] > self.num_leads:
            ecg = ecg[: self.num_leads]
        ecg = ecg.float()
        return {"ecg_signals": ecg}


@dataclass
class Pulse2PulseMIMICConfig:
    data_dir: str
    batch_size: int = 128
    num_workers: int = 4
    max_samples: Optional[int] = None
    skip_missing_check: bool = True
    num_channels: int = 8
    seq_length: int = 5000


class Pulse2PulseMIMICDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MIMIC‑IV‑ECG ECG generation with Pulse2Pulse.

    This reuses the dataset implementation from the ecg_diffusion project.
    """

    def __init__(self, config: Pulse2PulseMIMICConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = Pulse2PulseMIMICConfig(**config)
        self.config = config

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        try:
            from ecg_diffusion.data.dataset import MIMICIVECGDataset
        except Exception as e:  # pragma: no cover - import depends on external repo
            raise ImportError(
                "ecg_diffusion.data.dataset.MIMICIVECGDataset is required for Pulse2PulseMIMICDataModule.\n"
                "Make sure the ecg_diffusion project is installed or available on PYTHONPATH."
            ) from e

        train_base = MIMICIVECGDataset(
            mimic_path=self.config.data_dir,
            split="train",
            max_samples=self.config.max_samples,
            skip_missing_check=self.config.skip_missing_check,
        )
        val_base = MIMICIVECGDataset(
            mimic_path=self.config.data_dir,
            split="val",
            max_samples=min(self.config.max_samples or 100, 100),
            skip_missing_check=self.config.skip_missing_check,
        )

        self.train_dataset = ECGDatasetAdapter(train_base, num_leads=self.config.num_channels)
        self.val_dataset = ECGDatasetAdapter(val_base, num_leads=self.config.num_channels)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set. Did you forget to call setup()? ")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not set. Did you forget to call setup()? ")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

