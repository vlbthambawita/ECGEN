from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader, Dataset


@dataclass
class ECGDataModule:
    """
    Minimal placeholder datamodule describing train/val/test datasets.
    """

    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    batch_size: int = 32
    num_workers: int = 4

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not set.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not set.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
