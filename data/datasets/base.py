from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseECGDataset(Dataset, ABC):
    """
    Base class for ECG datasets.
    
    All ECG datasets should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return a sample from the dataset.
        
        Should return a dictionary with at least:
        - 'ecg_signals': torch.Tensor of shape (num_leads, seq_length)
        """
        pass
    
    @property
    @abstractmethod
    def num_leads(self) -> int:
        """Return the number of ECG leads."""
        pass
    
    @property
    @abstractmethod
    def seq_length(self) -> int:
        """Return the sequence length."""
        pass
