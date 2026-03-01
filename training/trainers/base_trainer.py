"""Base trainer class (future implementation)."""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base trainer class for all model types.
    
    Future implementation for unified training interface.
    """
    
    @abstractmethod
    def train(self):
        """Train the model."""
        pass
    
    @abstractmethod
    def validate(self):
        """Validate the model."""
        pass
