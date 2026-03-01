"""Training progress visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    metric_name: str = "Loss",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        metric_name: Name of the metric
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, train_losses, 'b-', label=f'Train {metric_name}', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label=f'Val {metric_name}', linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Training Progress - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_multiple_metrics(
    metrics_dict: dict[str, dict[str, list[float]]],
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Plot multiple metrics in subplots.
    
    Args:
        metrics_dict: Dictionary of metrics, e.g.:
                     {"loss": {"train": [...], "val": [...]},
                      "accuracy": {"train": [...], "val": [...]}}
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    n_metrics = len(metrics_dict)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        
        for split_name, split_values in values.items():
            epochs = range(1, len(split_values) + 1)
            ax.plot(epochs, split_values, label=split_name, linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_learning_rate_schedule(
    learning_rates: list[float],
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot learning rate schedule over epochs.
    
    Args:
        learning_rates: Learning rates per epoch
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    epochs = range(1, len(learning_rates) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
