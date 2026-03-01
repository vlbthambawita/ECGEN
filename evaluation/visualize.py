"""Evaluation visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_evaluation_results(
    metrics: dict,
    save_path: str | Path = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot evaluation results as bar charts.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.7, edgecolor='black')
    
    # Color bars based on value (assuming lower is better for most metrics)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylabel("Metric Value")
    ax.set_title("Evaluation Metrics")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_matrix(
    models: list[str],
    metrics: dict[str, list[float]],
    save_path: str | Path = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot comparison matrix of multiple models across metrics.
    
    Args:
        models: List of model names
        metrics: Dictionary of metric names to lists of values (one per model)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    metric_names = list(metrics.keys())
    n_models = len(models)
    n_metrics = len(metric_names)
    
    # Create matrix
    matrix = np.array([metrics[m] for m in metric_names])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_metrics))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels(metric_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Metric Value", rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(n_metrics):
        for j in range(n_models):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title("Model Comparison Matrix")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
