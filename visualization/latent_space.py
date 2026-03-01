"""Latent space visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_latent_space_2d(
    latents: Tensor | np.ndarray,
    labels: np.ndarray = None,
    method: str = "pca",
    title: str = "Latent Space Visualization",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Visualize latent space in 2D using PCA or t-SNE.
    
    Args:
        latents: Latent vectors of shape (n_samples, latent_dim, latent_length)
                 or (n_samples, latent_dim)
        labels: Optional labels for coloring points
        method: "pca" or "tsne"
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(latents, Tensor):
        latents = latents.cpu().numpy()
    
    # Flatten if needed
    if latents.ndim == 3:
        n_samples, latent_dim, latent_length = latents.shape
        latents = latents.reshape(n_samples, -1)
    
    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        latents_2d = reducer.fit_transform(latents)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
        subtitle = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                  alpha=0.6, s=20, c='blue')
    
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_latent_distribution(
    latents: Tensor | np.ndarray,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot distribution of latent dimensions.
    
    Args:
        latents: Latent vectors
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if isinstance(latents, Tensor):
        latents = latents.cpu().numpy()
    
    if latents.ndim == 3:
        latents = latents.reshape(latents.shape[0], -1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Mean and std per dimension
    means = latents.mean(axis=0)
    stds = latents.std(axis=0)
    
    ax1.plot(means, 'b-', label='Mean')
    ax1.fill_between(range(len(means)), 
                     means - stds, means + stds, 
                     alpha=0.3, label='±1 std')
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("Value")
    ax1.set_title("Latent Dimension Statistics")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of all latent values
    ax2.hist(latents.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Latent Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Latent Value Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
