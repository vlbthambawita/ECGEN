"""Diversity metrics for generated ECG samples."""

import torch
import numpy as np
from torch import Tensor
from scipy.spatial.distance import pdist


def pairwise_distance_diversity(samples: Tensor | np.ndarray, metric: str = "euclidean") -> float:
    """
    Calculate diversity using pairwise distances.
    
    Args:
        samples: Generated samples of shape (n_samples, ...)
        metric: Distance metric ("euclidean", "cosine", etc.)
        
    Returns:
        Mean pairwise distance
    """
    if isinstance(samples, Tensor):
        samples = samples.cpu().numpy()
    
    # Flatten samples
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)
    
    # Calculate pairwise distances
    distances = pdist(samples_flat, metric=metric)
    
    return float(np.mean(distances))


def intra_class_diversity(samples: Tensor | np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate diversity within each class.
    
    Args:
        samples: Generated samples
        labels: Class labels for each sample
        
    Returns:
        Dictionary of diversity per class
    """
    if isinstance(samples, Tensor):
        samples = samples.cpu().numpy()
    
    unique_labels = np.unique(labels)
    diversity_per_class = {}
    
    for label in unique_labels:
        class_samples = samples[labels == label]
        if len(class_samples) > 1:
            diversity = pairwise_distance_diversity(class_samples)
            diversity_per_class[str(label)] = diversity
    
    return diversity_per_class


def mode_coverage(samples: Tensor | np.ndarray, real_samples: Tensor | np.ndarray, 
                  n_clusters: int = 10) -> float:
    """
    Estimate mode coverage using clustering.
    
    Args:
        samples: Generated samples
        real_samples: Real samples
        n_clusters: Number of clusters
        
    Returns:
        Mode coverage score (0-1)
    """
    if isinstance(samples, Tensor):
        samples = samples.cpu().numpy()
    if isinstance(real_samples, Tensor):
        real_samples = real_samples.cpu().numpy()
    
    from sklearn.cluster import KMeans
    
    # Flatten
    samples_flat = samples.reshape(len(samples), -1)
    real_flat = real_samples.reshape(len(real_samples), -1)
    
    # Cluster real samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(real_flat)
    
    # Check which clusters are covered by generated samples
    generated_labels = kmeans.predict(samples_flat)
    covered_clusters = len(np.unique(generated_labels))
    
    coverage = covered_clusters / n_clusters
    
    return float(coverage)
