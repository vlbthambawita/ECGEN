"""Fidelity metrics for generated ECG samples."""

import torch
import numpy as np
from torch import Tensor
from scipy import linalg


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                    mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculate Fréchet Distance between two Gaussian distributions.
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability
        
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(real_features: Tensor | np.ndarray, 
                 generated_features: Tensor | np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and generated features.
    
    Args:
        real_features: Features from real samples
        generated_features: Features from generated samples
        
    Returns:
        FID score
    """
    if isinstance(real_features, Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(generated_features, Tensor):
        generated_features = generated_features.cpu().numpy()
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid


def reconstruction_error(real: Tensor | np.ndarray, 
                        reconstructed: Tensor | np.ndarray,
                        metric: str = "mse") -> float:
    """
    Calculate reconstruction error.
    
    Args:
        real: Real samples
        reconstructed: Reconstructed samples
        metric: Error metric ("mse", "mae", "rmse")
        
    Returns:
        Reconstruction error
    """
    if isinstance(real, Tensor):
        real = real.cpu().numpy()
    if isinstance(reconstructed, Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    if metric == "mse":
        error = np.mean((real - reconstructed) ** 2)
    elif metric == "mae":
        error = np.mean(np.abs(real - reconstructed))
    elif metric == "rmse":
        error = np.sqrt(np.mean((real - reconstructed) ** 2))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(error)


def structural_similarity(real: Tensor | np.ndarray, 
                         generated: Tensor | np.ndarray) -> float:
    """
    Calculate structural similarity between real and generated samples.
    
    Args:
        real: Real samples
        generated: Generated samples
        
    Returns:
        Structural similarity score
    """
    if isinstance(real, Tensor):
        real = real.cpu().numpy()
    if isinstance(generated, Tensor):
        generated = generated.cpu().numpy()
    
    # Normalize
    real_norm = (real - real.mean()) / (real.std() + 1e-8)
    gen_norm = (generated - generated.mean()) / (generated.std() + 1e-8)
    
    # Cross-correlation
    correlation = np.mean(real_norm * gen_norm)
    
    return float(correlation)
