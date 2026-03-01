from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def vqvae_loss(
    recon: Tensor,
    x: Tensor,
    vq_loss: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    VQ-VAE loss function.
    
    Combines reconstruction loss with vector quantization loss.
    Unlike standard VAE, there's no KL divergence term since we use discrete codes.
    
    Args:
        recon: Reconstructed signal [B, C, L]
        x: Original signal [B, C, L]
        vq_loss: Vector quantization loss (codebook + commitment)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        vq_loss: Vector quantization loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    
    # Total loss
    total_loss = recon_loss + vq_loss
    
    return total_loss, recon_loss, vq_loss


def prior_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Prior model loss function.
    
    Cross-entropy loss for autoregressive prediction of discrete codes.
    
    Args:
        logits: Predicted logits [B, num_embeddings, L]
        targets: Target indices [B, L]
        
    Returns:
        Cross-entropy loss
    """
    # Reshape for cross entropy: [B, num_embeddings, L] -> [B*L, num_embeddings]
    B, num_embeddings, L = logits.shape
    logits = logits.permute(0, 2, 1).contiguous()  # [B, L, num_embeddings]
    logits = logits.view(-1, num_embeddings)  # [B*L, num_embeddings]
    
    # Flatten targets: [B, L] -> [B*L]
    targets = targets.view(-1)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, targets, reduction="mean")
    
    return loss
