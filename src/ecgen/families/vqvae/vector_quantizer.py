from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.
    
    Implements discrete latent space quantization using a learned codebook.
    Uses straight-through estimator for gradient flow during backpropagation.
    
    Args:
        num_embeddings: Size of the codebook (number of discrete codes)
        embedding_dim: Dimensionality of each code vector
        commitment_cost: Weight for commitment loss (encourages encoder outputs to commit to codes)
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Quantize continuous latent vectors to discrete codes.
        
        Args:
            z: Encoder output [B, C, L] where C = embedding_dim
            
        Returns:
            vq_loss: Vector quantization loss (codebook + commitment)
            quantized: Quantized latent codes [B, C, L]
            indices: Codebook indices [B, L]
        """
        # Reshape: [B, C, L] -> [B, L, C]
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate L2 distances to all codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T * e
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find nearest codebook entry for each latent vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=z.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize by looking up codebook entries
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)

        # Loss: Codebook loss + Commitment loss
        # Codebook loss: move codebook vectors towards encoder outputs
        # Commitment loss: encourage encoder outputs to commit to chosen codes
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()

        # Reshape back: [B, L, C] -> [B, C, L]
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        # Return indices as [B, L]
        indices = encoding_indices.view(z.shape[0], -1)

        return vq_loss, quantized, indices

    @torch.no_grad()
    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """
        Get quantized vectors from codebook indices.
        
        Args:
            indices: Codebook indices [B, L]
            
        Returns:
            Quantized vectors [B, C, L]
        """
        # Get embeddings
        quantized = self.embedding(indices)  # [B, L, C]
        
        # Reshape to [B, C, L]
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized
