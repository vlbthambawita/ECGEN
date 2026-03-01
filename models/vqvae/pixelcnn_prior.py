from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GatedMaskedConv1d(nn.Module):
    """
    Gated masked 1D convolution for autoregressive modeling.
    
    Ensures that prediction at position t only depends on positions < t.
    Uses gated activation: tanh(W_f * x) * sigmoid(W_g * x)
    
    Args:
        mask_type: 'A' for first layer (strict causality), 'B' for subsequent layers
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        assert mask_type in ["A", "B"], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        self.kernel_size = kernel_size

        # Gated convolution: outputs 2x channels for tanh and sigmoid gates
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size, padding=padding
        )

        # Create causal mask
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self) -> Tensor:
        """Create causal mask for the convolution kernel."""
        mask = torch.ones(1, 1, self.kernel_size)
        
        # For mask type A: zero out current and future positions
        # For mask type B: zero out only future positions
        if self.mask_type == "A":
            # Strict causality: can't see current position
            mask[:, :, self.kernel_size // 2 :] = 0
        else:
            # Can see current position but not future
            mask[:, :, self.kernel_size // 2 + 1 :] = 0
            
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            Output tensor [B, out_channels, L]
        """
        # Apply masked convolution
        self.conv.weight.data *= self.mask
        out = self.conv(x)

        # Split into two halves for gated activation
        out_tanh, out_sigmoid = torch.chunk(out, 2, dim=1)

        # Gated activation
        return torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)


class PixelCNNPrior(nn.Module):
    """
    PixelCNN-style autoregressive prior for VQ-VAE discrete codes.
    
    Learns the distribution p(z) over discrete latent codes.
    Uses causal masked convolutions to ensure autoregressive property.
    
    Args:
        num_embeddings: Size of the codebook (vocabulary size)
        hidden_dim: Hidden dimension for the network
        num_layers: Number of gated masked conv layers
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        # Embedding layer: convert discrete indices to continuous vectors
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)

        # First layer with mask type A (strict causality)
        layers = [GatedMaskedConv1d("A", hidden_dim, hidden_dim)]

        # Subsequent layers with mask type B
        for _ in range(num_layers - 1):
            layers.append(GatedMaskedConv1d("B", hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

        # Output layer: project to logits over codebook
        self.logits = nn.Conv1d(hidden_dim, num_embeddings, kernel_size=1)

    def forward(self, indices: Tensor) -> Tensor:
        """
        Compute logits for next discrete code prediction.
        
        Args:
            indices: Discrete code indices [B, L]
            
        Returns:
            Logits over codebook [B, num_embeddings, L]
        """
        # Embed indices: [B, L] -> [B, L, hidden_dim]
        x = self.embedding(indices)

        # Reshape for convolution: [B, L, hidden_dim] -> [B, hidden_dim, L]
        x = x.permute(0, 2, 1)

        # Apply masked convolutions
        x = self.net(x)

        # Project to logits
        logits = self.logits(x)

        return logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        latent_length: int = 312,
        temperature: float = 1.0,
        device: str = "cuda",
    ) -> Tensor:
        """
        Sample discrete codes autoregressively from the prior.
        
        Args:
            batch_size: Number of samples to generate
            latent_length: Length of latent sequence
            temperature: Sampling temperature (higher = more random)
            device: Device to generate samples on
            
        Returns:
            Sampled indices [B, L]
        """
        # Start with zeros (or could use a special start token)
        indices = torch.zeros(batch_size, latent_length, dtype=torch.long, device=device)

        # Sample autoregressively
        for i in range(latent_length):
            # Get logits for current position
            logits = self.forward(indices)  # [B, num_embeddings, L]
            logits = logits[:, :, i] / temperature  # [B, num_embeddings]

            # Sample from categorical distribution
            probs = torch.softmax(logits, dim=-1)
            indices[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return indices
