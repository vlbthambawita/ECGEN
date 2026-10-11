from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalGatedBlock(nn.Module):
    """
    Residual gated causal 1D convolution (WaveNet-style).

    Output at position t depends only on inputs at positions <= t: the input is
    left-padded by (kernel_size - 1) * dilation, so no future positions are seen.
    Uses gated activation: tanh(W_f * x) * sigmoid(W_g * x)

    Args:
        channels: Number of input/output channels
        kernel_size: Size of convolutional kernel
        dilation: Dilation of the convolution
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        cond_dim: int = 0,
        dense_channels: int = 0,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        # Conditional PixelCNN (van den Oord et al., 2016): the condition adds
        # V_f h / V_g h to both gate pre-activations; zero-initialised
        self.cond_proj = nn.Linear(cond_dim, 2 * channels) if cond_dim else None
        self.dense_proj = (
            nn.Conv1d(dense_channels, 2 * channels, 1) if dense_channels else None
        )
        for m in (self.cond_proj, self.dense_proj):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # Gated convolution: outputs 2x channels for tanh and sigmoid gates
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(
        self, x: Tensor, c: Optional[Tensor] = None, dense: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [B, C, L]

        Returns:
            (residual output [B, C, L], skip output [B, C, L])
        """
        out = self.conv(F.pad(x, (self.left_pad, 0)))
        if self.cond_proj is not None and c is not None:
            out = out + self.cond_proj(c)[..., None]
        if self.dense_proj is not None and dense is not None:
            out = out + self.dense_proj(dense)

        # Split into two halves for gated activation
        out_tanh, out_sigmoid = torch.chunk(out, 2, dim=1)
        h = torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)

        return x + self.residual(h), self.skip(h)


class PixelCNNPrior(nn.Module):
    """
    Autoregressive prior for VQ-VAE discrete codes.

    Learns the distribution p(z) over discrete latent codes. The embedded codes
    are shifted right by one position, then passed through causal dilated
    convolutions, so the logits at position t depend only on codes < t and
    predict code t. Dilations double every layer (1, 2, 4, ...) and restart
    every ``dilation_cycle`` layers, so the receptive field grows exponentially
    with depth (see ``receptive_field``).

    Args:
        num_embeddings: Size of the codebook (vocabulary size)
        hidden_dim: Hidden dimension for the network
        num_layers: Number of gated causal conv layers
        kernel_size: Kernel size of the causal convolutions
        dilation_cycle: Number of layers after which the dilation resets to 1
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 10,
        kernel_size: int = 2,
        dilation_cycle: int = 10,
        cond_dim: int = 0,
        dense_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        # Embedding layer: convert discrete indices to continuous vectors
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)

        self.dilations = [2 ** (i % dilation_cycle) for i in range(num_layers)]
        self.kernel_size = kernel_size
        self.blocks = nn.ModuleList(
            CausalGatedBlock(hidden_dim, kernel_size, d, cond_dim, dense_channels)
            for d in self.dilations
        )

        # Output head: project summed skip connections to logits over codebook
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, num_embeddings, kernel_size=1),
        )

    @property
    def receptive_field(self) -> int:
        """Number of preceding codes that can influence the prediction of a code."""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(
        self,
        indices: Tensor,
        c: Optional[Tensor] = None,
        dense: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute logits for each discrete code given the codes before it.

        Args:
            indices: Discrete code indices [B, L]
            c: Optional global condition [B, cond_dim]
            dense: Optional condition at code resolution [B, dense_channels, L]

        Returns:
            Logits over codebook [B, num_embeddings, L]; position t predicts
            indices[:, t] from indices[:, :t]
        """
        # Embed indices: [B, L] -> [B, hidden_dim, L]
        x = self.embedding(indices).permute(0, 2, 1)

        # Shift right by one so position t only sees codes < t (zeros at t = 0)
        x = F.pad(x, (1, 0))[:, :, :-1]

        skip = torch.zeros_like(x)
        for block in self.blocks:
            x, s = block(x, c, dense)
            skip = skip + s

        return self.head(skip)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        latent_length: int = 625,  # 5000 samples / 8x VQ-VAE downsampling
        temperature: float = 1.0,
        device: str = "cuda",
        c: Optional[Tensor] = None,
        dense: Optional[Tensor] = None,
        c_uncond: Optional[Tensor] = None,
        dense_uncond: Optional[Tensor] = None,
        guidance: float = 1.0,
    ) -> Tensor:
        """
        Sample discrete codes autoregressively from the prior.

        Args:
            batch_size: Number of samples to generate
            latent_length: Length of latent sequence
            temperature: Sampling temperature (higher = more random)
            device: Device to generate samples on
            c: Global condition (see :meth:`forward`).
            dense: Condition at code resolution, covering all ``latent_length`` positions.
            c_uncond: Null global condition, for classifier-free guidance.
            dense_uncond: Null dense condition, for classifier-free guidance.
            guidance: CFG scale ``w``; logits = ``l_u + w (l_c - l_u)``. 1 = no guidance.

        Returns:
            Sampled indices [B, L]
        """
        indices = torch.zeros(
            batch_size, latent_length, dtype=torch.long, device=device
        )
        use_cfg = guidance != 1.0 and (c_uncond is not None or dense_uncond is not None)

        def window(d: Optional[Tensor], start: int, stop: int) -> Optional[Tensor]:
            return None if d is None else d[..., start:stop]

        for i in range(latent_length):
            # Logits at position i depend only on codes < i, so only the last
            # receptive_field codes before i are needed
            start = max(0, i - self.receptive_field)
            ctx = indices[:, start : i + 1]
            logits = self.forward(ctx, c, window(dense, start, i + 1))[:, :, -1]
            if use_cfg:
                uncond = self.forward(
                    ctx, c_uncond, window(dense_uncond, start, i + 1)
                )[:, :, -1]
                logits = uncond + guidance * (logits - uncond)
            logits = logits / temperature

            # Sample from categorical distribution
            probs = torch.softmax(logits, dim=-1)
            indices[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return indices
