#!/usr/bin/env python3
"""
Standalone VQ-VAE Training Script for ECG Generation
====================================================

This script trains a VQ-VAE model in two stages:
- Stage 1: Train VQ-VAE (encoder + vector quantizer + decoder)
- Stage 2: Train PixelCNN Prior on discrete codes

All model components are included in this single file for easy deployment.

Usage:
    # Stage 1: Train VQ-VAE
    python train_vqvae_standalone.py --stage 1 --data-dir /path/to/mimic --exp-name vqvae_exp
    
    # Stage 2: Train Prior
    python train_vqvae_standalone.py --stage 2 --data-dir /path/to/mimic --exp-name prior_exp \\
        --vqvae-checkpoint runs/vqvae_exp/seed_42/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Optional wandb import
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    wandb = None


# ============================================================================
# Utility Functions
# ============================================================================

def set_global_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic is True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Loss Functions
# ============================================================================

def vqvae_loss(
    recon: Tensor,
    x: Tensor,
    vq_loss: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """VQ-VAE loss: reconstruction + vector quantization."""
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss, vq_loss


def prior_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Prior model loss: cross-entropy for autoregressive prediction."""
    B, num_embeddings, L = logits.shape
    logits = logits.permute(0, 2, 1).contiguous().view(-1, num_embeddings)
    targets = targets.view(-1)
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss


# ============================================================================
# Model Components
# ============================================================================

class ResidualBlock1D(nn.Module):
    """1D Residual block with group normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + residual


class Encoder1D(nn.Module):
    """1D Encoder for ECG signals."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder1D(nn.Module):
    """1D Decoder for ECG signals."""

    def __init__(
        self,
        out_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        channel_multipliers = tuple(reversed(channel_multipliers))
        ch = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)
        self.mid_block1 = ResidualBlock1D(ch, ch)
        self.mid_block2 = ResidualBlock1D(ch, ch)
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock1D(ch, out_ch))
                ch = out_ch
            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_block2(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQ-VAE."""

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
        """Quantize continuous latent to discrete codes."""
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=z.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)

        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        indices = encoding_indices.view(z.shape[0], -1)

        return vq_loss, quantized, indices

    @torch.no_grad()
    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """Get quantized vectors from codebook indices."""
        quantized = self.embedding(indices)
        quantized = quantized.permute(0, 2, 1).contiguous()
        return quantized


class VQVAE1D(nn.Module):
    """Vector Quantized Variational Autoencoder for ECG signals."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        latent_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_embeddings = num_embeddings

        self.encoder = Encoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            commitment_cost=commitment_cost,
        )

        self.decoder = Decoder1D(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z, _ = self.encoder(x)
        return z, z

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vq(z)

    def decode(self, z_q: Tensor) -> Tensor:
        return self.decoder(z_q)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z, _ = self.encode(x)
        vq_loss, z_q, indices = self.quantize(z)
        recon = self.decode(z_q)
        return recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor) -> Tensor:
        z, _ = self.encode(x)
        _, _, indices = self.quantize(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        z_q = self.vq.get_codebook_entry(indices)
        recon = self.decode(z_q)
        return recon


class GatedMaskedConv1d(nn.Module):
    """Gated masked 1D convolution for autoregressive modeling."""

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        assert mask_type in ["A", "B"]
        self.mask_type = mask_type
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size, padding=padding)
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self) -> Tensor:
        mask = torch.ones(1, 1, self.kernel_size)
        if self.mask_type == "A":
            mask[:, :, self.kernel_size // 2 :] = 0
        else:
            mask[:, :, self.kernel_size // 2 + 1 :] = 0
        return mask

    def forward(self, x: Tensor) -> Tensor:
        self.conv.weight.data *= self.mask
        out = self.conv(x)
        out_tanh, out_sigmoid = torch.chunk(out, 2, dim=1)
        return torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)


class PixelCNNPrior(nn.Module):
    """PixelCNN-style autoregressive prior for VQ-VAE discrete codes."""

    def __init__(
        self,
        num_embeddings: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        layers = [GatedMaskedConv1d("A", hidden_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(GatedMaskedConv1d("B", hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.logits = nn.Conv1d(hidden_dim, num_embeddings, kernel_size=1)

    def forward(self, indices: Tensor) -> Tensor:
        x = self.embedding(indices)
        x = x.permute(0, 2, 1)
        x = self.net(x)
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
        indices = torch.zeros(batch_size, latent_length, dtype=torch.long, device=device)
        for i in range(latent_length):
            logits = self.forward(indices)
            logits = logits[:, :, i] / temperature
            probs = torch.softmax(logits, dim=-1)
            indices[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return indices


# ============================================================================
# Dataset
# ============================================================================

class MIMICIVECGDataset(Dataset):
    """MIMIC-IV-ECG dataset."""

    FEATURE_NAMES = [
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ]

    def __init__(
        self,
        mimic_path: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        skip_missing_check: bool = False,
        num_leads: int = 12,
        seq_length: int = 5000,
    ) -> None:
        self.mimic_path = mimic_path
        self.split = split
        self.seed = seed
        self.skip_missing_check = skip_missing_check
        self.num_leads = num_leads
        self.seq_length = seq_length

        self.load_measurements()
        self.create_splits(val_split, test_split)
        self.filter_by_split()

        if not skip_missing_check:
            self.filter_missing_files()

        if max_samples is not None:
            self.measurements = self.measurements.head(max_samples).reset_index(drop=True)

        # Final check for empty dataset
        if len(self.measurements) == 0:
            raise ValueError(
                f"Dataset is empty after filtering! "
                f"Split: {self.split}, Data path: {self.mimic_path}. "
                "Please check:\n"
                "1. Data path is correct\n"
                "2. machine_measurements.csv exists\n"
                "3. ECG files (.hea/.dat) are present\n"
                "4. Try using --skip-missing-check flag"
            )

        self.compute_feature_stats()
        print(f"Dataset loaded: {len(self.measurements)} samples for split '{self.split}'")

    def load_measurements(self) -> None:
        path = os.path.join(self.mimic_path, "machine_measurements.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"machine_measurements.csv not found at {path}")
        self.measurements = pd.read_csv(path)

    def create_splits(self, val_split: float, test_split: float) -> None:
        subjects = self.measurements["subject_id"].unique()
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_split, random_state=self.seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_split / (1 - test_split), random_state=self.seed
        )
        self.train_subjects = set(train_subjects)
        self.val_subjects = set(val_subjects)
        self.test_subjects = set(test_subjects)

    def filter_by_split(self) -> None:
        if self.split == "train":
            subjects = self.train_subjects
        elif self.split == "val":
            subjects = self.val_subjects
        elif self.split == "test":
            subjects = self.test_subjects
        else:
            raise ValueError(f"Unknown split: {self.split}")
        self.measurements = self.measurements[
            self.measurements["subject_id"].isin(subjects)
        ].reset_index(drop=True)

    def filter_missing_files(self) -> None:
        valid_indices = []
        print(f"Checking {len(self.measurements)} files for existence...")
        for idx in range(len(self.measurements)):
            row = self.measurements.iloc[idx]
            path = self._get_wfdb_path(row)
            if os.path.exists(path + ".hea") and os.path.exists(path + ".dat"):
                valid_indices.append(idx)
        
        print(f"Found {len(valid_indices)} valid files out of {len(self.measurements)}")
        
        if len(valid_indices) == 0:
            import warnings
            warnings.warn(
                f"No valid ECG files found in {self.mimic_path}. "
                "Please check your data path or use --skip-missing-check to skip validation.",
                UserWarning
            )
        
        self.measurements = self.measurements.iloc[valid_indices].reset_index(drop=True)

    def compute_feature_stats(self) -> None:
        features = self.measurements[self.FEATURE_NAMES].values.astype(np.float32)
        self.feature_mean = np.nanmean(features, axis=0)
        self.feature_std = np.nanstd(features, axis=0) + 1e-6

    def _get_wfdb_path(self, row: pd.Series) -> str:
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        # MIMIC-IV-ECG uses first 4 digits for directory structure (e.g., p1000, p1001, etc.)
        subject_str = str(subject_id)
        if len(subject_str) >= 4:
            p_dir = f"p{subject_str[:4]}"
        else:
            p_dir = f"p{subject_str[:2]}"  # Fallback for shorter IDs
        p_subdir = f"p{subject_id}"
        s_dir = f"s{study_id}"
        return os.path.join(self.mimic_path, "files", p_dir, p_subdir, s_dir, str(study_id))

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.measurements.iloc[idx]
        path = self._get_wfdb_path(row)

        try:
            record = wfdb.rdrecord(path)
            ecg = record.p_signal.T.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            ecg = np.zeros((self.num_leads, self.seq_length), dtype=np.float32)

        if ecg.shape[0] != self.num_leads:
            ecg = ecg[: self.num_leads]
        if ecg.shape[1] < self.seq_length:
            pad_width = ((0, 0), (0, self.seq_length - ecg.shape[1]))
            ecg = np.pad(ecg, pad_width, mode="constant")
        elif ecg.shape[1] > self.seq_length:
            ecg = ecg[:, : self.seq_length]

        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        features = row[self.FEATURE_NAMES].values.astype(np.float32)
        features = (features - self.feature_mean) / self.feature_std

        return torch.from_numpy(ecg), torch.from_numpy(features)


# ============================================================================
# Lightning Modules
# ============================================================================

@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE model."""
    in_channels: int = 12
    base_channels: int = 64
    latent_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999


class VQVAELightning(pl.LightningModule):
    """PyTorch Lightning wrapper for VQ-VAE model."""

    def __init__(self, config: VQVAEConfig | dict) -> None:
        super().__init__()
        # Handle both VQVAEConfig object and dict (for checkpoint loading)
        if isinstance(config, dict):
            config = VQVAEConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.vqvae = VQVAE1D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
        )

        self._val_real_sample: Tensor | None = None
        self._val_recon_sample: Tensor | None = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vqvae(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        recon, vq_loss_val, indices = self.vqvae(ecgs)
        total_loss, recon_loss, vq_loss_component = vqvae_loss(recon, ecgs, vq_loss_val)

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/vq_loss", vq_loss_component, on_step=True, on_epoch=True, prog_bar=False)

        if batch_idx % 100 == 0:
            unique_codes = torch.unique(indices).numel()
            self.log("train/unique_codes", float(unique_codes), on_step=True, on_epoch=False)
            self.log("train/codebook_usage", float(unique_codes) / self.config.num_embeddings, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        recon, vq_loss_val, indices = self.vqvae(ecgs)
        total_loss, recon_loss, vq_loss_component = vqvae_loss(recon, ecgs, vq_loss_val)

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/vq_loss", vq_loss_component, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        unique_codes = torch.unique(indices).numel()
        self.log("val/unique_codes", float(unique_codes), on_step=False, on_epoch=True)
        self.log("val/codebook_usage", float(unique_codes) / self.config.num_embeddings, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self._val_real_sample = ecgs[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()

        return total_loss

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor) -> Tensor:
        self.vqvae.eval()
        return self.vqvae.encode_to_indices(x)

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        self.vqvae.eval()
        return self.vqvae.decode_from_indices(indices)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.vqvae.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer


@dataclass
class PriorConfig:
    """Configuration for PixelCNN prior model."""
    num_embeddings: int = 512
    hidden_dim: int = 128
    num_layers: int = 3
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    vqvae_checkpoint: str = ""


class PriorLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for PixelCNN prior model."""

    def __init__(self, config: PriorConfig | dict) -> None:
        super().__init__()
        # Handle both PriorConfig object and dict (for checkpoint loading)
        if isinstance(config, dict):
            config = PriorConfig(**config)
        self.save_hyperparameters(config.__dict__)
        self.config = config

        self.prior = PixelCNNPrior(
            num_embeddings=config.num_embeddings,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )

        self.vqvae: VQVAELightning | None = None
        if config.vqvae_checkpoint:
            self._load_vqvae(config.vqvae_checkpoint)

    def _load_vqvae(self, checkpoint_path: str) -> None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
        print(f"Loading VQ-VAE from: {checkpoint_path}")
        self.vqvae = VQVAELightning.load_from_checkpoint(checkpoint_path)
        self.vqvae.eval()
        self.vqvae.freeze()
        print("VQ-VAE loaded and frozen")

    def forward(self, indices: Tensor) -> Tensor:
        return self.prior(indices)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        with torch.no_grad():
            if self.vqvae is None:
                raise RuntimeError("VQ-VAE not loaded")
            indices = self.vqvae.encode_to_indices(ecgs)

        logits = self.prior(indices)
        loss = prior_loss(logits[:, :, :-1], indices[:, 1:])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ecgs, _ = batch
        with torch.no_grad():
            if self.vqvae is None:
                raise RuntimeError("VQ-VAE not loaded")
            indices = self.vqvae.encode_to_indices(ecgs)

        logits = self.prior(indices)
        loss = prior_loss(logits[:, :, :-1], indices[:, 1:])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 16,
        seq_length: int = 5000,
        temperature: float = 1.0,
    ) -> Tensor:
        if self.vqvae is None:
            raise RuntimeError("VQ-VAE not loaded")
        self.prior.eval()
        device = next(self.prior.parameters()).device
        latent_length = seq_length // 16
        indices = self.prior.sample(
            batch_size=n_samples,
            latent_length=latent_length,
            temperature=temperature,
            device=device,
        )
        samples = self.vqvae.decode_from_indices(indices)
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.prior.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer


# ============================================================================
# Data Module
# ============================================================================

class VQVAEMIMICDataModule(pl.LightningDataModule):
    """Data module for VQ-VAE training with MIMIC-IV-ECG."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: int = None,
        seed: int = 42,
        skip_missing_check: bool = False,
        num_leads: int = 12,
        seq_length: int = 5000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.max_samples = max_samples
        self.seed = seed
        self.skip_missing_check = skip_missing_check
        self.num_leads = num_leads
        self.seq_length = seq_length

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="train",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
                skip_missing_check=self.skip_missing_check,
                num_leads=self.num_leads,
                seq_length=self.seq_length,
            )
            self.val_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="val",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
                skip_missing_check=self.skip_missing_check,
                num_leads=self.num_leads,
                seq_length=self.seq_length,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


# ============================================================================
# Visualization Callback
# ============================================================================

class VAEVisualizationCallback(Callback):
    """Callback to visualize reconstructions during training."""

    def __init__(
        self,
        save_dir: Path,
        log_every_n_epochs: int = 5,
        num_samples: int = 4,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if not hasattr(pl_module, "_val_real_sample") or not hasattr(pl_module, "_val_recon_sample"):
            return

        real = pl_module._val_real_sample
        recon = pl_module._val_recon_sample

        if real is None or recon is None:
            return

        save_path = self.save_dir / f"epoch_{trainer.current_epoch:04d}.png"
        self._plot_comparison(real, recon, save_path, epoch=trainer.current_epoch)

    def _plot_comparison(self, real: Tensor, recon: Tensor, path: Path, epoch: int):
        real_np = real.cpu().numpy()
        recon_np = recon.cpu().numpy()
        n_leads = min(real_np.shape[0], 8)

        fig, axs = plt.subplots(n_leads, 2, figsize=(12, 1.5 * n_leads))
        if n_leads == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_leads):
            axs[i, 0].plot(real_np[i], linewidth=0.7)
            axs[i, 0].set_title(f"Real - Lead {i}")
            axs[i, 1].plot(recon_np[i], linewidth=0.7)
            axs[i, 1].set_title(f"Reconstructed - Lead {i}")

        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()


# ============================================================================
# Training Functions
# ============================================================================

def train_stage1_vqvae(args):
    """Train VQ-VAE (Stage 1)."""
    print("=" * 80)
    print("STAGE 1: Training VQ-VAE")
    print("=" * 80)
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        print("Please set the correct path using --data-dir or DATA_DIR environment variable")
        sys.exit(1)
    
    measurements_file = os.path.join(args.data_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_file):
        print(f"ERROR: machine_measurements.csv not found at: {measurements_file}")
        print("Please ensure you have the correct MIMIC-IV-ECG dataset path")
        sys.exit(1)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Measurements file: {measurements_file}")

    set_global_seed(args.seed)

    run_dir = Path(args.runs_root) / args.exp_name / f"seed_{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (checkpoints_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_config = VQVAEConfig(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=args.num_res_blocks,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
    )

    model = VQVAELightning(model_config)

    datamodule = VQVAEMIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        seed=args.seed,
        skip_missing_check=args.skip_missing_check,
        num_leads=args.in_channels,
        seq_length=args.seq_length,
    )

    # Setup loggers
    loggers_list = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
    loggers_list.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if args.wandb and WANDB_AVAILABLE:
        wandb_run_name = args.wandb_run_name or f"{args.exp_name}_seed{args.seed}"
        
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                save_dir=str(run_dir),
                tags=args.wandb_tags if args.wandb_tags else ["vqvae", "ecg", "stage1"],
                config={
                    "stage": 1,
                    "in_channels": args.in_channels,
                    "base_channels": args.base_channels,
                    "latent_channels": args.latent_channels,
                    "num_res_blocks": args.num_res_blocks,
                    "num_embeddings": args.num_embeddings,
                    "commitment_cost": args.commitment_cost,
                    "seq_length": args.seq_length,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "seed": args.seed,
                },
            )
            loggers_list.append(wandb_logger)
            print(f"✓ Weights & Biases logging enabled: {args.wandb_project}/{wandb_run_name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize wandb: {e}")
            print("  Continuing with TensorBoard only")
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠ Warning: wandb requested but not installed")
        print("  Install with: pip install wandb")
        print("  Continuing with TensorBoard only")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch{epoch:03d}-step{step:06d}",
        save_last=True,
        save_top_k=args.save_top_k,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    viz_callback = VAEVisualizationCallback(
        save_dir=samples_dir,
        log_every_n_epochs=args.viz_every_n_epochs,
        num_samples=args.viz_num_samples,
    )

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, viz_callback]

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip,
        deterministic=True,
    )

    print(f"Run directory: {run_dir}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    trainer.fit(model=model, datamodule=datamodule)

    print()
    print("=" * 80)
    print("VQ-VAE training (Stage 1) finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print()
    print("Next step: Train the prior model (Stage 2) using:")
    print(f"  python {sys.argv[0]} --stage 2 --data-dir {args.data_dir} \\")
    print(f"    --vqvae-checkpoint {checkpoint_callback.best_model_path}")
    print("=" * 80)


def train_stage2_prior(args):
    """Train PixelCNN Prior (Stage 2)."""
    print("=" * 80)
    print("STAGE 2: Training PixelCNN Prior")
    print("=" * 80)
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        print("Please set the correct path using --data-dir or DATA_DIR environment variable")
        sys.exit(1)
    
    measurements_file = os.path.join(args.data_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_file):
        print(f"ERROR: machine_measurements.csv not found at: {measurements_file}")
        print("Please ensure you have the correct MIMIC-IV-ECG dataset path")
        sys.exit(1)

    if not args.vqvae_checkpoint:
        print("Error: --vqvae-checkpoint is required for Stage 2")
        sys.exit(1)

    if not Path(args.vqvae_checkpoint).exists():
        print(f"Error: VQ-VAE checkpoint not found: {args.vqvae_checkpoint}")
        sys.exit(1)

    set_global_seed(args.seed)

    run_dir = Path(args.runs_root) / args.exp_name / f"seed_{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for d in (checkpoints_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_config = PriorConfig(
        num_embeddings=args.num_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
        vqvae_checkpoint=args.vqvae_checkpoint,
    )

    model = PriorLightning(model_config)

    datamodule = VQVAEMIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        seed=args.seed,
        skip_missing_check=args.skip_missing_check,
        num_leads=12,
        seq_length=args.seq_length,
    )

    # Setup loggers
    loggers_list = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
    loggers_list.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if args.wandb and WANDB_AVAILABLE:
        wandb_run_name = args.wandb_run_name or f"{args.exp_name}_seed{args.seed}"
        
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                save_dir=str(run_dir),
                tags=args.wandb_tags if args.wandb_tags else ["vqvae", "prior", "pixelcnn", "stage2"],
                config={
                    "stage": 2,
                    "num_embeddings": args.num_embeddings,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "seq_length": args.seq_length,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "seed": args.seed,
                    "vqvae_checkpoint": args.vqvae_checkpoint,
                },
            )
            loggers_list.append(wandb_logger)
            print(f"✓ Weights & Biases logging enabled: {args.wandb_project}/{wandb_run_name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize wandb: {e}")
            print("  Continuing with TensorBoard only")
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠ Warning: wandb requested but not installed")
        print("  Install with: pip install wandb")
        print("  Continuing with TensorBoard only")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch{epoch:03d}-step{step:06d}",
        save_last=True,
        save_top_k=args.save_top_k,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=loggers_list,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip,
        deterministic=True,
    )

    print()
    print(f"Run directory: {run_dir}")
    print(f"VQ-VAE checkpoint: {args.vqvae_checkpoint}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    trainer.fit(model=model, datamodule=datamodule)

    print()
    print("=" * 80)
    print("Prior training (Stage 2) finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print()
    print("You can now generate ECG samples using the trained prior model!")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for ECG generation")

    # Stage selection
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage: 1 (VQ-VAE) or 2 (Prior)")

    # Experiment
    parser.add_argument("--exp-name", type=str, default="vqvae_mimic", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")

    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Path to MIMIC-IV-ECG dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for debugging)")
    parser.add_argument("--skip-missing-check", action="store_true", help="Skip missing file check")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")

    # Model (Stage 1)
    parser.add_argument("--in-channels", type=int, default=12, help="Number of ECG leads")
    parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels")
    parser.add_argument("--latent-channels", type=int, default=64, help="Latent channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--num-embeddings", type=int, default=512, help="Codebook size")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--seq-length", type=int, default=5000, help="ECG sequence length")

    # Model (Stage 2)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for PixelCNN")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of gated conv layers")
    parser.add_argument("--vqvae-checkpoint", type=str, default=None, help="Path to VQ-VAE checkpoint (Stage 2)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu/cpu)")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Device IDs")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-top-k", type=int, default=3, help="Save top k checkpoints")

    # Visualization (Stage 1 only)
    parser.add_argument("--viz-every-n-epochs", type=int, default=5, help="Generate visualizations every N epochs")
    parser.add_argument("--viz-num-samples", type=int, default=4, help="Number of samples to visualize")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ecg-vqvae", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="W&B tags")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == 1:
        train_stage1_vqvae(args)
    elif args.stage == 2:
        # Adjust learning rate for Stage 2 if not explicitly set
        if args.lr == 1e-4:
            args.lr = 1e-3
        train_stage2_prior(args)


if __name__ == "__main__":
    main()
