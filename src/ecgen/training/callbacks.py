"""
Training callbacks for ECG generation models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

LEAD_NAMES_8 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2"]
LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGVisualizationCallback(Callback):
    """
    Callback to visualize real vs generated ECG samples during training.
    """

    def __init__(
        self,
        save_dir: str | Path,
        every_n_epochs: int = 10,
        n_samples: int = 1,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "_val_real_sample") or not hasattr(pl_module, "_val_fake_sample"):
            return

        real = pl_module._val_real_sample
        fake = pl_module._val_fake_sample

        if real is None or fake is None:
            return

        save_path = self.save_dir / f"sample_epoch_{trainer.current_epoch:04d}.png"
        self._plot_comparison(real, fake, save_path, title=f"Epoch {trainer.current_epoch}")
        
        # Log to wandb if available
        if trainer.logger is not None:
            for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                    try:
                        import wandb
                        logger.experiment.log({
                            "samples/real_vs_fake": wandb.Image(str(save_path)),
                            "epoch": trainer.current_epoch,
                        })
                    except (ImportError, AttributeError):
                        pass

    def _plot_comparison(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        path: Path,
        title: str = "Real vs Generated",
    ) -> None:
        """Plot real vs generated ECG (first sample, all leads)."""
        real_np = real.cpu().numpy()
        fake_np = fake.cpu().numpy()
        n_leads = min(real_np.shape[0], 8)

        fig, axs = plt.subplots(n_leads, 2, figsize=(12, 1.5 * n_leads))
        if n_leads == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_leads):
            axs[i, 0].plot(real_np[i], color="C0", linewidth=0.7)
            axs[i, 0].set_title(f"Real - {LEAD_NAMES_8[i] if i < len(LEAD_NAMES_8) else f'Lead {i}'}")
            axs[i, 0].set_xlim(0, real_np.shape[-1])

            axs[i, 1].plot(fake_np[i], color="C1", linewidth=0.7)
            axs[i, 1].set_title(f"Generated - {LEAD_NAMES_8[i] if i < len(LEAD_NAMES_8) else f'Lead {i}'}")
            axs[i, 1].set_xlim(0, fake_np.shape[-1])

        axs[-1, 0].set_xlabel("Time (samples)")
        axs[-1, 1].set_xlabel("Time (samples)")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()


class GeneratedSamplesCallback(Callback):
    """
    Callback to generate and save ECG samples periodically.
    """

    def __init__(
        self,
        save_dir: str | Path,
        every_n_epochs: int = 25,
        n_samples: int = 16,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "generate_samples"):
            return

        samples = pl_module.generate_samples(self.n_samples)
        
        save_path = self.save_dir / f"generated_epoch_{trainer.current_epoch:04d}.pt"
        torch.save(samples.cpu(), save_path)

        plot_path = self.save_dir / f"generated_epoch_{trainer.current_epoch:04d}.png"
        self._plot_generated(samples[0].cpu(), plot_path, title=f"Generated ECG - Epoch {trainer.current_epoch}")
        
        # Log to wandb if available
        if trainer.logger is not None:
            for logger in trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                    try:
                        import wandb
                        logger.experiment.log({
                            "samples/generated": wandb.Image(str(plot_path)),
                            "epoch": trainer.current_epoch,
                        })
                    except (ImportError, AttributeError):
                        pass

    def _plot_generated(
        self,
        fake: torch.Tensor,
        path: Path,
        title: str = "Generated ECG",
    ) -> None:
        """Plot generated ECG sample (all leads)."""
        fake_np = fake.numpy()
        n_leads = min(fake_np.shape[0], 8)

        fig, axs = plt.subplots(n_leads, 1, figsize=(12, 1.2 * n_leads))
        if n_leads == 1:
            axs = [axs]

        for i in range(n_leads):
            axs[i].plot(fake_np[i], color="C1", linewidth=0.7)
            axs[i].set_ylabel(LEAD_NAMES_8[i] if i < len(LEAD_NAMES_8) else f"L{i}")
            axs[i].set_xlim(0, fake_np.shape[-1])

        axs[-1].set_xlabel("Time (samples)")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()


class VAEVisualizationCallback(Callback):
    """
    Callback to visualize VAE reconstructions during training.
    
    Saves comparison plots of real vs reconstructed ECG signals at specified intervals.
    """
    
    def __init__(
        self,
        save_dir: str | Path,
        log_every_n_epochs: int = 5,
        num_samples: int = 4,
        plot_all_leads: bool = True,
        log_to_tensorboard: bool = True,
        log_to_wandb: bool = False,
    ):
        """
        Args:
            save_dir: Directory to save visualization plots
            log_every_n_epochs: Generate visualizations every N epochs
            num_samples: Number of samples to visualize
            plot_all_leads: If True, plot all leads separately; if False, overlay them
            log_to_tensorboard: Log images to TensorBoard
            log_to_wandb: Log images to Weights & Biases
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.plot_all_leads = plot_all_leads
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_wandb = log_to_wandb
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called at the end of validation epoch."""
        current_epoch = trainer.current_epoch
        
        # Only log at specified intervals
        if current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return
        
        # Get a batch of validation data
        batch = next(iter(val_dataloader))
        ecg, _ = batch
        
        # Move to device
        ecg = ecg.to(pl_module.device)
        
        # Limit to num_samples
        ecg = ecg[:self.num_samples]
        
        # Generate reconstructions
        pl_module.eval()
        with torch.no_grad():
            if hasattr(pl_module, 'vae'):
                recon, _, _ = pl_module.vae(ecg)
            else:
                recon, _, _ = pl_module(ecg)
        pl_module.train()
        
        # Move to CPU for plotting
        ecg_cpu = ecg.cpu()
        recon_cpu = recon.cpu()
        
        # Create visualization
        if self.plot_all_leads:
            fig = self._plot_comparison_separate_leads(ecg_cpu, recon_cpu, current_epoch)
        else:
            fig = self._plot_comparison_overlay(ecg_cpu, recon_cpu, current_epoch)
        
        # Save to disk
        save_path = self.save_dir / f"epoch_{current_epoch:04d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[Epoch {current_epoch}] Saved VAE visualization to: {save_path}")
        
        # Log to TensorBoard
        if self.log_to_tensorboard and trainer.logger is not None:
            for logger in trainer.loggers if hasattr(trainer, 'loggers') else [trainer.logger]:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_figure'):
                    logger.experiment.add_figure(
                        "vae_reconstruction",
                        fig,
                        global_step=current_epoch
                    )
        
        # Log to Weights & Biases
        if self.log_to_wandb:
            try:
                import wandb
                for logger in trainer.loggers if hasattr(trainer, 'loggers') else [trainer.logger]:
                    if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                        logger.experiment.log({
                            "vae_reconstruction": wandb.Image(str(save_path)),
                            "epoch": current_epoch,
                        })
            except (ImportError, AttributeError):
                pass
    
    def _plot_comparison_separate_leads(
        self,
        real: torch.Tensor,
        recon: torch.Tensor,
        epoch: int
    ) -> plt.Figure:
        """
        Plot real vs reconstructed ECG with separate subplots for each lead.
        
        Args:
            real: Real ECG tensor [batch, channels, length]
            recon: Reconstructed ECG tensor [batch, channels, length]
            epoch: Current epoch number
            
        Returns:
            Matplotlib figure
        """
        n_samples = real.shape[0]
        n_leads = real.shape[1]
        
        # Create figure with subplots: rows for samples, columns for leads
        fig, axes = plt.subplots(
            n_samples * 2,  # 2 rows per sample (real + recon)
            n_leads,
            figsize=(2 * n_leads, 2 * n_samples * 2)
        )
        
        # Ensure axes is 2D
        if n_samples == 1 and n_leads == 1:
            axes = [[axes]]
        elif n_samples == 1:
            axes = [axes]
        elif n_leads == 1:
            axes = [[ax] for ax in axes]
        
        for sample_idx in range(n_samples):
            real_ecg = real[sample_idx].numpy()
            recon_ecg = recon[sample_idx].numpy()
            
            for lead_idx in range(n_leads):
                # Real ECG
                ax_real = axes[sample_idx * 2][lead_idx]
                ax_real.plot(real_ecg[lead_idx], color='C0', linewidth=0.7, label='Real')
                ax_real.set_xlim(0, real_ecg.shape[-1])
                ax_real.grid(True, alpha=0.3)
                
                # Add lead name on the left
                if lead_idx == 0:
                    ax_real.set_ylabel(f"Sample {sample_idx + 1}\nReal", fontsize=8)
                
                # Add lead name on top
                if sample_idx == 0:
                    lead_name = LEAD_NAMES_12[lead_idx] if lead_idx < len(LEAD_NAMES_12) else f"L{lead_idx}"
                    ax_real.set_title(lead_name, fontsize=9)
                
                # Remove x-axis labels for real (top row of each sample)
                ax_real.set_xticklabels([])
                
                # Reconstructed ECG
                ax_recon = axes[sample_idx * 2 + 1][lead_idx]
                ax_recon.plot(recon_ecg[lead_idx], color='C1', linewidth=0.7, label='Recon')
                ax_recon.set_xlim(0, recon_ecg.shape[-1])
                ax_recon.grid(True, alpha=0.3)
                
                if lead_idx == 0:
                    ax_recon.set_ylabel("Recon", fontsize=8)
                
                # Only show x-axis label on bottom row
                if sample_idx == n_samples - 1:
                    ax_recon.set_xlabel("Time", fontsize=8)
        
        fig.suptitle(f"VAE Reconstruction - Epoch {epoch}", fontsize=12, y=0.995)
        plt.tight_layout()
        
        return fig
    
    def _plot_comparison_overlay(
        self,
        real: torch.Tensor,
        recon: torch.Tensor,
        epoch: int
    ) -> plt.Figure:
        """
        Plot real vs reconstructed ECG with overlaid leads.
        
        Args:
            real: Real ECG tensor [batch, channels, length]
            recon: Reconstructed ECG tensor [batch, channels, length]
            epoch: Current epoch number
            
        Returns:
            Matplotlib figure
        """
        n_samples = real.shape[0]
        n_leads = real.shape[1]
        
        # Create figure with 2 columns (real, recon) and n_samples rows
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
        
        # Ensure axes is 2D
        if n_samples == 1:
            axes = [axes]
        
        for sample_idx in range(n_samples):
            real_ecg = real[sample_idx].numpy()
            recon_ecg = recon[sample_idx].numpy()
            
            # Plot real ECG
            ax_real = axes[sample_idx][0]
            for lead_idx in range(n_leads):
                lead_name = LEAD_NAMES_12[lead_idx] if lead_idx < len(LEAD_NAMES_12) else f"L{lead_idx}"
                ax_real.plot(
                    real_ecg[lead_idx],
                    linewidth=0.5,
                    alpha=0.7,
                    label=lead_name
                )
            ax_real.set_title(f"Sample {sample_idx + 1} - Real", fontsize=10)
            ax_real.set_xlim(0, real_ecg.shape[-1])
            ax_real.grid(True, alpha=0.3)
            if sample_idx == 0:
                ax_real.legend(loc="upper right", fontsize=6, ncol=2)
            
            # Plot reconstructed ECG
            ax_recon = axes[sample_idx][1]
            for lead_idx in range(n_leads):
                lead_name = LEAD_NAMES_12[lead_idx] if lead_idx < len(LEAD_NAMES_12) else f"L{lead_idx}"
                ax_recon.plot(
                    recon_ecg[lead_idx],
                    linewidth=0.5,
                    alpha=0.7,
                    label=lead_name
                )
            ax_recon.set_title(f"Sample {sample_idx + 1} - Reconstructed", fontsize=10)
            ax_recon.set_xlim(0, recon_ecg.shape[-1])
            ax_recon.grid(True, alpha=0.3)
            
            # Add x-label on bottom row
            if sample_idx == n_samples - 1:
                ax_real.set_xlabel("Time (samples)", fontsize=9)
                ax_recon.set_xlabel("Time (samples)", fontsize=9)
        
        fig.suptitle(f"VAE Reconstruction - Epoch {epoch}", fontsize=12)
        plt.tight_layout()
        
        return fig
