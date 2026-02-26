"""
Training callbacks for ECG generation models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

LEAD_NAMES_8 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2"]


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
