"""
Complete training script for VAE model with MIMIC-IV-ECG dataset
"""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ecgen.models.vae import VAEConfig, VAELightning
from ecgen.data.mimic_dataset import MIMICIVECGDataset
from ecgen.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model for ECG generation")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to MIMIC-IV-ECG data directory")
    parser.add_argument("--output_dir", type=str, default="runs/vae", 
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default="vae_mimic",
                        help="Experiment name")
    
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data workers")
    parser.add_argument("--max_epochs", type=int, default=100, 
                        help="Maximum epochs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to use (for debugging)")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.0001, 
                        help="KL divergence weight")
    
    parser.add_argument("--in_channels", type=int, default=12, 
                        help="Number of input channels (ECG leads)")
    parser.add_argument("--base_channels", type=int, default=64, 
                        help="Base number of channels")
    parser.add_argument("--latent_channels", type=int, default=8, 
                        help="Latent channels")
    parser.add_argument("--num_res_blocks", type=int, default=2, 
                        help="Number of residual blocks")
    
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test split ratio")
    
    parser.add_argument("--gpus", type=int, default=1, 
                        help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--save_top_k", type=int, default=3,
                        help="Save top k checkpoints")
    
    return parser.parse_args()


class VAEDataModule:
    """Simple data module wrapper for VAE training"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: int = None,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.max_samples = max_samples
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="train",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
            )
            
            self.val_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="val",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="test",
                val_split=self.val_split,
                test_split=self.test_split,
                max_samples=self.max_samples,
                seed=self.seed,
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
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class VAEDataWrapper(pl.LightningModule):
    """Wrapper to adapt MIMIC dataset output to VAE input format"""
    
    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model.vae
        self.config = vae_model.config
        self.save_hyperparameters(vae_model.config.__dict__)
        
        self._val_real_sample = None
        self._val_recon_sample = None
    
    def forward(self, x):
        return self.vae(x)
    
    def training_step(self, batch, batch_idx):
        ecg, features = batch
        
        from ecgen.models.vae import vae_loss
        recon, mean, logvar = self.vae(ecg)
        total_loss, recon_loss, kl_loss = vae_loss(
            recon, ecg, mean, logvar, self.config.kl_weight
        )
        
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        ecg, features = batch
        
        from ecgen.models.vae import vae_loss
        recon, mean, logvar = self.vae(ecg)
        total_loss, recon_loss, kl_loss = vae_loss(
            recon, ecg, mean, logvar, self.config.kl_weight
        )
        
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0:
            self._val_real_sample = ecg[0].detach().cpu()
            self._val_recon_sample = recon[0].detach().cpu()
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2),
        )
        return optimizer


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("VAE Training Configuration")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"KL weight: {args.kl_weight}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    config = VAEConfig(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=args.num_res_blocks,
        lr=args.learning_rate,
        kl_weight=args.kl_weight,
        b1=0.9,
        b2=0.999,
    )
    
    vae_model = VAELightning(config)
    model = VAEDataWrapper(vae_model)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    datamodule = VAEDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="vae-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=args.gradient_clip,
        deterministic=True,
    )
    
    print("\nStarting training...")
    print("=" * 80)
    
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
