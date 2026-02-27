"""
Training script for VAE model
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ecgen.models.vae import VAEConfig, VAELightning
from ecgen.data.datamodule import ECGDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model for ECG generation")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="runs/vae", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.0001, help="KL divergence weight")
    
    parser.add_argument("--in_channels", type=int, default=12, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=64, help="Base number of channels")
    parser.add_argument("--latent_channels", type=int, default=8, help="Latent channels")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks")
    
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    pl.seed_everything(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = VAEConfig(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=args.num_res_blocks,
        lr=args.learning_rate,
        kl_weight=args.kl_weight,
    )
    
    model = VAELightning(config)
    
    # Note: You'll need to implement ECGDataModule or use your own data loading
    # datamodule = ECGDataModule(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    # )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="vae-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
    )
    
    print(f"Training VAE model...")
    print(f"Config: {config}")
    print(f"Output directory: {output_dir}")
    
    # Uncomment when datamodule is ready
    # trainer.fit(model, datamodule=datamodule)
    
    print("Training script ready. Implement ECGDataModule to start training.")


if __name__ == "__main__":
    main()
