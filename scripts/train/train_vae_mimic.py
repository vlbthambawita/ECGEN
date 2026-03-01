#!/usr/bin/env python3
"""
Standalone script to train VAE using PyTorch Lightning.
Similar to train_pulse2pulse.py but for VAE model.
"""

import argparse
import sys
from pathlib import Path
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models.vae import VAELightning, VAEConfig
from data.datasets.mimic.dataset import MIMICIVECGDataset
from utils.seed import set_global_seed
from training.callbacks.visualization import VAEVisualizationCallback

import torch
from torch.utils.data import DataLoader


class VAEMIMICDataModule(pl.LightningDataModule):
    """Data module for VAE training with MIMIC-IV-ECG"""
    
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
        
        if stage == "test" or stage is None:
            self.test_dataset = MIMICIVECGDataset(
                mimic_path=self.data_dir,
                split="test",
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
    
    def __init__(self, vae_config: VAEConfig):
        super().__init__()
        from ecgen.models.vae import VAE1D, vae_loss
        
        self.vae = VAE1D(
            in_channels=vae_config.in_channels,
            base_channels=vae_config.base_channels,
            latent_channels=vae_config.latent_channels,
            channel_multipliers=vae_config.channel_multipliers,
            num_res_blocks=vae_config.num_res_blocks,
        )
        self.config = vae_config
        self.vae_loss = vae_loss
        self.save_hyperparameters(vae_config.__dict__)
        
        self._val_real_sample = None
        self._val_recon_sample = None
    
    def forward(self, x):
        return self.vae(x)
    
    def training_step(self, batch, batch_idx):
        ecg, features = batch
        
        recon, mean, logvar = self.vae(ecg)
        total_loss, recon_loss, kl_loss = self.vae_loss(
            recon, ecg, mean, logvar, self.config.kl_weight
        )
        
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        ecg, features = batch
        
        recon, mean, logvar = self.vae(ecg)
        total_loss, recon_loss, kl_loss = self.vae_loss(
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


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE for ECG generation")
    
    # Config file option
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # Experiment
    parser.add_argument("--exp-name", type=str, default="vae_mimic", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to MIMIC-IV-ECG dataset (required if not using --config)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (for debugging)")
    parser.add_argument("--skip-missing-check", action="store_true", help="Skip missing file check")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    
    # Model
    parser.add_argument("--in-channels", type=int, default=12, help="Number of ECG leads")
    parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--seq-length", type=int, default=5000, help="ECG sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=0.0001, help="KL divergence weight")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    
    # Training
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu/cpu)")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Device IDs")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-top-k", type=int, default=3, help="Save top k checkpoints")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ecg-vae", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=["vae", "ecg"], help="W&B tags")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Visualization
    parser.add_argument("--viz-every-n-epochs", type=int, default=5, help="Generate visualizations every N epochs")
    parser.add_argument("--viz-num-samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument("--viz-plot-all-leads", action="store_true", help="Plot all leads separately (default: overlay)")
    parser.add_argument("--viz-log-to-tensorboard", action="store_true", default=True, help="Log visualizations to TensorBoard")
    parser.add_argument("--viz-log-to-wandb", action="store_true", help="Log visualizations to W&B")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validate that either config or data-dir is provided
    if not args.config and not args.data_dir:
        print("Error: Either --config or --data-dir must be provided")
        print("Usage:")
        print("  python train_vae_mimic.py --config configs/experiments/vae_mimic.yaml")
        print("  python train_vae_mimic.py --data-dir /path/to/mimic-iv-ecg")
        sys.exit(1)
    
    # Load config from YAML if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        
        # Override args with config values
        if 'experiment' in yaml_config:
            exp_cfg = yaml_config['experiment']
            args.exp_name = exp_cfg.get('name', args.exp_name)
            args.seed = exp_cfg.get('seed', args.seed)
            args.runs_root = exp_cfg.get('runs_root', args.runs_root)
        
        if 'model' in yaml_config and 'params' in yaml_config['model']:
            model_cfg = yaml_config['model']['params'].get('config', {})
            args.in_channels = model_cfg.get('in_channels', args.in_channels)
            args.base_channels = model_cfg.get('base_channels', args.base_channels)
            args.latent_channels = model_cfg.get('latent_channels', args.latent_channels)
            args.num_res_blocks = model_cfg.get('num_res_blocks', args.num_res_blocks)
            args.lr = model_cfg.get('lr', args.lr)
            args.kl_weight = model_cfg.get('kl_weight', args.kl_weight)
            args.b1 = model_cfg.get('b1', args.b1)
            args.b2 = model_cfg.get('b2', args.b2)
        
        if 'data' in yaml_config and 'params' in yaml_config['data']:
            data_cfg = yaml_config['data']['params']
            args.data_dir = data_cfg.get('mimic_path', args.data_dir)
            args.batch_size = data_cfg.get('batch_size', args.batch_size)
            args.num_workers = data_cfg.get('num_workers', args.num_workers)
            args.max_samples = data_cfg.get('max_samples', args.max_samples)
            args.skip_missing_check = data_cfg.get('skip_missing_check', args.skip_missing_check)
            args.val_split = data_cfg.get('val_split', args.val_split)
            args.test_split = data_cfg.get('test_split', args.test_split)
            args.seq_length = data_cfg.get('seq_length', args.seq_length)
        
        if 'trainer' in yaml_config:
            trainer_cfg = yaml_config['trainer']
            args.max_epochs = trainer_cfg.get('max_epochs', args.max_epochs)
            args.accelerator = trainer_cfg.get('accelerator', args.accelerator)
            args.devices = trainer_cfg.get('devices', args.devices)
            args.log_every_n_steps = trainer_cfg.get('log_every_n_steps', args.log_every_n_steps)
            args.check_val_every_n_epoch = trainer_cfg.get('check_val_every_n_epoch', args.check_val_every_n_epoch)
            args.gradient_clip = trainer_cfg.get('gradient_clip_val', args.gradient_clip)
        
        if 'wandb' in yaml_config:
            wandb_cfg = yaml_config['wandb']
            if wandb_cfg.get('enabled', False):
                args.wandb = True
                args.wandb_project = wandb_cfg.get('project', args.wandb_project)
                args.wandb_entity = wandb_cfg.get('entity', args.wandb_entity)
                args.wandb_run_name = wandb_cfg.get('run_name', args.wandb_run_name)
                args.wandb_tags = wandb_cfg.get('tags', args.wandb_tags)
        
        if 'visualization' in yaml_config:
            viz_cfg = yaml_config['visualization']
            args.viz_every_n_epochs = viz_cfg.get('every_n_epochs', args.viz_every_n_epochs)
            args.viz_num_samples = viz_cfg.get('num_samples', args.viz_num_samples)
            args.viz_plot_all_leads = viz_cfg.get('plot_all_leads', args.viz_plot_all_leads)
            args.viz_log_to_tensorboard = viz_cfg.get('log_to_tensorboard', args.viz_log_to_tensorboard)
            args.viz_log_to_wandb = viz_cfg.get('log_to_wandb', args.viz_log_to_wandb)
        
        print(f"Configuration loaded successfully")
        print(f"Experiment: {args.exp_name}")
        print(f"Data directory: {args.data_dir}")
        print()
    
    set_global_seed(args.seed)
    
    # Setup directories
    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.exp_name / f"seed_{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    
    for d in (checkpoints_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    # Model config
    model_config = VAEConfig(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        kl_weight=args.kl_weight,
        b1=args.b1,
        b2=args.b2,
    )
    
    # Initialize model
    model = VAEDataWrapper(model_config)
    
    # Initialize data module
    datamodule = VAEMIMICDataModule(
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
    
    # Loggers
    loggers_list = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="tb",
    )
    loggers_list.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if args.wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            import wandb
            
            wandb_run_name = args.wandb_run_name or f"{args.exp_name}_seed{args.seed}"
            
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                save_dir=str(run_dir),
                tags=args.wandb_tags,
                log_model=True,
                config={
                    "in_channels": args.in_channels,
                    "base_channels": args.base_channels,
                    "latent_channels": args.latent_channels,
                    "num_res_blocks": args.num_res_blocks,
                    "seq_length": args.seq_length,
                    "lr": args.lr,
                    "kl_weight": args.kl_weight,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                },
            )
            loggers_list.append(wandb_logger)
            print(f"Weights & Biases logging enabled: {args.wandb_project}/{wandb_run_name}")
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
    
    # Callbacks
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
    
    # Visualization callback
    viz_callback = VAEVisualizationCallback(
        save_dir=samples_dir,
        log_every_n_epochs=args.viz_every_n_epochs,
        num_samples=args.viz_num_samples,
        plot_all_leads=args.viz_plot_all_leads,
        log_to_tensorboard=args.viz_log_to_tensorboard,
        log_to_wandb=args.viz_log_to_wandb,
    )
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, viz_callback]
    
    # Trainer
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
    
    # Train
    print("=" * 80)
    print(f"Starting training for experiment '{args.exp_name}'")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Model config: {model_config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)
    print()
    
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    print()
    print("=" * 80)
    print("Training finished.")
    print(f"Checkpoints saved to: {checkpoints_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
