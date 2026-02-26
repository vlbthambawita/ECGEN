#!/usr/bin/env python3
"""
Standalone script to train Pulse2Pulse WaveGAN using PyTorch Lightning.
Similar to pulse2pulse_simple.py but integrated with ECGEN framework.
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ecgen.models.pulse2pulse import Pulse2PulseGAN, Pulse2PulseConfig
from ecgen.data.pulse2pulse_mimic import Pulse2PulseMIMICDataModule, Pulse2PulseMIMICConfig
from ecgen.training.callbacks import ECGVisualizationCallback, GeneratedSamplesCallback
from ecgen.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pulse2Pulse WaveGAN for ECG generation")
    
    # Experiment
    parser.add_argument("--exp-name", type=str, default="pulse2pulse_mimic", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root directory for runs")
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to MIMIC-IV-ECG dataset",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use (for debugging)")
    parser.add_argument("--skip-missing-check", action="store_true", help="Skip missing file check")
    
    # Model
    parser.add_argument("--model-size", type=int, default=50, help="Model size parameter")
    parser.add_argument("--num-channels", type=int, default=8, help="Number of ECG leads")
    parser.add_argument("--seq-length", type=int, default=5000, help="ECG sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.9, help="Adam beta2")
    parser.add_argument("--lmbda", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument("--n-critic", type=int, default=5, help="Discriminator updates per generator update")
    
    # Training
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu/cpu)")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Device IDs")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5, help="Validate every N epochs")
    
    # Callbacks
    parser.add_argument("--viz-every-n-epochs", type=int, default=10, help="Visualize every N epochs")
    parser.add_argument("--save-samples-every-n-epochs", type=int, default=25, help="Save samples every N epochs")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ecg-pulse2pulse", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=["pulse2pulse", "wavegan"], help="W&B tags")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    set_global_seed(args.seed)
    
    # Setup directories
    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.exp_name / f"seed_{args.seed}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    
    for d in (checkpoints_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    # Model config
    model_config = Pulse2PulseConfig(
        model_size=args.model_size,
        num_channels=args.num_channels,
        seq_length=args.seq_length,
        lr=args.lr,
        b1=args.b1,
        b2=args.b2,
        lmbda=args.lmbda,
        n_critic=args.n_critic,
    )
    
    # Data config
    data_config = Pulse2PulseMIMICConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        skip_missing_check=args.skip_missing_check,
        num_channels=args.num_channels,
        seq_length=args.seq_length,
    )
    
    # Initialize model and data
    model = Pulse2PulseGAN(model_config)
    datamodule = Pulse2PulseMIMICDataModule(data_config)
    
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
                    "model_size": args.model_size,
                    "num_channels": args.num_channels,
                    "seq_length": args.seq_length,
                    "lr": args.lr,
                    "b1": args.b1,
                    "b2": args.b2,
                    "lmbda": args.lmbda,
                    "n_critic": args.n_critic,
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
        save_top_k=3,
        monitor="val/d_wasserstein",
        mode="max",
        auto_insert_metric_name=False,
    )
    
    viz_callback = ECGVisualizationCallback(
        save_dir=samples_dir,
        every_n_epochs=args.viz_every_n_epochs,
        n_samples=1,
    )
    
    samples_callback = GeneratedSamplesCallback(
        save_dir=samples_dir,
        every_n_epochs=args.save_samples_every_n_epochs,
        n_samples=16,
    )
    
    callbacks = [checkpoint_callback, viz_callback, samples_callback]
    
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
    )
    
    # Train
    print(f"Starting training for experiment '{args.exp_name}'")
    print(f"Run directory: {run_dir}")
    print(f"Model config: {model_config}")
    print(f"Data config: {data_config}")
    
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    print("Training finished.")
    print(f"Checkpoints saved to: {checkpoints_dir}")
    print(f"Samples saved to: {samples_dir}")


if __name__ == "__main__":
    main()
