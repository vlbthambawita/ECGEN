#!/usr/bin/env python3
"""
Template training script for new models.

Copy this file and modify for your model.
"""

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Import your model, dataset, and utilities
# from models.your_model import YourModel
# from data.datasets.your_dataset import YourDataset
# from utils.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train your model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/your_model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    # set_global_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement your training logic here
    print("Training template - implement your model training logic")


if __name__ == "__main__":
    main()
