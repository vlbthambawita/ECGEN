#!/usr/bin/env python3
"""
Generate ECG samples using trained VAE model.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# from models.vae import VAELightning


def main():
    parser = argparse.ArgumentParser(description="Generate VAE samples")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Output directory")
    parser.add_argument("--seq_length", type=int, default=5000, help="Sequence length")
    args = parser.parse_args()

    # TODO: Implement VAE generation
    print(f"Generating {args.num_samples} samples from {args.checkpoint}")
    print("VAE generation - to be implemented")


if __name__ == "__main__":
    main()
