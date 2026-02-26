#!/usr/bin/env python3
"""
Generate ECG samples using trained Pulse2Pulse model.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Add src to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ecgen.models.pulse2pulse import Pulse2PulseGAN

LEAD_NAMES_8 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2"]


def plot_ecg_grid(samples: torch.Tensor, save_path: Path, n_cols: int = 4) -> None:
    """Plot grid of generated ECG samples."""
    n_samples = samples.shape[0]
    n_leads = samples.shape[1]
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx in range(n_samples):
        ax = axes[idx]
        ecg = samples[idx].cpu().numpy()
        
        # Plot all leads overlaid
        for lead_idx in range(n_leads):
            ax.plot(ecg[lead_idx], linewidth=0.5, alpha=0.7, label=LEAD_NAMES_8[lead_idx] if lead_idx < len(LEAD_NAMES_8) else f"L{lead_idx}")
        
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xlim(0, ecg.shape[-1])
        if idx == 0:
            ax.legend(loc="upper right", fontsize=6)
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved grid plot to: {save_path}")


def plot_ecg_detailed(sample: torch.Tensor, save_path: Path) -> None:
    """Plot single ECG sample with separate subplots for each lead."""
    ecg = sample.cpu().numpy()
    n_leads = ecg.shape[0]
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 1.5 * n_leads))
    if n_leads == 1:
        axes = [axes]
    
    for i in range(n_leads):
        axes[i].plot(ecg[i], color="C1", linewidth=0.7)
        axes[i].set_ylabel(LEAD_NAMES_8[i] if i < len(LEAD_NAMES_8) else f"L{i}")
        axes[i].set_xlim(0, ecg.shape[-1])
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (samples)")
    plt.suptitle("Generated ECG")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved detailed plot to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECG samples with Pulse2Pulse")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_samples",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Save generated samples as .pt files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    torch.manual_seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = Pulse2PulseGAN.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully")
    print(f"Config: {model.config}")
    
    # Generate samples
    print(f"Generating {args.n_samples} samples...")
    with torch.no_grad():
        samples = model.generate_samples(args.n_samples)
    
    print(f"Generated samples shape: {samples.shape}")
    
    # Save tensors
    if args.save_tensors:
        tensor_path = output_dir / "generated_samples.pt"
        torch.save(samples.cpu(), tensor_path)
        print(f"Saved tensors to: {tensor_path}")
    
    # Plot grid
    grid_path = output_dir / "generated_grid.png"
    plot_ecg_grid(samples, grid_path, n_cols=4)
    
    # Plot first sample in detail
    detailed_path = output_dir / "generated_sample_0_detailed.png"
    plot_ecg_detailed(samples[0], detailed_path)
    
    # Save individual samples
    for idx in range(min(args.n_samples, 4)):
        sample_path = output_dir / f"generated_sample_{idx}.png"
        plot_ecg_detailed(samples[idx], sample_path)
    
    print(f"\nGeneration complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
