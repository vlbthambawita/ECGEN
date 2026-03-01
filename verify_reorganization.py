#!/usr/bin/env python3
"""
Verification script for the reorganized ECGEN repository.
Tests that all imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

def test_imports():
    """Test all critical imports."""
    
    print("Testing imports...")
    print("=" * 60)
    
    tests = []
    
    # Test models
    try:
        from models.vae import VAE1D, VAELightning, VAEConfig
        print("✓ Models - VAE imports successful")
        tests.append(("VAE models", True))
    except Exception as e:
        print(f"✗ Models - VAE imports failed: {e}")
        tests.append(("VAE models", False))
    
    try:
        from models.components import ResidualBlock1D, Encoder1D, Decoder1D
        print("✓ Models - Components imports successful")
        tests.append(("Model components", True))
    except Exception as e:
        print(f"✗ Models - Components imports failed: {e}")
        tests.append(("Model components", False))
    
    # Test data
    try:
        from data.datasets.mimic.dataset import MIMICIVECGDataset
        print("✓ Data - MIMIC dataset import successful")
        tests.append(("MIMIC dataset", True))
    except Exception as e:
        print(f"✗ Data - MIMIC dataset import failed: {e}")
        tests.append(("MIMIC dataset", False))
    
    # Test training
    try:
        from training.callbacks.visualization import (
            VAEVisualizationCallback,
            ECGVisualizationCallback,
            GeneratedSamplesCallback,
        )
        print("✓ Training - Callbacks imports successful")
        tests.append(("Training callbacks", True))
    except Exception as e:
        print(f"✗ Training - Callbacks imports failed: {e}")
        tests.append(("Training callbacks", False))
    
    try:
        from training.losses.vae_losses import vae_loss
        print("✓ Training - VAE losses import successful")
        tests.append(("VAE losses", True))
    except Exception as e:
        print(f"✗ Training - VAE losses import failed: {e}")
        tests.append(("VAE losses", False))
    
    # Test evaluation
    try:
        from evaluation.metrics import (
            signal_to_noise_ratio,
            calculate_quality_metrics,
            pairwise_distance_diversity,
            calculate_fid,
        )
        print("✓ Evaluation - Metrics imports successful")
        tests.append(("Evaluation metrics", True))
    except Exception as e:
        print(f"✗ Evaluation - Metrics imports failed: {e}")
        tests.append(("Evaluation metrics", False))
    
    # Test visualization
    try:
        from visualization import (
            plot_ecg_leads,
            plot_latent_space_2d,
            plot_training_curves,
        )
        print("✓ Visualization - Imports successful")
        tests.append(("Visualization", True))
    except Exception as e:
        print(f"✗ Visualization - Imports failed: {e}")
        tests.append(("Visualization", False))
    
    # Test utils
    try:
        from utils.seed import set_global_seed
        from utils.io import save_checkpoint
        print("✓ Utils - Imports successful")
        tests.append(("Utils", True))
    except Exception as e:
        print(f"✗ Utils - Imports failed: {e}")
        tests.append(("Utils", False))
    
    print("=" * 60)
    
    # Summary
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All imports successful! Reorganization verified.")
        return 0
    else:
        print("❌ Some imports failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
