#!/usr/bin/env python3
"""
Test script to verify Pulse2Pulse setup and model architecture.
"""

import sys
from pathlib import Path

# Add src to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from ecgen.models.pulse2pulse import (
    Pulse2PulseGAN,
    Pulse2PulseConfig,
    WaveGANGenerator,
    WaveGANDiscriminator,
)
from ecgen.data.pulse2pulse_mimic import Pulse2PulseMIMICConfig
from ecgen.training.callbacks import ECGVisualizationCallback, GeneratedSamplesCallback
from ecgen.training.losses import wgan_discriminator_loss, wgan_generator_loss
from ecgen.training.metrics import wasserstein_distance, gradient_penalty_loss


def test_model_architecture():
    """Test model instantiation and forward pass."""
    print("=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    config = Pulse2PulseConfig(
        model_size=50,
        num_channels=8,
        seq_length=5000,
        lr=1e-4,
        b1=0.5,
        b2=0.9,
        lmbda=10.0,
        n_critic=5,
    )
    
    print(f"\nConfig: {config}")
    
    # Test generator
    print("\n[1] Testing Generator...")
    netG = WaveGANGenerator(
        model_size=config.model_size,
        num_channels=config.num_channels,
    )
    
    batch_size = 4
    noise = torch.randn(batch_size, config.num_channels, config.seq_length)
    
    with torch.no_grad():
        fake = netG(noise)
    
    print(f"  Input shape: {noise.shape}")
    print(f"  Output shape: {fake.shape}")
    print(f"  Output range: [{fake.min().item():.3f}, {fake.max().item():.3f}]")
    assert fake.shape == (batch_size, config.num_channels, config.seq_length), "Generator output shape mismatch"
    print("  ✓ Generator test passed")
    
    # Test discriminator
    print("\n[2] Testing Discriminator...")
    netD = WaveGANDiscriminator(
        model_size=config.model_size,
        num_channels=config.num_channels,
    )
    
    with torch.no_grad():
        d_out = netD(fake)
    
    print(f"  Input shape: {fake.shape}")
    print(f"  Output shape: {d_out.shape}")
    print(f"  Output range: [{d_out.min().item():.3f}, {d_out.max().item():.3f}]")
    assert d_out.shape == (batch_size, 1), "Discriminator output shape mismatch"
    print("  ✓ Discriminator test passed")
    
    # Test Lightning module
    print("\n[3] Testing Lightning Module...")
    model = Pulse2PulseGAN(config)
    
    batch = {"ecg_signals": torch.randn(batch_size, config.num_channels, config.seq_length)}
    
    # Test forward
    with torch.no_grad():
        generated = model(noise)
    
    print(f"  Generated shape: {generated.shape}")
    assert generated.shape == (batch_size, config.num_channels, config.seq_length), "Model forward output shape mismatch"
    
    # Test generate_samples
    with torch.no_grad():
        samples = model.generate_samples(n_samples=8)
    
    print(f"  Generated samples shape: {samples.shape}")
    assert samples.shape == (8, config.num_channels, config.seq_length), "generate_samples output shape mismatch"
    print("  ✓ Lightning module test passed")
    
    print("\n✓ All model architecture tests passed!")


def test_losses_and_metrics():
    """Test loss functions and metrics."""
    print("\n" + "=" * 60)
    print("Testing Losses and Metrics")
    print("=" * 60)
    
    batch_size = 4
    d_real = torch.randn(batch_size, 1)
    d_fake = torch.randn(batch_size, 1)
    
    print("\n[1] Testing Wasserstein distance...")
    wd = wasserstein_distance(d_real, d_fake)
    print(f"  Wasserstein distance: {wd.item():.4f}")
    assert wd.shape == (), "Wasserstein distance should be scalar"
    print("  ✓ Wasserstein distance test passed")
    
    print("\n[2] Testing WGAN losses...")
    gp = torch.tensor(5.0)
    d_loss = wgan_discriminator_loss(d_real.mean(), d_fake.mean(), gp)
    g_loss = wgan_generator_loss(d_fake.mean())
    print(f"  Discriminator loss: {d_loss.item():.4f}")
    print(f"  Generator loss: {g_loss.item():.4f}")
    print("  ✓ WGAN losses test passed")
    
    print("\n✓ All loss and metric tests passed!")


def test_callbacks():
    """Test callback instantiation."""
    print("\n" + "=" * 60)
    print("Testing Callbacks")
    print("=" * 60)
    
    print("\n[1] Testing ECGVisualizationCallback...")
    viz_callback = ECGVisualizationCallback(
        save_dir="test_output",
        every_n_epochs=10,
        n_samples=1,
    )
    print(f"  Save dir: {viz_callback.save_dir}")
    print(f"  Every N epochs: {viz_callback.every_n_epochs}")
    print("  ✓ ECGVisualizationCallback test passed")
    
    print("\n[2] Testing GeneratedSamplesCallback...")
    samples_callback = GeneratedSamplesCallback(
        save_dir="test_output",
        every_n_epochs=25,
        n_samples=16,
    )
    print(f"  Save dir: {samples_callback.save_dir}")
    print(f"  Every N epochs: {samples_callback.every_n_epochs}")
    print("  ✓ GeneratedSamplesCallback test passed")
    
    print("\n✓ All callback tests passed!")


def test_configs():
    """Test configuration classes."""
    print("\n" + "=" * 60)
    print("Testing Configuration Classes")
    print("=" * 60)
    
    print("\n[1] Testing Pulse2PulseConfig...")
    model_config = Pulse2PulseConfig(
        model_size=50,
        num_channels=8,
        seq_length=5000,
    )
    print(f"  Config: {model_config}")
    print("  ✓ Pulse2PulseConfig test passed")
    
    print("\n[2] Testing Pulse2PulseMIMICConfig...")
    data_config = Pulse2PulseMIMICConfig(
        data_dir="/path/to/data",
        batch_size=128,
        num_workers=4,
    )
    print(f"  Config: {data_config}")
    print("  ✓ Pulse2PulseMIMICConfig test passed")
    
    print("\n✓ All configuration tests passed!")


def print_model_summary():
    """Print model parameter summary."""
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    config = Pulse2PulseConfig()
    model = Pulse2PulseGAN(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    gen_params = sum(p.numel() for p in model.netG.parameters())
    disc_params = sum(p.numel() for p in model.netD.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    print(f"\nGenerator:")
    print(f"  Model size: {config.model_size}")
    print(f"  Num channels: {config.num_channels}")
    print(f"  Sequence length: {config.seq_length}")
    
    print(f"\nTraining:")
    print(f"  Learning rate: {config.lr}")
    print(f"  Adam betas: ({config.b1}, {config.b2})")
    print(f"  Gradient penalty λ: {config.lmbda}")
    print(f"  Critic iterations: {config.n_critic}")


def main():
    print("\n" + "=" * 60)
    print("Pulse2Pulse Setup Test")
    print("=" * 60)
    
    try:
        test_configs()
        test_model_architecture()
        test_losses_and_metrics()
        test_callbacks()
        print_model_summary()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Pulse2Pulse setup is ready for training.")
        print("You can now run:")
        print("  python scripts/train_pulse2pulse.py --data-dir /path/to/MIMIC-IV-ECG")
        print("or:")
        print("  python -m ecgen.training.train --config configs/experiments/pulse2pulse_mimic.yaml")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
