#!/usr/bin/env python3
"""
Simple test script to verify Pulse2Pulse model components without Lightning.
"""

import sys
from pathlib import Path

# Add src to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch


def test_basic_imports():
    """Test basic model imports."""
    print("=" * 60)
    print("Testing Basic Imports")
    print("=" * 60)
    
    print("\n[1] Importing model components...")
    from ecgen.models.pulse2pulse import (
        WaveGANGenerator,
        WaveGANDiscriminator,
        Transpose1dLayer,
        Transpose1dLayerMultiInput,
        PhaseShuffle,
        calc_gradient_penalty,
        Pulse2PulseConfig,
    )
    print("  ✓ Model components imported successfully")
    
    return {
        'WaveGANGenerator': WaveGANGenerator,
        'WaveGANDiscriminator': WaveGANDiscriminator,
        'Transpose1dLayer': Transpose1dLayer,
        'Transpose1dLayerMultiInput': Transpose1dLayerMultiInput,
        'PhaseShuffle': PhaseShuffle,
        'calc_gradient_penalty': calc_gradient_penalty,
        'Pulse2PulseConfig': Pulse2PulseConfig,
    }


def test_generator(components):
    """Test generator architecture."""
    print("\n" + "=" * 60)
    print("Testing Generator")
    print("=" * 60)
    
    WaveGANGenerator = components['WaveGANGenerator']
    
    model_size = 50
    num_channels = 8
    seq_length = 5000
    batch_size = 4
    
    print(f"\n[1] Creating generator (model_size={model_size}, num_channels={num_channels})...")
    netG = WaveGANGenerator(
        model_size=model_size,
        num_channels=num_channels,
    )
    print("  ✓ Generator created")
    
    print(f"\n[2] Testing forward pass...")
    noise = torch.randn(batch_size, num_channels, seq_length)
    print(f"  Input shape: {noise.shape}")
    
    with torch.no_grad():
        output = netG(noise)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    assert output.shape == (batch_size, num_channels, seq_length), f"Expected shape {(batch_size, num_channels, seq_length)}, got {output.shape}"
    print("  ✓ Forward pass successful")
    
    # Count parameters
    total_params = sum(p.numel() for p in netG.parameters())
    print(f"\n[3] Generator parameters: {total_params:,}")
    
    return netG


def test_discriminator(components):
    """Test discriminator architecture."""
    print("\n" + "=" * 60)
    print("Testing Discriminator")
    print("=" * 60)
    
    WaveGANDiscriminator = components['WaveGANDiscriminator']
    
    model_size = 64
    num_channels = 8
    seq_length = 5000
    batch_size = 4
    
    print(f"\n[1] Creating discriminator (model_size={model_size}, num_channels={num_channels})...")
    netD = WaveGANDiscriminator(
        model_size=model_size,
        num_channels=num_channels,
    )
    print("  ✓ Discriminator created")
    
    print(f"\n[2] Testing forward pass...")
    ecg = torch.randn(batch_size, num_channels, seq_length)
    print(f"  Input shape: {ecg.shape}")
    
    with torch.no_grad():
        output = netD(ecg)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    print("  ✓ Forward pass successful")
    
    # Count parameters
    total_params = sum(p.numel() for p in netD.parameters())
    print(f"\n[3] Discriminator parameters: {total_params:,}")
    
    return netD


def test_gradient_penalty(components, netD):
    """Test gradient penalty calculation."""
    print("\n" + "=" * 60)
    print("Testing Gradient Penalty")
    print("=" * 60)
    
    calc_gradient_penalty = components['calc_gradient_penalty']
    
    batch_size = 4
    num_channels = 8
    seq_length = 5000
    
    print(f"\n[1] Creating real and fake data...")
    real_data = torch.randn(batch_size, num_channels, seq_length, requires_grad=False)
    fake_data = torch.randn(batch_size, num_channels, seq_length, requires_grad=False)
    
    print(f"[2] Computing gradient penalty...")
    gp = calc_gradient_penalty(netD, real_data, fake_data, lmbda=10.0)
    
    print(f"  Gradient penalty: {gp.item():.4f}")
    assert gp.item() >= 0, "Gradient penalty should be non-negative"
    print("  ✓ Gradient penalty computation successful")


def test_phase_shuffle(components):
    """Test phase shuffle layer."""
    print("\n" + "=" * 60)
    print("Testing Phase Shuffle")
    print("=" * 60)
    
    PhaseShuffle = components['PhaseShuffle']
    
    batch_size = 4
    channels = 8
    length = 1000
    
    print(f"\n[1] Creating phase shuffle layer (shift_factor=2)...")
    ps = PhaseShuffle(shift_factor=2)
    
    print(f"[2] Testing forward pass...")
    x = torch.randn(batch_size, channels, length)
    y = ps(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    assert x.shape == y.shape, "Phase shuffle should preserve shape"
    print("  ✓ Phase shuffle successful")


def test_config(components):
    """Test configuration dataclass."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    Pulse2PulseConfig = components['Pulse2PulseConfig']
    
    print(f"\n[1] Creating default config...")
    config = Pulse2PulseConfig()
    print(f"  Config: {config}")
    
    print(f"\n[2] Creating custom config...")
    custom_config = Pulse2PulseConfig(
        model_size=100,
        num_channels=12,
        seq_length=10000,
        lr=2e-4,
    )
    print(f"  Custom config: {custom_config}")
    print("  ✓ Configuration tests successful")


def main():
    print("\n" + "=" * 60)
    print("Pulse2Pulse Model Test (Without Lightning)")
    print("=" * 60)
    
    try:
        # Test imports
        components = test_basic_imports()
        
        # Test config
        test_config(components)
        
        # Test generator
        netG = test_generator(components)
        
        # Test discriminator
        netD = test_discriminator(components)
        
        # Test gradient penalty
        test_gradient_penalty(components, netD)
        
        # Test phase shuffle
        test_phase_shuffle(components)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Pulse2Pulse model components are working correctly.")
        print("Model architecture is ready for training.")
        
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
