"""
Test script for VAE model
"""

import torch
from ecgen.models.vae import VAE1D, VAEConfig, VAELightning, vae_loss


def test_vae_forward():
    """Test VAE forward pass"""
    print("Testing VAE forward pass...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VAE1D(
        in_channels=12,
        base_channels=64,
        latent_channels=8,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
    ).to(device)

    x = torch.randn(4, 12, 5000).to(device)

    recon, mean, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Reconstruction shape: {recon.shape}")

    total_loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nVAE forward pass test successful!")


def test_vae_lightning():
    """Test VAE Lightning module"""
    print("\nTesting VAE Lightning module...")

    config = VAEConfig(
        in_channels=12,
        base_channels=64,
        latent_channels=8,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        lr=1e-4,
        kl_weight=0.0001,
    )

    model = VAELightning(config)

    batch = {"ecg_signals": torch.randn(4, 12, 5000)}

    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")

    val_loss = model.validation_step(batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")

    samples = model.sample(n_samples=4, seq_length=5000)
    print(f"Generated samples shape: {samples.shape}")

    print("\nVAE Lightning module test successful!")


if __name__ == "__main__":
    test_vae_forward()
    test_vae_lightning()
    print("\nAll tests passed!")
