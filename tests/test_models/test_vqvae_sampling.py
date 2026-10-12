"""
Samples decoded from the VQ-VAE and the PixelCNN prior must have the requested length.
"""

import pytest
import torch

from ecgen.families.vqvae import PriorLightning, VQVAEConfig, VQVAELightning
from ecgen.families.vqvae.config import PriorConfig

SMALL = dict(base_channels=16, latent_channels=8, num_embeddings=32)


def _prior(vqvae: VQVAELightning) -> PriorLightning:
    prior = PriorLightning(
        PriorConfig(num_embeddings=32, hidden_dim=16, vqvae_config=dict(vqvae.config.__dict__))
    )
    prior.vqvae.load_state_dict(vqvae.state_dict())
    return prior


@pytest.mark.parametrize("multipliers, factor", [((1, 2, 4, 4), 8), ((1, 2, 4), 4), ((1, 2, 2, 4, 4), 16)])
def test_downsample_factor_matches_encoder(multipliers, factor):
    vqvae = VQVAELightning(VQVAEConfig(channel_multipliers=multipliers, **SMALL))
    assert vqvae.downsample_factor == factor
    codes = vqvae.encode_to_indices(torch.randn(1, 12, 400))
    assert codes.shape[-1] == vqvae.latent_length(400)


@pytest.mark.parametrize("multipliers", [(1, 2, 4, 4), (1, 2, 4)])
def test_prior_sample_has_requested_length(multipliers):
    vqvae = VQVAELightning(VQVAEConfig(channel_multipliers=multipliers, **SMALL))
    samples = _prior(vqvae).sample(n_samples=2, seq_length=400)
    assert samples.shape == (2, 12, 400)


def test_vqvae_sample_has_requested_length():
    samples = VQVAELightning(VQVAEConfig(**SMALL)).sample(n_samples=2, seq_length=400)
    assert samples.shape == (2, 12, 400)


def test_seq_length_not_divisible_by_factor_rejected():
    prior = _prior(VQVAELightning(VQVAEConfig(**SMALL)))
    with pytest.raises(ValueError, match="multiple of the downsampling factor 8"):
        prior.sample(n_samples=1, seq_length=4999)
