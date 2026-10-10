"""
The prior must be strictly causal (logits at t see only codes < t), cover the
receptive field it reports, and be trained and sampled with the same alignment.
"""

import pytest
import torch

from models.vqvae import PixelCNNPrior, PriorLightning
from models.vqvae.config import PriorConfig


def _prior(**kwargs) -> PixelCNNPrior:
    torch.manual_seed(0)
    return PixelCNNPrior(num_embeddings=16, hidden_dim=16, **kwargs).eval()


@pytest.mark.parametrize("t", [0, 5, 31])
def test_logits_at_t_do_not_see_code_t_or_later(t):
    prior = _prior(num_layers=4)
    indices = torch.randint(0, 16, (2, 32))
    changed = indices.clone()
    changed[:, t:] = (changed[:, t:] + 1) % 16

    with torch.no_grad():
        a, b = prior(indices), prior(changed)
    assert torch.equal(a[:, :, : t + 1], b[:, :, : t + 1])
    if t + 1 < 32:
        assert not torch.allclose(a[:, :, t + 1], b[:, :, t + 1])


@pytest.mark.parametrize("num_layers, kernel_size, expected", [(4, 2, 16), (3, 3, 15), (10, 2, 1024)])
def test_receptive_field(num_layers, kernel_size, expected):
    prior = _prior(num_layers=num_layers, kernel_size=kernel_size)
    assert prior.receptive_field == expected


def test_receptive_field_is_exact():
    prior = _prior(num_layers=4)  # dilations 1, 2, 4, 8
    rf = prior.receptive_field
    indices = torch.randint(0, 16, (1, rf + 4))
    changed = indices.clone()
    changed[:, 0] = (changed[:, 0] + 1) % 16

    with torch.no_grad():
        a, b = prior(indices), prior(changed)
    assert not torch.allclose(a[:, :, rf], b[:, :, rf])  # code 0 is rf codes back
    assert torch.equal(a[:, :, rf + 1 :], b[:, :, rf + 1 :])  # out of reach


def test_default_prior_covers_full_latent_sequence():
    assert PixelCNNPrior().receptive_field >= 625  # 5000 samples / 8x downsampling


def test_cropped_sampling_matches_full_context():
    prior = _prior(num_layers=3)  # receptive field 8, shorter than the sequence
    indices = torch.randint(0, 16, (2, 40))
    with torch.no_grad():
        full = prior(indices)
        for i in range(40):
            start = max(0, i - prior.receptive_field)
            cropped = prior(indices[:, start : i + 1])[:, :, -1]
            assert torch.allclose(full[:, :, i], cropped, atol=1e-5), i


def test_training_and_sampling_alignment_learns_pattern():
    """Train on sequences where each code is the previous code + 1 (mod 4)."""
    torch.manual_seed(0)
    model = PriorLightning(PriorConfig(num_embeddings=4, hidden_dim=32, num_layers=3))
    opt = torch.optim.Adam(model.prior.parameters(), lr=1e-2)

    for _ in range(150):
        start = torch.randint(0, 4, (32, 1))
        indices = (start + torch.arange(24)) % 4
        loss = model.compute_loss(indices)
        opt.zero_grad()
        loss.backward()
        opt.step()

    samples = model.prior.eval().sample(batch_size=8, latent_length=24, temperature=0.1, device="cpu")
    steps = (samples[:, 1:] - samples[:, :-1]) % 4
    assert torch.all(steps == 1), samples
