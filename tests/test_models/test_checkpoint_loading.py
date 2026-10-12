"""
Checkpoints saved by the Lightning wrappers must reload with a plain
``Wrapper.load_from_checkpoint(path)``, without passing ``config=``.
"""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from ecgen.families.gan.pulse2pulse import Pulse2PulseConfig, Pulse2PulseGAN
from ecgen.families.vae import VAEConfig, VAELightning
from ecgen.families.vqvae import PriorLightning, VQVAEConfig, VQVAELightning
from ecgen.families.vqvae.config import PriorConfig

SMALL_VQVAE = VQVAEConfig(base_channels=16, latent_channels=8, num_embeddings=32)


def _save(model: pl.LightningModule, path: Path) -> Path:
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)
    return path


def _assert_same_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_vae_reload(tmp_path):
    model = VAELightning(VAEConfig(base_channels=16, latent_channels=4))
    loaded = VAELightning.load_from_checkpoint(_save(model, tmp_path / "vae.ckpt"))
    assert loaded.config.base_channels == 16
    _assert_same_weights(model, loaded)


def test_vqvae_reload(tmp_path):
    model = VQVAELightning(SMALL_VQVAE)
    loaded = VQVAELightning.load_from_checkpoint(_save(model, tmp_path / "vqvae.ckpt"))
    assert loaded.config.num_embeddings == 32
    _assert_same_weights(model, loaded)


def test_explicit_config_still_accepted(tmp_path):
    ckpt = _save(VQVAELightning(SMALL_VQVAE), tmp_path / "vqvae.ckpt")
    loaded = VQVAELightning.load_from_checkpoint(ckpt, config=SMALL_VQVAE)
    assert loaded.config == SMALL_VQVAE


def test_unknown_field_rejected():
    with pytest.raises(TypeError, match="Unexpected VQVAEConfig fields"):
        VQVAELightning(num_embedings=32)


def test_prior_loads_stage1_and_reloads_without_it(tmp_path):
    stage1 = _save(VQVAELightning(SMALL_VQVAE), tmp_path / "vqvae.ckpt")
    prior = PriorLightning(
        PriorConfig(num_embeddings=32, hidden_dim=16, vqvae_checkpoint=str(stage1))
    )
    assert prior.vqvae is not None
    prior_ckpt = _save(prior, tmp_path / "prior.ckpt")

    # The prior checkpoint carries the VQ-VAE, so stage 1 is no longer needed
    stage1.unlink()
    loaded = PriorLightning.load_from_checkpoint(prior_ckpt)
    assert loaded.vqvae is not None
    assert not any(p.requires_grad for p in loaded.vqvae.parameters())
    _assert_same_weights(prior, loaded)


def test_pulse2pulse_reload_with_lazy_discriminator_layer(tmp_path):
    model = Pulse2PulseGAN(Pulse2PulseConfig(model_size=4))
    with torch.no_grad():
        model.netD(torch.randn(1, 8, 5000))  # creates netD.fc1
    loaded = Pulse2PulseGAN.load_from_checkpoint(_save(model, tmp_path / "p2p.ckpt"))
    _assert_same_weights(model, loaded)
