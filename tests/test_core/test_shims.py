"""
Old import paths keep working and point at the same objects as ecgen.families.
"""

import ecgen.families as fam


def test_research_layout_shims():
    from models.gan.pulse2pulse import Pulse2PulseGAN
    from models.vae import VAELightning
    from models.vqvae import PriorLightning, VQVAELightning
    from training.losses import vae_loss
    from utils.config import resolve_config

    assert VAELightning is fam.VAELightning
    assert VQVAELightning is fam.VQVAELightning
    assert PriorLightning is fam.PriorLightning
    assert Pulse2PulseGAN is fam.Pulse2PulseGAN
    from ecgen.families.vae import vae_loss as new_vae_loss
    from ecgen.utils.config import resolve_config as new_resolve

    assert vae_loss is new_vae_loss and resolve_config is new_resolve


def test_legacy_ecgen_models_shims():
    from ecgen.models import Pulse2PulseGAN, VAELightning, WaveGANGenerator
    from ecgen.models.pulse2pulse import Pulse2PulseConfig

    assert VAELightning is fam.VAELightning
    assert Pulse2PulseGAN is fam.Pulse2PulseGAN
    assert Pulse2PulseConfig is fam.Pulse2PulseConfig
    from ecgen.families.gan import WaveGANGenerator as new_gen

    assert WaveGANGenerator is new_gen


def test_legacy_ecgen_training_callbacks():
    from ecgen.training import VAEVisualizationCallback
    from training.callbacks import VAEVisualizationCallback as root_cb

    assert VAEVisualizationCallback is root_cb
