"""
``ecgen.models`` is a deprecated alias of ``ecgen.families`` (removed in 2.0).
"""

import importlib
import sys
from pathlib import Path

import pytest


def test_ecgen_models_warns_and_aliases_families():
    for name in [m for m in sys.modules if m.startswith("ecgen.models")]:
        del sys.modules[name]
    with pytest.warns(DeprecationWarning, match="ecgen.families"):
        legacy = importlib.import_module("ecgen.models")
    import ecgen.families.gan as gan
    import ecgen.families.vae as vae

    assert legacy.Pulse2PulseGAN is gan.Pulse2PulseGAN
    assert legacy.VAELightning is vae.VAELightning
    assert legacy.WaveGANGenerator is gan.WaveGANGenerator


def test_root_shim_packages_are_gone():
    # Other installed projects may provide top-level ``models``/``utils``/``data``,
    # so check the repo tree rather than importability.
    root = Path(__file__).resolve().parents[2]
    for path in ("models", "utils", "data", "training/callbacks", "training/trainers"):
        assert not (root / path).exists(), path
