"""
Latent diffusion / flow on frozen stage-1 autoencoders, and the DiT1d / S4D /
Mamba backbones.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecgen.families import (
    DiffusionModule,
    LatentDiffusionModule,
    LatentFlowMatchingModule,
)
from ecgen.nn.backbones import DiT1d, S4D, SSMBackbone
from ecgen.training.runner import train
from ecgen.types import ConditionOutput

SEQ = 256


def _cfg(tmp_path: Path, model: dict, name: str = "m") -> dict:
    return {
        "experiment": {"name": name, "seed": 0, "runs_root": str(tmp_path)},
        "model": model,
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 8, "seq_length": SEQ},
            },
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
        },
    }


def _last(tmp_path: Path, name: str = "m") -> Path:
    return tmp_path / name / "seed_0" / "checkpoints" / "last.ckpt"


STAGE1 = {
    "vae": {"name": "vae", "params": {"base_channels": 16, "latent_channels": 4}},
    "vqvae": {
        "name": "vqvae",
        "params": {"base_channels": 16, "latent_channels": 8, "num_embeddings": 16},
    },
}
LATENT_UNET = {
    "name": "unet1d",
    "params": {"base_channels": 16, "channel_mults": [1, 2], "attn_levels": [1]},
}


@pytest.mark.parametrize("family", ["vae", "vqvae"])
@pytest.mark.parametrize("model_name", ["latent_diffusion", "latent_flow_matching"])
def test_latent_models_train_reload_and_sample(tmp_path, family, model_name):
    train(_cfg(tmp_path, STAGE1[family], name="stage1"))
    params = {
        "autoencoder_checkpoint": str(_last(tmp_path, "stage1")),
        "autoencoder_family": family,
        "backbone": LATENT_UNET,
        "warmup_steps": 1,
        "conditioning": {
            "vector_dim": 16,
            "encoders": [
                {
                    "field": "measurements",
                    "name": "numeric",
                    "params": {"num_features": 2, "dim": 16},
                }
            ],
        },
    }
    train(_cfg(tmp_path, {"name": model_name, "params": params}))
    _last(
        tmp_path, "stage1"
    ).unlink()  # the latent checkpoint must carry the autoencoder
    cls = (
        LatentDiffusionModule
        if model_name == "latent_diffusion"
        else LatentFlowMatchingModule
    )
    model = cls.load_from_checkpoint(_last(tmp_path))
    assert float(model.latent_scale) > 0
    assert not any(p.requires_grad for p in model.autoencoder.parameters())
    x = model.sample(2, cond={"measurements": [70.0, 1.0]}, steps=3, guidance=2.0)
    assert x.shape == (2, 12, SEQ) and torch.isfinite(x).all()
    with pytest.raises(ValueError, match="signal space"):
        model.sample(1, obs=torch.zeros(1, 12, SEQ), inpaint="replace", steps=2)


def test_latent_requires_spatial_unconditional_vae(tmp_path):
    vae = {
        "name": "vae",
        "params": {"base_channels": 16, "latent_type": "global", "seq_length": SEQ},
    }
    train(_cfg(tmp_path, vae, name="stage1"))
    with pytest.raises(ValueError, match="latent_type='spatial'"):
        LatentDiffusionModule(
            autoencoder_checkpoint=str(_last(tmp_path, "stage1")), backbone=LATENT_UNET
        )


def test_missing_stage1_checkpoint():
    with pytest.raises(FileNotFoundError):
        LatentDiffusionModule(autoencoder_checkpoint="nope.ckpt")


# -------------------------------------------------------------- backbones


def _cond(b=2, t=SEQ) -> ConditionOutput:
    return ConditionOutput(
        vector=torch.randn(b, 8),
        tokens=torch.randn(b, 3, 6),
        tokens_mask=torch.ones(b, 3, dtype=torch.bool),
        dense=torch.randn(b, 2, t),
    )


def test_dit_shapes_zero_init_and_patch_check():
    net = DiT1d(
        4,
        patch_size=16,
        dim=32,
        depth=2,
        heads=4,
        cond_dim=8,
        context_dim=6,
        dense_channels=2,
    )
    x, t = torch.randn(2, 4, SEQ), torch.rand(2) * 1000
    assert torch.equal(net(x, t, _cond()), torch.zeros(2, 4, SEQ))
    with pytest.raises(ValueError, match="patch_size"):
        net(torch.randn(1, 4, 250), t[:1])


def test_s4d_impulse_response_is_the_kernel():
    layer = S4D(3, state_size=16, bidirectional=False)
    u = torch.zeros(1, 3, 64)
    u[..., 0] = 1
    k = layer.kernel(64)[0]
    assert torch.allclose(layer(u)[0], k + layer.d[:, None] * u[0], atol=1e-5)


def test_bidirectional_s4d_sees_the_future():
    layer = S4D(2, state_size=8, bidirectional=True)
    u = torch.zeros(1, 2, 32)
    u[..., -1] = 1
    assert layer(u)[..., 0].abs().sum() > 0


def test_ssm_backbone_rejects_tokens():
    net = SSMBackbone(4, channels=16, depth=2, cond_dim=8, dense_channels=2)
    c = _cond()
    out = net(
        torch.randn(2, 4, SEQ),
        torch.rand(2),
        ConditionOutput(vector=c.vector, dense=c.dense),
    )
    assert out.shape == (2, 4, SEQ)
    with pytest.raises(ValueError, match="token"):
        SSMBackbone(4, context_dim=6)


@pytest.mark.parametrize(
    "backbone",
    [
        {
            "name": "dit1d",
            "params": {"patch_size": 16, "dim": 32, "depth": 2, "heads": 4},
        },
        {"name": "ssm", "params": {"channels": 16, "depth": 2, "state_size": 16}},
    ],
    ids=["dit1d", "ssm"],
)
def test_backbones_in_the_diffusion_family(tmp_path, backbone):
    params = {
        "backbone": backbone,
        "warmup_steps": 1,
        "conditioning": {
            "vector_dim": 16,
            "encoders": [
                {
                    "field": "rhythm",
                    "name": "categorical",
                    "params": {"num_classes": 3, "dim": 16},
                }
            ],
        },
    }
    train(_cfg(tmp_path, {"name": "diffusion", "params": params}))
    model = DiffusionModule.load_from_checkpoint(_last(tmp_path))
    assert model.sample(2, cond={"rhythm": 0}, steps=3).shape == (2, 12, SEQ)


def test_family_rejects_slots_the_backbone_cannot_take():
    tokens = {
        "field": "labels",
        "name": "multilabel",
        "params": {"num_labels": 3, "as_tokens": True},
    }
    with pytest.raises(ValueError, match="cannot consume"):
        DiffusionModule(
            backbone={"name": "ssm", "params": {"channels": 16, "depth": 1}},
            conditioning={"encoders": [tokens]},
        )


def _mamba_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _mamba_available(), reason="needs CUDA and mamba-ssm")
def test_mamba_backbone_on_gpu():
    from ecgen.nn.backbones import MambaBackbone

    net = MambaBackbone(4, channels=32, depth=2, cond_dim=8).cuda()
    out = net(
        torch.randn(2, 4, 512, device="cuda"),
        torch.rand(2, device="cuda"),
        ConditionOutput(vector=torch.randn(2, 8, device="cuda")),
    )
    assert out.shape == (2, 4, 512)
