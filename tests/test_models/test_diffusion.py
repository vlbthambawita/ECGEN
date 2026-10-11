"""
DDPM process, UNet1d backbone, EMA, and the diffusion family end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecgen.families import DiffusionModule
from ecgen.nn.backbones import UNet1d
from ecgen.processes import DDPMProcess, Inpainting, cfg_model_fn
from ecgen.training.ema import EMA
from ecgen.training.runner import train
from ecgen.types import ConditionOutput

# ---------------------------------------------------------------- process


def _oracle(proc: DDPMProcess, x0: torch.Tensor):
    """Network output that is exactly right for data concentrated at x0."""

    def fn(xt, t):
        tt = t.long()
        ac = proc._ac(tt)
        eps = (xt - ac.sqrt() * x0) / (1 - ac).sqrt()
        return {
            "eps": eps,
            "x0": x0.expand_as(xt),
            "v": ac.sqrt() * eps - (1 - ac).sqrt() * x0,
        }[proc.prediction]

    return fn


@pytest.mark.parametrize("prediction", ["eps", "v", "x0"])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_ddim_with_oracle_recovers_x0(prediction, eta):
    proc = DDPMProcess(num_steps=200, prediction=prediction)
    x0 = torch.randn(1, 3, 32)
    out = proc.sample(
        _oracle(proc, x0),
        (4, 3, 32),
        steps=20,
        eta=eta,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.allclose(out, x0.expand_as(out), atol=1e-3)


@pytest.mark.parametrize("prediction", ["eps", "v", "x0"])
def test_training_target_matches_conversion(prediction):
    proc = DDPMProcess(num_steps=100, prediction=prediction, min_snr_gamma=None)
    x0, noise = torch.randn(8, 2, 16), torch.randn(8, 2, 16)
    pair = proc.training_pair(x0, noise)
    rec_x0, rec_eps = proc.to_x0_eps(pair.target, pair.xt, pair.t.long())
    assert torch.allclose(rec_x0, x0, atol=1e-3) and torch.allclose(
        rec_eps, noise, atol=1e-3
    )
    assert torch.equal(pair.weight, torch.ones(8))


def test_min_snr_weights_are_bounded():
    pair = DDPMProcess(num_steps=1000, prediction="v", min_snr_gamma=5.0).training_pair(
        torch.randn(256, 1, 4), torch.randn(256, 1, 4)
    )
    assert (pair.weight > 0).all() and (pair.weight <= 1.0 + 1e-6).all()


def test_schedules_and_timesteps():
    for schedule in ("linear", "cosine"):
        ac = DDPMProcess(num_steps=100, schedule=schedule).alphas_cumprod
        assert (ac[1:] < ac[:-1]).all() and 0 < ac[-1] < ac[0] < 1
    ts = DDPMProcess(num_steps=1000).timesteps(10)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 10
    with pytest.raises(ValueError):
        DDPMProcess(schedule="bogus")


@pytest.mark.parametrize("mode", ["replace", "repaint", "clean"])
def test_inpainting_keeps_observed_region(mode):
    proc = DDPMProcess(num_steps=100)
    obs = torch.randn(2, 3, 16)
    mask = torch.zeros_like(obs)
    mask[:, 0] = 1
    zero_fn = lambda x, t: torch.zeros_like(x)  # noqa: E731
    out = proc.sample(
        zero_fn, obs.shape, steps=5, inpaint=Inpainting(obs, mask, mode, jumps=2)
    )
    assert torch.allclose(out[:, 0], obs[:, 0])


def test_cfg_model_fn():
    c = lambda x, t: torch.ones_like(x)  # noqa: E731
    u = lambda x, t: torch.zeros_like(x)  # noqa: E731
    x = torch.zeros(1, 1, 2)
    assert torch.equal(cfg_model_fn(c, u, 3.0)(x, None), torch.full_like(x, 3.0))
    assert cfg_model_fn(c, None, 3.0) is c and cfg_model_fn(c, u, 1.0) is c


# --------------------------------------------------------------- backbone


def test_unet_shapes_conditioning_and_zero_init():
    net = UNet1d(
        4,
        base_channels=16,
        channel_mults=(1, 2),
        attn_levels=(1,),
        cond_dim=8,
        context_dim=6,
        dense_channels=2,
    )
    x, t = torch.randn(2, 4, 64), torch.tensor([3.0, 500.0])
    cond = ConditionOutput(
        vector=torch.randn(2, 8),
        tokens=torch.randn(2, 3, 6),
        tokens_mask=torch.ones(2, 3, dtype=torch.bool),
        dense=torch.randn(2, 2, 64),
    )
    assert torch.equal(
        net(x, t, cond), torch.zeros(2, 4, 64)
    )  # zero-initialised output conv
    with torch.no_grad():
        for p in net.parameters():  # un-zero the zero-initialised layers
            if p.abs().sum() == 0:
                p.normal_(std=0.1)
    out = net(x, t, cond)
    assert out.shape == (2, 4, 64)
    cond2 = ConditionOutput(cond.vector + 1, cond.tokens, cond.tokens_mask, cond.dense)
    assert not torch.allclose(net(x, t, cond2), out)
    with pytest.raises(ValueError, match="divisible"):
        net(torch.randn(1, 4, 63), t[:1], None)


# -------------------------------------------------------------------- EMA


def test_ema_update_swap_and_state():
    lin = torch.nn.Linear(2, 2)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    ema = EMA([lin], decay=0.5)
    with torch.no_grad():
        lin.weight.fill_(2.0)
    ema.update()
    with ema.swap():
        assert torch.allclose(lin.weight, torch.full((2, 2), 1.0))
    assert torch.allclose(lin.weight, torch.full((2, 2), 2.0))
    ema2 = EMA([lin], decay=0.5)
    ema2.load_state_dict(ema.state_dict())
    assert torch.equal(ema2.shadow["0.weight"], ema.shadow["0.weight"])


# ----------------------------------------------------------------- family

TINY = {
    "name": "unet1d",
    "params": {"base_channels": 16, "channel_mults": [1, 2], "attn_levels": [1]},
}
MEAS = {
    "field": "measurements",
    "name": "numeric",
    "params": {"num_features": 2, "dim": 16},
}


def _cfg(tmp_path: Path, params: dict, masks=None) -> dict:
    return {
        "experiment": {"name": "d", "seed": 0, "runs_root": str(tmp_path)},
        "model": {
            "name": "diffusion",
            "params": {"backbone": TINY, "warmup_steps": 1, **params},
        },
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 8, "seq_length": 256},
            },
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "masks": masks,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
        },
    }


def _load(tmp_path: Path) -> DiffusionModule:
    return DiffusionModule.load_from_checkpoint(
        tmp_path / "d" / "seed_0" / "checkpoints" / "last.ckpt"
    ).eval()


def test_conditional_diffusion_with_cfg_and_ema(tmp_path):
    params = {
        "conditioning": {
            "vector_dim": 16,
            "encoders": [MEAS],
            "dropout": {"measurements": 0.2},
        }
    }
    train(_cfg(tmp_path, params))
    ckpt = torch.load(
        tmp_path / "d" / "seed_0" / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert "ema" in ckpt
    model = _load(tmp_path)
    a = model.sample(
        2, cond={"measurements": [70.0, 1.0]}, guidance=2.0, steps=4, seed=1
    )
    b = model.sample(
        2, cond={"measurements": [70.0, 1.0]}, guidance=2.0, steps=4, seed=1
    )
    assert a.shape == (2, 12, 256) and torch.equal(
        a, b
    )  # seeded sampling is reproducible
    live = model.sample(
        2, cond={"measurements": [70.0, 1.0]}, steps=4, seed=1, use_ema=False
    )
    assert not torch.equal(
        live, model.sample(2, cond={"measurements": [70.0, 1.0]}, steps=4, seed=1)
    )


def test_lead_conditioned_diffusion(tmp_path):
    params = {
        "conditioning": {
            "encoders": [
                {"field": "obs", "name": "masked_signal", "params": {"num_leads": 12}}
            ]
        },
        "loss_on": "unobserved",
        "clean_observed": True,
    }
    masks = {"name": "lead_subset", "params": {"keep": [["I"], ["I", "II"]]}}
    train(_cfg(tmp_path, params, masks))
    model = _load(tmp_path)
    obs = torch.randn(2, 12, 256)
    mask = torch.zeros_like(obs)
    mask[:, 0] = 1
    out = model.sample(2, obs=obs, obs_mask=mask, steps=3)
    assert torch.allclose(out[:, 0], obs[:, 0]) and out.shape == obs.shape


def test_unconditional_diffusion_repaint(tmp_path):
    train(_cfg(tmp_path, {"ema_decay": 0.0}))
    model = _load(tmp_path)
    assert model.ema is None
    obs = torch.randn(1, 12, 256)
    mask = torch.zeros_like(obs)
    mask[:, :, :128] = 1
    out = model.sample(
        1,
        obs=obs,
        obs_mask=mask,
        inpaint="repaint",
        steps=3,
        jumps=2,
        paste_observed=False,
    )
    assert torch.allclose(out[..., :128], obs[..., :128], atol=1e-5)
    with pytest.raises(ValueError, match="trained without conditioning"):
        model.sample(1, cond={"measurements": [70, 1]}, steps=2)


def test_token_conditioning_builds_cross_attention():
    tokens = {
        "field": "labels",
        "name": "multilabel",
        "params": {"num_labels": 3, "as_tokens": True},
    }
    model = DiffusionModule(
        backbone=TINY, conditioning={"token_dim": 8, "encoders": [tokens]}
    )
    from ecgen.nn.injection import CrossAttention1d

    assert any(isinstance(m, CrossAttention1d) for m in model.backbone.modules())
