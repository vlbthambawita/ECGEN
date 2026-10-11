"""
Conditional VAE, conditional / paired Pulse2Pulse, and the conditional PixelCNN
prior: training through the runner, reloading, and sampling with raw conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ecgen.evaluation.measure import adherence, heart_rate
from ecgen.families import PriorLightning, Pulse2PulseGAN, VAELightning
from ecgen.families.vqvae import PixelCNNPrior
from ecgen.training.runner import train

SEQ = 512
MEAS = {
    "field": "measurements",
    "name": "numeric",
    "params": {"num_features": 2, "dim": 16},
}
OBS = {"field": "obs", "name": "masked_signal", "params": {"num_leads": 12}}
OBS8 = {"field": "obs", "name": "masked_signal", "params": {"num_leads": 8}}


def _cfg(
    tmp_path: Path, model: dict, seq=SEQ, leads=None, masks=None, **trainer
) -> dict:
    return {
        "experiment": {"name": "t", "seed": 0, "runs_root": str(tmp_path)},
        "model": model,
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 8, "seq_length": seq, "leads": leads},
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
            **trainer,
        },
    }


def _last(tmp_path: Path) -> Path:
    return tmp_path / "t" / "seed_0" / "checkpoints" / "last.ckpt"


# ------------------------------------------------------------------ VAE


def test_unconditional_vae_has_no_new_parameters():
    keys = set(VAELightning(base_channels=16, latent_channels=4).state_dict())
    assert not any("films" in k or "prior" in k or "global" in k for k in keys)


@pytest.mark.parametrize("latent_type", ["spatial", "global"])
def test_cvae_trains_and_samples_with_raw_condition(tmp_path, latent_type):
    params = {
        "base_channels": 16,
        "latent_channels": 4,
        "learned_prior": True,
        "latent_type": latent_type,
        "seq_length": SEQ,
        "conditioning": {
            "vector_dim": 16,
            "encoders": [MEAS],
            "dropout": {"measurements": 0.2},
        },
    }
    train(_cfg(tmp_path, {"name": "vae", "params": params}))
    model = VAELightning.load_from_checkpoint(_last(tmp_path)).eval()
    assert model.is_conditional and model.vae.prior is not None
    x = model.sample(3, seq_length=SEQ, cond={"measurements": [70.0, 1.0]})
    assert x.shape == (3, 12, SEQ) and torch.isfinite(x).all()


def test_cvae_lead_reconstruction_pastes_observed_leads(tmp_path):
    params = {
        "base_channels": 16,
        "latent_channels": 4,
        "learned_prior": True,
        "recon_on": "unobserved",
        "conditioning": {"encoders": [OBS]},
    }
    masks = {"name": "lead_subset", "params": {"keep": [["I"], ["I", "II"]]}}
    train(_cfg(tmp_path, {"name": "vae", "params": params}, masks=masks))
    model = VAELightning.load_from_checkpoint(_last(tmp_path)).eval()
    obs = torch.randn(2, 12, SEQ)
    mask = torch.zeros(2, 12, SEQ)
    mask[:, 0] = 1
    x = model.sample(2, seq_length=SEQ, obs=obs, obs_mask=mask)
    assert torch.allclose(x[:, 0], obs[:, 0]) and not torch.allclose(x[:, 1], obs[:, 1])


def test_learned_prior_requires_conditioning():
    with pytest.raises(ValueError, match="learned_prior needs"):
        VAELightning(learned_prior=True)


# ------------------------------------------------------------------ GAN


P2P = {"model_size": 10, "num_channels": 8, "seq_length": 5000, "n_critic": 1}


def test_pulse2pulse_critic_head_is_trained(tmp_path):
    model = Pulse2PulseGAN(**P2P)
    before = model.netD.fc1.weight.detach().clone()
    cfg = _cfg(
        tmp_path, {"name": "pulse2pulse", "params": P2P}, seq=5000, leads="independent8"
    )
    cfg["checkpoint"] = {"monitor": "val/d_wasserstein", "mode": "max"}
    cfg["trainer"]["limit_train_batches"] = 1
    trainer = train(cfg)
    trained = trainer.lightning_module
    opt_d_params = {
        id(p)
        for g in trained.optimizers()[1].optimizer.param_groups
        for p in g["params"]
    }
    assert id(trained.netD.fc1.weight) in opt_d_params
    assert before.shape == trained.netD.fc1.weight.shape


@pytest.mark.parametrize(
    "conditioning,masks,l1",
    [
        ({"vector_dim": 16, "encoders": [MEAS]}, None, 0.0),
        (
            {"encoders": [OBS8]},
            {"name": "lead_subset", "params": {"keep": [["I"]]}},
            10.0,
        ),
    ],
    ids=["vector-projection", "paired-leads"],
)
def test_conditional_pulse2pulse(tmp_path, conditioning, masks, l1):
    params = {**P2P, "conditioning": conditioning, "l1_weight": l1}
    cfg = _cfg(
        tmp_path,
        {"name": "pulse2pulse", "params": params},
        seq=5000,
        leads="independent8",
        masks=masks,
    )
    cfg["checkpoint"] = {"monitor": "val/d_wasserstein", "mode": "max"}
    cfg["trainer"]["limit_train_batches"] = 1
    train(cfg)
    model = Pulse2PulseGAN.load_from_checkpoint(_last(tmp_path)).eval()
    assert model.d_conditioner is not None and model.d_conditioner.dropout == {}
    if masks is None:
        x = model.sample(2, cond={"measurements": [70.0, 1.0]})
        assert x.shape == (2, 8, 5000)
        assert model.netD.projection is not None
    else:
        obs = torch.randn(2, 8, 5000)
        mask = torch.zeros_like(obs)
        mask[:, 0] = 1
        x = model.sample(2, obs=obs, obs_mask=mask)
        assert torch.allclose(x[:, 0], obs[:, 0])


def test_unconditional_pulse2pulse_rejects_conditions():
    with pytest.raises(ValueError, match="without conditioning"):
        Pulse2PulseGAN(model_size=10).sample(1, cond={"measurements": [70, 1]})


# ---------------------------------------------------------------- prior


def test_conditional_prior_zero_init_matches_unconditional():
    torch.manual_seed(0)
    plain = PixelCNNPrior(num_embeddings=16, hidden_dim=8, num_layers=3)
    torch.manual_seed(0)
    cond = PixelCNNPrior(
        num_embeddings=16, hidden_dim=8, num_layers=3, cond_dim=4, dense_channels=2
    )
    cond.load_state_dict(plain.state_dict(), strict=False)
    idx = torch.randint(0, 16, (2, 20))
    out = cond(idx, torch.randn(2, 4), torch.randn(2, 2, 20))
    assert torch.allclose(out, plain(idx))


def test_conditional_prior_with_cfg(tmp_path):
    stage1_cfg = _cfg(
        tmp_path / "s1",
        {
            "name": "vqvae",
            "params": {"base_channels": 16, "latent_channels": 8, "num_embeddings": 16},
        },
    )
    train(stage1_cfg)
    stage1 = _last(tmp_path / "s1")
    params = {
        "num_embeddings": 16,
        "hidden_dim": 16,
        "num_layers": 3,
        "vqvae_checkpoint": str(stage1),
        "conditioning": {
            "vector_dim": 16,
            "encoders": [MEAS],
            "dropout": {"measurements": 0.2},
        },
    }
    train(_cfg(tmp_path / "s2", {"name": "vqvae_prior", "params": params}))
    prior = PriorLightning.load_from_checkpoint(_last(tmp_path / "s2")).eval()
    x = prior.sample(
        2, seq_length=SEQ, cond={"measurements": [70.0, 1.0]}, guidance=2.0
    )
    assert x.shape == (2, 12, SEQ) and torch.isfinite(x).all()


# ----------------------------------------------------------- measurement


def test_heart_rate_on_synthetic_ground_truth():
    from ecgen.data.datasets import SyntheticECG

    ds = SyntheticECG(num_samples=20)
    x = np.stack([ds[i]["signal"].numpy() for i in range(20)])
    true = [ds[i]["meta"]["heart_rate"] for i in range(20)]
    result = adherence(true, heart_rate(x, ds.fs))
    assert (
        result["n_valid"] == 20 and result["mae"] < 1.0 and result["pearson_r"] > 0.99
    )
    assert np.isnan(heart_rate(np.zeros((12, 5000)), 500)[0])
