"""
Rectified-flow process and the flow_matching family (same module as diffusion).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecgen.families import DiffusionModule, FlowMatchingModule
from ecgen.processes import Inpainting, RectifiedFlowProcess, ot_pairing
from ecgen.training.runner import train


def _oracle(proc: RectifiedFlowProcess, x1: torch.Tensor):
    """Exact velocity for data concentrated at x1: v = (x1 - x) / (1 - t)."""

    def fn(x, t):
        tt = (t / proc.time_scale).view(-1, 1, 1)
        return (x1 - x) / (1 - tt).clamp_min(1e-6)

    return fn


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_oracle_flow_reaches_the_data(solver):
    proc = RectifiedFlowProcess(solver=solver)
    x1 = torch.randn(1, 3, 32)
    out = proc.sample(_oracle(proc, x1), (4, 3, 32), steps=10)
    assert torch.allclose(out, x1.expand_as(out), atol=1e-4)


def test_training_pair_path_and_target():
    proc = RectifiedFlowProcess(t_sampling="uniform", sigma_min=0.0)
    x1, x0 = torch.randn(16, 2, 8), torch.randn(16, 2, 8)
    pair = proc.training_pair(x1, x0)
    t = (pair.t / proc.time_scale).view(-1, 1, 1)
    assert torch.allclose(pair.xt, t * x1 + (1 - t) * x0, atol=1e-6)
    assert torch.allclose(pair.target, x1 - x0) and torch.equal(
        pair.weight, torch.ones(16)
    )


def test_logit_normal_times_are_inside_unit_interval():
    t = RectifiedFlowProcess(t_sampling="logit_normal").sample_t(10000, "cpu")
    assert (t > 0).all() and (t < 1).all() and abs(float(t.mean()) - 0.5) < 0.02


def test_ot_pairing_lowers_transport_cost():
    torch.manual_seed(0)
    x1, x0 = torch.randn(32, 4) + 3, torch.randn(32, 4)
    perm = ot_pairing(x1, x0)
    assert sorted(perm.tolist()) == list(range(32))
    assert ((x1 - x0[perm]) ** 2).sum() < ((x1 - x0) ** 2).sum()


def test_flow_inpainting_modes():
    proc = RectifiedFlowProcess()
    obs = torch.randn(2, 3, 16)
    mask = torch.zeros_like(obs)
    mask[:, 1] = 1
    zero = lambda x, t: torch.zeros_like(x)  # noqa: E731
    for mode in ("replace", "clean"):
        out = proc.sample(zero, obs.shape, steps=4, inpaint=Inpainting(obs, mask, mode))
        assert torch.allclose(out[:, 1], obs[:, 1])
    with pytest.raises(ValueError, match="RePaint"):
        proc.sample(zero, obs.shape, steps=2, inpaint=Inpainting(obs, mask, "repaint"))


def test_bad_options():
    with pytest.raises(ValueError):
        RectifiedFlowProcess(t_sampling="beta")
    with pytest.raises(ValueError):
        RectifiedFlowProcess(solver="rk4")


TINY = {
    "name": "unet1d",
    "params": {"base_channels": 16, "channel_mults": [1, 2], "attn_levels": [1]},
}


def test_flow_matching_family_end_to_end(tmp_path: Path):
    cfg = {
        "experiment": {"name": "f", "seed": 0, "runs_root": str(tmp_path)},
        "model": {
            "name": "flow_matching",
            "params": {
                "backbone": TINY,
                "process": {"name": "rectified_flow", "params": {"ot_coupling": True}},
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
                    "dropout": {"rhythm": 0.2},
                },
            },
        },
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 8, "seq_length": 256},
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
    train(cfg)
    model = FlowMatchingModule.load_from_checkpoint(
        tmp_path / "f" / "seed_0" / "checkpoints" / "last.ckpt"
    )
    assert model.hparams["process"]["name"] == "rectified_flow"
    x = model.sample(
        2, cond={"rhythm": 2}, guidance=2.0, steps=4, solver="heun", seed=0
    )
    assert x.shape == (2, 12, 256) and torch.isfinite(x).all()


def test_default_processes_differ():
    tiny = {"backbone": TINY}
    assert type(DiffusionModule(**tiny).process).__name__ == "DDPMProcess"
    assert type(FlowMatchingModule(**tiny).process).__name__ == "RectifiedFlowProcess"
