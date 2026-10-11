"""
The YAML runner trains every built-in family end to end on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ecgen.cli import main as cli_main
from ecgen.families import PriorLightning, Pulse2PulseGAN, VAELightning, VQVAELightning
from ecgen.families.vqvae import VQVAEConfig
from ecgen.registry import DATASETS
from ecgen.training.runner import apply_overrides, build_model, resolve_config, train

SEQ = 512


def _cfg(
    tmp_path: Path, model: dict, leads=None, seq_length: int = SEQ, **trainer
) -> dict:
    return {
        "experiment": {"name": "test", "seed": 0, "runs_root": str(tmp_path / "runs")},
        "model": model,
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 8, "seq_length": seq_length, "leads": leads},
            },
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
            **trainer,
        },
    }


SMALL_VAE = {"name": "vae", "params": {"base_channels": 16, "latent_channels": 4}}
SMALL_VQVAE = {
    "name": "vqvae",
    "params": {"base_channels": 16, "latent_channels": 8, "num_embeddings": 32},
}


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "test" / "seed_0"


def test_vae_trains_and_writes_run_dir(tmp_path):
    trainer = train(_cfg(tmp_path, SMALL_VAE))
    run_dir = _run_dir(tmp_path)
    assert trainer.state.finished
    assert (run_dir / "config_resolved.yaml").is_file() and (
        run_dir / "metadata.json"
    ).is_file()
    assert (run_dir / "metrics" / "train_val_metrics.json").is_file()
    ckpt = run_dir / "checkpoints" / "last.ckpt"
    assert ckpt.is_file()

    # The checkpoint reloads with the family class and records the data contract
    model = VAELightning.load_from_checkpoint(ckpt)
    assert model.config.base_channels == 16
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["ECGDataModule"]
    assert (
        state["lead_names"][:3] == ["I", "II", "III"]
        and state["fs"] == 500
        and state["seq_length"] == SEQ
    )


def test_vqvae_then_prior(tmp_path):
    train(_cfg(tmp_path / "stage1", SMALL_VQVAE))
    stage1 = _run_dir(tmp_path / "stage1") / "checkpoints" / "last.ckpt"
    assert VQVAELightning.load_from_checkpoint(stage1).config.num_embeddings == 32

    prior_spec = {
        "name": "vqvae_prior",
        "params": {
            "num_embeddings": 32,
            "hidden_dim": 16,
            "num_layers": 3,
            "vqvae_checkpoint": str(stage1),
        },
    }
    train(_cfg(tmp_path / "stage2", prior_spec))
    prior = PriorLightning.load_from_checkpoint(
        _run_dir(tmp_path / "stage2") / "checkpoints" / "last.ckpt"
    )
    assert prior.sample(n_samples=2, seq_length=SEQ).shape == (2, 12, SEQ)


def test_pulse2pulse_with_callbacks(tmp_path):
    cfg = _cfg(
        tmp_path,
        {
            "name": "pulse2pulse",
            "params": {
                "model_size": 8,
                "num_channels": 8,
                "seq_length": 5000,
                "n_critic": 1,
            },
        },
        leads="independent8",
        seq_length=5000,
    )
    cfg["checkpoint"] = {"monitor": "val/d_wasserstein", "mode": "max"}
    cfg["callbacks"] = [
        {"name": "ecg_visualization", "params": {"every_n_epochs": 1, "n_samples": 1}},
        {"name": "generated_samples", "params": {"every_n_epochs": 1, "n_samples": 2}},
    ]
    train(cfg)
    samples = _run_dir(tmp_path) / "samples"
    assert (
        samples / "generated_epoch_0000.pt"
    ).is_file()  # save_dir defaulted to ${run_dir}/samples
    model = Pulse2PulseGAN.load_from_checkpoint(
        _run_dir(tmp_path) / "checkpoints" / "last.ckpt"
    )
    assert model.sample(2).shape == (2, 8, 5000)


def test_channel_mismatch_is_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="expects 12 channels but the dataset yields 8"
    ):
        train(_cfg(tmp_path, SMALL_VAE, leads="independent8"))


def test_target_import_path_and_legacy_config_key(tmp_path):
    spec = {
        "target": "ecgen.families.vqvae.VQVAELightning",
        "params": {
            "config": {"base_channels": 16, "latent_channels": 8, "num_embeddings": 16}
        },
    }
    train(_cfg(tmp_path, spec))
    model = VQVAELightning.load_from_checkpoint(
        _run_dir(tmp_path) / "checkpoints" / "last.ckpt"
    )
    assert model.config == VQVAEConfig(
        base_channels=16,
        latent_channels=8,
        num_embeddings=16,
        channel_multipliers=(1, 2, 4, 4),
    )


def test_seeded_runs_are_reproducible(tmp_path):
    for name in ("a", "b"):
        train(_cfg(tmp_path / name, SMALL_VAE))
    a, b = (
        torch.load(
            _run_dir(tmp_path / n) / "checkpoints" / "last.ckpt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        for n in ("a", "b")
    )
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- config


def test_apply_overrides():
    cfg = {
        "trainer": {"devices": [0]},
        "callbacks": [{"name": "x", "params": {"n": 1}}],
    }
    out = apply_overrides(
        cfg,
        [
            "trainer.devices=[1, 2]",
            "callbacks.0.params.n=5",
            "model.params.lr=1e-3",
            "resume=null",
        ],
    )
    assert out["trainer"]["devices"] == [1, 2]
    assert out["callbacks"][0]["params"]["n"] == 5
    assert out["model"]["params"]["lr"] == 1e-3 and out["resume"] is None
    assert cfg["trainer"]["devices"] == [0]  # input untouched
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["trainer.devices"])


def test_resolve_config_defaults_and_interpolation(tmp_path):
    cfg = {
        "experiment": {"name": "exp", "seed": 3, "runs_root": str(tmp_path)},
        "model": {"name": "vae"},
        "data": {"dataset": {"name": "synthetic"}},
        "callbacks": [
            {
                "name": "x",
                "params": {"save_dir": "${run_dir}/plots", "tag": "${name}-${seed}"},
            }
        ],
    }
    resolved, paths = resolve_config(cfg)
    assert paths.run_dir == tmp_path / "exp" / "seed_3"
    assert resolved["callbacks"][0]["params"] == {
        "save_dir": f"{paths.run_dir}/plots",
        "tag": "exp-3",
    }
    assert (
        resolved["checkpoint"]["monitor"] == "val_loss"
        and resolved["checkpoint"]["save_last"] is True
    )
    with pytest.raises(ValueError, match="Unknown config sections"):
        resolve_config({**cfg, "visualization": {}})


def test_cli_print_config(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_cfg(tmp_path, SMALL_VAE)))
    assert cli_main(["train", str(path), "--print-config", "trainer.max_epochs=7"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["trainer"]["max_epochs"] == 7


EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


@pytest.mark.parametrize("name", sorted(p.stem for p in EXPERIMENTS.glob("*.yaml")))
def test_shipped_experiment_configs_resolve(name):
    repo = Path(__file__).resolve().parents[2]
    cfg, _ = resolve_config(
        yaml.safe_load((repo / "configs" / "experiments" / f"{name}.yaml").read_text())
    )
    assert cfg["data"]["dataset"]["name"] in DATASETS
    if any(e.get("name") == "text" for e in ((cfg["model"].get("params") or {}).get("conditioning") or {}).get("encoders", [])):
        return  # needs a downloaded Hugging Face model
    # Build the model too (catches bad params), except priors that need a stage-1 file
    if "vqvae_checkpoint" not in cfg["model"].get("params", {}):
        assert build_model(cfg["model"]) is not None
