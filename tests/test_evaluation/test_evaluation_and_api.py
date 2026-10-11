"""
Evaluation metrics, the adherence classifier, ecgen.sample / ecgen sample,
ecgen evaluate, and backward compatibility of ecgen.generate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import wfdb
import yaml

import ecgen
from ecgen.cli import main as cli_main
from ecgen.cli import parse_cond
from ecgen.data.datasets import SyntheticECG
from ecgen.data.leads import INDEPENDENT_8, STANDARD_12, derive_standard12
from ecgen.evaluation import (
    distribution_metrics,
    einthoven_residual,
    fit_classifier,
    frechet_distance,
    label_adherence,
    memorization_ratio,
    mmd_rbf,
    nearest_neighbour_distances,
    precision_recall,
    reconstruction_metrics,
)
from ecgen.training.runner import train


def _synthetic(split: str, n: int, seq: int = 1000) -> tuple[np.ndarray, list]:
    ds = SyntheticECG(split, num_samples=n, seq_length=seq)
    items = [ds[i] for i in range(n)]
    return np.stack([s["signal"].numpy() for s in items]), items


# ---------------------------------------------------------------- metrics


def test_distribution_metrics_separate_real_from_noise():
    a, _ = _synthetic("train", 40)
    b, _ = _synthetic("val", 40)
    noise = np.random.default_rng(0).standard_normal(b.shape).astype(np.float32)
    same, diff = (
        distribution_metrics(a, b, fs=500),
        distribution_metrics(a, noise, fs=500),
    )
    assert diff["frechet_distance"] > 10 * same["frechet_distance"]
    assert diff["mmd"] > same["mmd"]
    assert same["precision"] > 0.5 and diff["precision"] == 0.0


def test_basic_metric_properties():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((200, 5))
    assert frechet_distance(f, f) == pytest.approx(0.0, abs=1e-6)
    assert abs(mmd_rbf(f[:100], f[100:])) < 0.05
    pr = precision_recall(f[:100], f[100:], k=3)
    assert 0.5 < pr["precision"] <= 1 and 0.5 < pr["recall"] <= 1


def test_einthoven_and_lead_derivation():
    x8 = torch.randn(3, 8, 64)
    x12 = derive_standard12(x8, list(INDEPENDENT_8))
    assert x12.shape == (3, 12, 64) and isinstance(x12, torch.Tensor)
    assert einthoven_residual(x12, list(STANDARD_12))["mean"] < 1e-5
    assert einthoven_residual(torch.randn(3, 12, 64), list(STANDARD_12))["mean"] > 0.5
    assert isinstance(derive_standard12(x8.numpy(), list(INDEPENDENT_8)), np.ndarray)
    with pytest.raises(ValueError):
        derive_standard12(x8[:, :7], list(INDEPENDENT_8)[:7])


def test_reconstruction_and_memorisation():
    true = np.random.default_rng(0).standard_normal((2, 3, 50))
    pred = true.copy()
    pred[:, 1] += 1.0
    m = reconstruction_metrics(pred, true, np.array([False, True, False]))
    assert m["mae"] == pytest.approx(1.0) and m["pearson_r"] == pytest.approx(1.0)
    train_x = np.random.default_rng(1).standard_normal((20, 2, 10))
    assert nearest_neighbour_distances(train_x[:3], train_x).max() == pytest.approx(
        0.0, abs=1e-4
    )
    copied = memorization_ratio(
        train_x[:5], train_x, np.random.default_rng(2).standard_normal((5, 2, 10))
    )
    assert copied["ratio"] < 0.1


def test_classifier_learns_synthetic_rhythm():
    x, items = _synthetic("train", 256, seq=2000)
    y = np.stack([s["cond"]["labels"].numpy() for s in items])
    model = fit_classifier(x, y, multilabel=True, epochs=15, width=16, device="cpu")
    xv, iv = _synthetic("val", 80, seq=2000)
    yv = np.stack([s["cond"]["labels"].numpy() for s in iv])
    result = label_adherence(model, xv, yv)
    assert result["macro_auroc"] > 0.9


# -------------------------------------------------------------------- API


def _train_tiny(tmp_path: Path) -> Path:
    cfg = {
        "experiment": {"name": "s", "seed": 0, "runs_root": str(tmp_path)},
        "model": {
            "name": "diffusion",
            "params": {
                "in_channels": 8,
                "backbone": {
                    "name": "unet1d",
                    "params": {
                        "base_channels": 16,
                        "channel_mults": [1, 2],
                        "attn_levels": [1],
                    },
                },
                "warmup_steps": 1,
                "conditioning": {
                    "vector_dim": 16,
                    "encoders": [
                        {
                            "field": "measurements",
                            "name": "numeric",
                            "params": {"num_features": 2, "dim": 16},
                        },
                        {
                            "field": "rhythm",
                            "name": "categorical",
                            "params": {"num_classes": 3, "dim": 16},
                        },
                    ],
                },
            },
        },
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {
                    "num_samples": 8,
                    "seq_length": 256,
                    "leads": "independent8",
                },
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
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    train(cfg)
    return tmp_path / "s" / "seed_0" / "checkpoints" / "last.ckpt"


def test_sample_api_writers_and_per_sample_conditions(tmp_path):
    ckpt = _train_tiny(tmp_path)
    assert (
        torch.load(ckpt, map_location="cpu", weights_only=False)["ecgen_family"]
        == "diffusion"
    )

    x = ecgen.sample(
        ckpt,
        3,
        cond={"measurements": [70.0, 1.0], "rhythm": [0, 1, 2]},
        steps=2,
        seed=0,
        batch_size=2,
        device="cpu",
        output_dir=tmp_path / "npy",
    )
    assert x.shape == (3, 8, 256) and (tmp_path / "npy" / "samples.npy").is_file()
    again = ecgen.sample(
        ckpt,
        3,
        cond={"measurements": [70.0, 1.0], "rhythm": [0, 1, 2]},
        steps=2,
        seed=0,
        batch_size=2,
        device="cpu",
    )
    assert np.allclose(x, again)  # seeded

    x12 = ecgen.sample(
        ckpt,
        2,
        steps=2,
        device="cpu",
        derive_12=True,
        output_dir=tmp_path / "csv",
        format="csv",
        plot=1,
    )
    assert x12.shape == (2, 12, 256)
    meta = json.loads((tmp_path / "csv" / "samples.json").read_text())
    assert meta["lead_names"] == list(STANDARD_12) and meta["units"] == "mV"
    header = (tmp_path / "csv" / "sample_1.csv").read_text().splitlines()[0]
    assert header.split(",") == list(STANDARD_12)
    assert (tmp_path / "csv" / "sample_1.html").is_file()

    ecgen.sample(
        ckpt, 2, steps=2, device="cpu", output_dir=tmp_path / "wfdb", format="wfdb"
    )
    rec = wfdb.rdrecord(str(tmp_path / "wfdb" / "sample_1"))
    assert rec.sig_name == list(INDEPENDENT_8) and rec.fs == 500


def test_sample_needs_family_for_old_checkpoints(tmp_path):
    ckpt = _train_tiny(tmp_path)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    del state["ecgen_family"]
    old = tmp_path / "old.ckpt"
    torch.save(state, old)
    with pytest.raises(ValueError, match="pass family="):
        ecgen.load_model(old, device="cpu")
    assert ecgen.sample(old, 1, family="diffusion", steps=2, device="cpu").shape == (
        1,
        8,
        256,
    )


def test_cli_sample_and_evaluate(tmp_path, capsys):
    ckpt = _train_tiny(tmp_path)
    out = tmp_path / "cli"
    assert (
        cli_main(
            [
                "sample",
                str(ckpt),
                "-n",
                "4",
                "-o",
                str(out),
                "--steps",
                "2",
                "--device",
                "cpu",
                "--cond",
                "measurements=80,nan",
                "--cond",
                "rhythm=2",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    assert np.load(out / "samples.npy").shape == (4, 8, 256)
    capsys.readouterr()
    assert (
        cli_main(
            [
                "evaluate",
                str(out / "samples.npy"),
                str(tmp_path / "cfg.yaml"),
                "-n",
                "8",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert {
        "frechet_distance",
        "mmd",
        "precision",
        "recall",
        "inter_lead_correlation_error",
    } <= set(result)


def test_parse_cond():
    c = parse_cond(["m=1,2,nan", "report='Sinus rhythm'", "rhythm=2"])
    assert c["m"][:2] == [1.0, 2.0] and np.isnan(c["m"][2])
    assert c["report"] == "Sinus rhythm" and c["rhythm"] == 2.0


# ------------------------------------------------- legacy ecgen.generate


def test_legacy_generate_still_works(tmp_path):
    from ecgen.families.gan import WaveGANGenerator

    torch.save(
        WaveGANGenerator(model_size=10, num_channels=8).state_dict(), tmp_path / "g.pt"
    )
    paths = ecgen.generate(
        str(tmp_path / "g.pt"),
        n_samples=2,
        output_dir=str(tmp_path / "gen"),
        model_size=10,
        device="cpu",
        format="csv",
    )
    csvs = sorted((tmp_path / "gen").glob("*.csv"))
    assert len(csvs) == 2 and paths is not None
