"""
Field encoders, the composite conditioner, injection layers, masks, and the
conditioning round trip through the runner and a checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ecgen.conditioning import CompositeConditioner
from ecgen.conditioning.encoders import (
    CategoricalEncoder,
    FourierScalarEncoder,
    MaskedSignalEncoder,
    MultiLabelEncoder,
    NumericEncoder,
)
from ecgen.data.datamodule import ECGDataModule
from ecgen.data.datasets import SyntheticECG
from ecgen.data.leads import STANDARD_12
from ecgen.data.masks import build_mask
from ecgen.families import VAELightning
from ecgen.nn.injection import (
    AdaGN,
    AdaLNZero,
    AddToTimeEmb,
    CrossAttention1d,
    FiLM,
    concat_dense,
)
from ecgen.training.runner import train
from tests.test_conditioning.toy import ToyConditional

B, F_, T = 4, 3, 64

# --------------------------------------------------------------- encoders


def test_categorical_unknown_maps_to_extra_row():
    enc = CategoricalEncoder(num_classes=3, dim=8)
    out = enc(torch.tensor([0, 2, -1, 1]), torch.tensor([False, False, False, True]))
    assert out.shape == (4, 8)
    unknown = enc.embed.weight[3]
    assert torch.equal(out[2], unknown) and torch.equal(out[3], unknown)


def test_multilabel_vector_and_tokens():
    y = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.float32)
    assert MultiLabelEncoder(3, dim=8)(y).shape == (2, 8)
    tok, mask = MultiLabelEncoder(3, dim=8, as_tokens=True)(y)
    assert tok.shape == (2, 3, 8) and mask.tolist() == [
        [True, False, True],
        [False, False, False],
    ]


def test_numeric_encoder_ignores_values_marked_missing():
    enc = NumericEncoder(F_, dim=8)
    x = torch.randn(B, F_)
    m = torch.tensor([[False, True, False]] * B)
    x2 = x.clone()
    x2[:, 1] = 1e6  # garbage in a missing slot must not matter
    assert torch.allclose(enc(x, m), enc(x2, m))
    tok_enc = NumericEncoder(F_, dim=8, as_tokens=True)
    tok, tmask = tok_enc(x, m)
    assert tok.shape == (B, F_, 8) and tmask.all()
    assert torch.equal(tok[:, 1], tok_enc.missing_token[1].expand(B, -1))


def test_fourier_and_masked_signal_shapes():
    assert FourierScalarEncoder(F_, dim=16, n_freqs=4)(torch.randn(B, F_)).shape == (
        B,
        16,
    )
    obs, mask = torch.randn(B, 12, T), torch.ones(B, 12, T)
    assert MaskedSignalEncoder(12)((obs, mask)).shape == (B, 24, T)
    assert MaskedSignalEncoder(12, channels=16)((obs, mask)).shape == (B, 16, T)


# ------------------------------------------------------------- composite


def _conditioner(dropout=None) -> CompositeConditioner:
    return CompositeConditioner(
        encoders=[
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
            {
                "field": "labels",
                "name": "multilabel",
                "params": {"num_labels": 3, "dim": 16, "as_tokens": True},
            },
            {"field": "obs", "name": "masked_signal", "params": {"num_leads": 12}},
        ],
        vector_dim=32,
        token_dim=24,
        dropout=dropout,
    )


def _inputs():
    cond = {
        "measurements": torch.randn(B, 2),
        "rhythm": torch.tensor([0, 1, 2, 1]),
        "labels": torch.tensor(
            [[1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ),
    }
    return cond, torch.randn(B, 12, T), torch.ones(B, 12, T)


def test_composite_slots_and_shapes():
    c = _conditioner()
    cond, obs, mask = _inputs()
    out = c(cond, obs=obs, obs_mask=mask)
    assert c.slots == {"vector", "tokens", "dense"} and c.dense_channels == 24
    assert out.vector.shape == (B, 32)
    assert out.tokens.shape == (B, 3, 24) and out.tokens_mask.shape == (B, 3)
    assert out.dense.shape == (B, 24, T)
    assert c.cond_fields == ["measurements", "rhythm", "labels"] and c.uses_obs


def test_drop_all_equals_absent_fields():
    c = _conditioner().eval()
    cond, obs, mask = _inputs()
    dropped = c(cond, obs=obs, obs_mask=mask, drop="all")
    absent = c({}, batch_size=B, obs=obs, obs_mask=mask * 0)
    null_vec = c.null_vector["measurements"] + c.null_vector["rhythm"]
    assert torch.allclose(dropped.vector, null_vec.expand(B, -1))
    assert torch.allclose(absent.vector, dropped.vector)
    assert torch.equal(dropped.tokens_mask, torch.tensor([[True, False, False]] * B))
    assert torch.equal(dropped.dense, torch.zeros_like(dropped.dense))


def test_training_dropout_gives_null_rows():
    c = _conditioner(dropout={"measurements": 1.0}).train()
    cond, obs, mask = _inputs()
    full = c(cond, obs=obs, obs_mask=mask, drop=None)
    only_rhythm = _conditioner().eval()
    only_rhythm.load_state_dict(c.state_dict())
    expected = only_rhythm(cond, obs=obs, obs_mask=mask, drop="measurements")
    assert torch.allclose(full.vector, expected.vector)

    c.eval()  # no dropout at eval
    assert not torch.allclose(c(cond, obs=obs, obs_mask=mask).vector, expected.vector)


def test_per_row_drop_mask():
    c = _conditioner().eval()
    cond, obs, mask = _inputs()
    rows = torch.tensor([True, False, False, True])
    out = c(cond, obs=obs, obs_mask=mask, drop={"measurements": rows, "rhythm": rows})
    ref_null = c(cond, obs=obs, obs_mask=mask, drop=["measurements", "rhythm"])
    ref_full = c(cond, obs=obs, obs_mask=mask)
    assert torch.allclose(out.vector[rows], ref_null.vector[rows])
    assert torch.allclose(out.vector[~rows], ref_full.vector[~rows])


def test_config_errors():
    with pytest.raises(ValueError, match="empty"):
        CompositeConditioner(encoders=[])
    with pytest.raises(ValueError, match="without encoders"):
        CompositeConditioner(
            encoders=[
                {"field": "rhythm", "name": "categorical", "params": {"num_classes": 3}}
            ],
            dropout={"other": 0.1},
        )
    with pytest.raises(ValueError, match="needs obs"):
        _conditioner()(_inputs()[0])


# -------------------------------------------------------------- injection


def test_injection_layers_start_as_identity():
    h = torch.randn(B, 16, T)
    c = torch.randn(B, 8)
    assert torch.equal(FiLM(8, 16)(h, c), h)
    ada = AdaGN(8, 16)
    assert torch.allclose(ada(h, c), ada.norm(h))
    temb = torch.randn(B, 32)
    assert torch.equal(AddToTimeEmb(8, 32)(temb, c), temb)
    x, gate = AdaLNZero(8, 16)(torch.randn(B, 5, 16), c)
    assert x.shape == (B, 5, 16) and torch.equal(gate, torch.zeros(B, 1, 16))
    ca = CrossAttention1d(16, context_dim=8, heads=4)
    assert torch.equal(ca(h, torch.randn(B, 3, 8)), h)


def test_cross_attention_masks_and_no_nan():
    ca = CrossAttention1d(16, context_dim=8, heads=4)
    torch.nn.init.normal_(ca.out.weight)
    h, tokens = torch.randn(2, 16, T), torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    out = ca(h, tokens, mask)
    assert torch.isfinite(out).all()
    changed = tokens.clone()
    changed[0, 1:] = 99.0  # masked-out tokens of row 0 must not matter
    assert torch.allclose(ca(h, changed, mask)[0], out[0], atol=1e-5)


def test_concat_dense_resamples():
    x, d = torch.randn(B, 4, 100), torch.randn(B, 2, 50)
    assert concat_dense(x, d).shape == (B, 6, 100)
    assert concat_dense(x, None) is x


# ------------------------------------------------------------------ masks


def test_mask_generators():
    leads = list(STANDARD_12)
    g = torch.Generator().manual_seed(0)
    m = build_mask({"name": "lead_subset", "params": {"keep": [["I", "V1"]]}}, leads)(
        12, T, g
    )
    assert m.sum(dim=1).tolist() == [T if n in ("I", "V1") else 0 for n in leads]

    for _ in range(20):
        m = build_mask(
            {"name": "lead_subset", "params": {"min_leads": 2, "max_leads": 3}}, leads
        )(12, T, g)
        assert 2 <= int(m[:, 0].sum()) <= 3

    span = build_mask(
        {"name": "time_span", "params": {"min_frac": 0.2, "max_frac": 0.4}}, leads
    )
    fracs = [1 - span(12, 1000, g)[0].mean().item() for _ in range(50)]
    assert 0.19 <= min(fracs) and max(fracs) <= 0.41

    m = build_mask(
        {"name": "forecast", "params": {"min_frac": 0.5, "max_frac": 0.5}}, leads
    )(12, T, g)
    assert m[:, : T // 2].all() and not m[:, T // 2 :].any()

    p = build_mask({"name": "random_points", "params": {"p_observed": 0.3}}, leads)(
        12, 2000, g
    ).mean()
    assert abs(float(p) - 0.3) < 0.02

    mix = build_mask(
        {
            "name": "mixture",
            "params": {
                "components": [
                    {"weight": 3, "name": "none"},
                    {"weight": 1, "name": "unobserved"},
                ]
            },
        },
        leads,
    )
    share = np.mean([mix(1, 4, g).mean().item() for _ in range(2000)])
    assert abs(share - 0.75) < 0.04

    with pytest.raises(ValueError, match="unknown leads"):
        build_mask({"name": "lead_subset", "params": {"keep": [["X"]]}}, leads)


def test_datamodule_adds_obs_mask():
    dm = ECGDataModule(
        dataset={"name": "synthetic", "params": {"num_samples": 4, "seq_length": T}},
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        masks={"name": "lead_subset", "params": {"keep": [["I"]]}},
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.obs_mask.shape == (4, 12, T)
    assert (
        torch.equal(batch.obs_mask[:, 0], torch.ones(4, T))
        and batch.obs_mask[:, 1:].sum() == 0
    )
    assert set(dm.state_dict()["normalizers"]) == {"measurements"}


# ---------------------------------------------------- families + runner


def test_unconditional_family_rejects_conditioning():
    model = VAELightning()
    with pytest.raises(ValueError, match="does not support conditioning"):
        model.init_conditioning(
            {
                "encoders": [
                    {
                        "field": "rhythm",
                        "name": "categorical",
                        "params": {"num_classes": 3},
                    }
                ]
            }
        )


def test_slot_validation():
    spec = {
        "encoders": [
            {
                "field": "labels",
                "name": "multilabel",
                "params": {"num_labels": 3, "as_tokens": True},
            }
        ]
    }
    with pytest.raises(ValueError, match="accepts condition slots"):
        ToyConditional(conditioning=spec)


TOY_COND = {
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
    "dropout": {"measurements": 0.1},
}


def _toy_cfg(tmp_path: Path, conditioning=TOY_COND, masks=None) -> dict:
    return {
        "experiment": {"name": "toy", "seed": 0, "runs_root": str(tmp_path)},
        "model": {
            "target": "tests.test_conditioning.toy.ToyConditional",
            "params": {"conditioning": conditioning},
        },
        "data": {
            "dataset": {
                "name": "synthetic",
                "params": {"num_samples": 16, "seq_length": 256},
            },
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "masks": masks,
        },
        "trainer": {
            "accelerator": "cpu",
            "max_epochs": 2,
            "enable_progress_bar": False,
            "num_sanity_val_steps": 0,
        },
    }


def test_round_trip_sample_with_raw_condition(tmp_path):
    train(_toy_cfg(tmp_path))
    ckpt = tmp_path / "toy" / "seed_0" / "checkpoints" / "last.ckpt"
    model = ToyConditional.load_from_checkpoint(ckpt).eval()

    info = model.data_info
    assert info["lead_names"] == list(STANDARD_12) and info["fs"] == 500
    assert info["normalizers"]["measurements"]["columns"] == ["heart_rate", "amplitude"]

    # Raw units are normalized with the stored training statistics
    values, missing = model.prepare_condition(
        {"measurements": [80.0, float("nan")]}, n=3
    )
    norm = SyntheticECG().feature_normalizer
    expected, exp_missing = norm.transform([80.0, float("nan")])
    assert torch.allclose(
        values["measurements"], torch.as_tensor(expected).expand(3, -1)
    )
    assert missing["measurements"].tolist() == [exp_missing.tolist()] * 3

    out = model.sample(3, cond={"measurements": [80.0, 1.0], "rhythm": 1})
    assert out.shape == (3, 12, 256)
    unconditional = model.sample(3, cond={})  # all fields absent -> null embeddings
    assert unconditional.shape == (3, 12, 256)


def test_runner_checks_condition_fields(tmp_path):
    cond = {
        "encoders": [{"field": "age", "name": "numeric", "params": {"num_features": 1}}]
    }
    with pytest.raises(
        ValueError, match=r"fields \['age'\] that the dataset does not provide"
    ):
        train(_toy_cfg(tmp_path, conditioning=cond))

    obs = {
        "encoders": [
            {"field": "obs", "name": "masked_signal", "params": {"num_leads": 12}}
        ]
    }
    with pytest.raises(ValueError, match="data.masks is not set"):
        train(_toy_cfg(tmp_path, conditioning=obs))
