"""
Text conditioning (hash and pretrained encoders), string fields through
collation, and the PTB-XL loader on a fake PTB-XL tree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from ecgen.conditioning import CompositeConditioner
from ecgen.conditioning.text import HashTextEncoder, tokenize
from ecgen.data.collate import ECGCollate
from ecgen.data.datasets import PTBXL, SyntheticECG
from ecgen.families import DiffusionModule
from ecgen.training.runner import train

# ------------------------------------------------------------------- text


def test_tokenize_and_hash_encoder():
    assert tokenize("Sinus-Rhythmus, normales EKG!") == [
        "sinus",
        "rhythmus",
        "normales",
        "ekg",
    ]
    enc = HashTextEncoder(dim=8, max_tokens=4)
    tok, mask = enc(["sinus rhythm", "", "a b c d e f"])
    assert tok.shape == (3, 4, 8)
    assert mask.sum(1).tolist() == [2, 0, 4]
    assert torch.equal(tok[1], torch.zeros(4, 8))
    a, _ = enc(["sinus rhythm"])
    assert torch.allclose(a[0], tok[0])  # deterministic hashing
    pooled = HashTextEncoder(dim=8, as_tokens=False)(["sinus rhythm"])
    assert pooled.shape == (1, 8)


def test_synthetic_report_and_string_collation():
    ds = SyntheticECG(num_samples=4, seq_length=128)
    batch = ECGCollate(ds.lead_names, ds.fs)([ds[i] for i in range(4)])
    assert isinstance(batch.cond["report"], list) and len(batch.cond["report"]) == 4
    assert all(r.startswith("sinus") for r in batch.cond["report"])
    for i in range(4):
        hr = ds[i]["meta"]["heart_rate"]
        expected = (
            "bradycardia" if hr < 60 else ("tachycardia" if hr > 100 else "rhythm")
        )
        assert expected in ds[i]["cond"]["report"]


def test_composite_with_text_tokens_and_dropout():
    c = CompositeConditioner(
        encoders=[
            {
                "field": "report",
                "name": "text_hash",
                "params": {"dim": 16, "max_tokens": 6},
            }
        ],
        token_dim=8,
        dropout={"report": 1.0},
    ).train()
    out = c({"report": ["sinus rhythm", "sinus tachycardia"]})
    assert out.tokens.shape == (2, 6, 8)
    assert (
        out.tokens_mask.tolist() == [[True] + [False] * 5] * 2
    )  # all dropped -> null token only


def _tiny_hf_model(path: Path) -> str:
    transformers = pytest.importorskip("transformers")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "sinus",
        "rhythm",
        "tachycardia",
        "high",
        "voltage",
        ",",
    ]
    path.mkdir()
    (path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizer(str(path / "vocab.txt"))
    tok.save_pretrained(str(path))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    transformers.BertModel(config).save_pretrained(str(path))
    return str(path)


def test_pretrained_text_encoder_is_frozen_cached_and_not_checkpointed(tmp_path):
    from ecgen.conditioning.text import PretrainedTextEncoder

    enc = PretrainedTextEncoder(_tiny_hf_model(tmp_path / "bert"), max_length=8)
    tok, mask = enc(["sinus rhythm", "sinus tachycardia, high voltage", ""])
    assert tok.shape[0] == 3 and tok.shape[2] == 16 and enc.out_dim == 16
    assert mask[2].sum() == 0 and mask[0].sum() > 0
    assert len(enc._cache) == 3
    assert (
        list(enc.state_dict()) == []
    )  # HF weights stay out of the generative checkpoint
    assert all(not p.requires_grad for p in enc._model.module.parameters())


def test_text_conditioned_diffusion_end_to_end(tmp_path):
    cfg = {
        "experiment": {"name": "t", "seed": 0, "runs_root": str(tmp_path)},
        "model": {
            "name": "diffusion",
            "params": {
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
                    "token_dim": 16,
                    "encoders": [
                        {"field": "report", "name": "text_hash", "params": {"dim": 16}}
                    ],
                    "dropout": {"report": 0.2},
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
    model = DiffusionModule.load_from_checkpoint(
        tmp_path / "t" / "seed_0" / "checkpoints" / "last.ckpt"
    )
    x = model.sample(2, cond={"report": "sinus tachycardia"}, guidance=2.0, steps=3)
    y = model.sample(
        2, cond={"report": ["sinus rhythm", "sinus bradycardia, high voltage"]}, steps=3
    )
    assert x.shape == y.shape == (2, 12, 256)
    with pytest.raises(ValueError, match="3 strings for 2 samples"):
        model.sample(2, cond={"report": ["a", "b", "c"]}, steps=2)


# ------------------------------------------------------------------ PTB-XL


def _fake_ptbxl(root: Path, n: int = 20) -> Path:
    root.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        rel = f"records500/00000/{i:05d}_hr"
        (root / "records500" / "00000").mkdir(parents=True, exist_ok=True)
        sig = np.tile(
            np.arange(1, 13, dtype=np.float64), (500, 1)
        ) + 0.01 * rng.standard_normal((500, 12))
        names = [
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        wfdb.wrsamp(
            f"{i:05d}_hr",
            fs=500,
            units=["mV"] * 12,
            sig_name=names,
            p_signal=sig,
            fmt=["16"] * 12,
            adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
            write_dir=str(root / "records500" / "00000"),
        )
        codes = {"NORM": 100.0, "SR": 0.0} if i % 2 else {"IMI": 50.0, "STACH": 0.0}
        rows.append(
            {
                "ecg_id": i,
                "patient_id": 1000 + i,
                "age": 40 + i,
                "sex": i % 2,
                "height": np.nan if i % 3 else 170.0,
                "weight": 70.0,
                "report": "sinusrhythmus normales ekg",
                "scp_codes": str(codes),
                "strat_fold": (i % 10) + 1,
                "validated_by_human": True,
                "filename_lr": rel.replace("500", "100").replace("_hr", "_lr"),
                "filename_hr": rel,
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    stmts = pd.DataFrame(
        {
            "description": [
                "normal ECG",
                "inferior MI",
                "sinus rhythm",
                "sinus tachycardia",
            ],
            "diagnostic": [1.0, 1.0, np.nan, np.nan],
            "form": [np.nan, np.nan, np.nan, np.nan],
            "rhythm": [np.nan, np.nan, 1.0, 1.0],
            "diagnostic_class": ["NORM", "MI", np.nan, np.nan],
            "diagnostic_subclass": ["NORM", "IMI", np.nan, np.nan],
        },
        index=["NORM", "IMI", "SR", "STACH"],
    )
    stmts.to_csv(root / "scp_statements.csv")
    return root


def test_ptbxl_fields_splits_and_vocabularies(tmp_path):
    root = str(_fake_ptbxl(tmp_path / "ptbxl"))
    train_ds, test_ds = (
        PTBXL(root, "train", seq_length=500),
        PTBXL(root, "test", seq_length=500),
    )
    assert set(train_ds.records["strat_fold"]) <= set(range(1, 9)) and set(
        test_ds.records["strat_fold"]
    ) == {10}
    s = train_ds[0]
    assert s["signal"].shape == (12, 500)
    assert set(s["cond"]) == {"scp", "demographics", "sex", "report"}
    assert train_ds.label_names == ["NORM", "IMI", "SR", "STACH"] and s["cond"][
        "scp"
    ].shape == (4,)

    sup = PTBXL(root, labels="superclass", seq_length=500)
    assert sup.label_names == ["MI", "NORM"]
    rhythm = PTBXL(root, labels="rhythm", seq_length=500)
    assert rhythm.label_names == ["SR", "STACH"]
    strict = PTBXL(root, min_likelihood=60, seq_length=500)
    imi = next(
        i for i in range(len(strict)) if strict.records.iloc[i]["ecg_id"] % 2 == 0
    )
    assert strict[imi]["cond"]["scp"].sum() == 0  # IMI (50) and STACH (0) are below 60

    # demographics are normalized with the training folds and report missing height
    norm = train_ds.field_normalizers["demographics"]
    assert norm.columns == ["age", "height", "weight"]
    assert test_ds.field_normalizers["demographics"].to_dict() == norm.to_dict()
    missing = [
        train_ds[i]["cond_missing"]["demographics"][1].item()
        for i in range(len(train_ds))
    ]
    assert any(missing) and not all(missing)


def test_ptbxl_scale_none_keeps_millivolts_in_standard_lead_order(tmp_path):
    root = str(_fake_ptbxl(tmp_path / "ptbxl"))
    ds = PTBXL(root, seq_length=500, scale="none", leads="independent8")
    assert ds[0]["signal"].mean(dim=1).round().tolist() == [1, 2, 7, 8, 9, 10, 11, 12]
