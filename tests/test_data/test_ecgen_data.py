"""
Dict datasets, collation to ECGBatch, the feature normalizer, and ECGDataModule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb
from pytorch_lightning.utilities import move_data_to_device

from ecgen.data.collate import ECGCollate
from ecgen.data.datamodule import ECGDataModule
from ecgen.data.datasets import MIMICIVECG, SyntheticECG, scale_signal
from ecgen.data.datasets.mimic import MEASUREMENT_COLUMNS, MIMIC_LEAD_ORDER
from ecgen.data.leads import INDEPENDENT_8, resolve_leads
from ecgen.data.normalizers import FeatureNormalizer
from ecgen.types import ECGBatch, get_signal

# ----------------------------------------------------------------- fake MIMIC

N_SUBJECTS = 20
SEQ = 600


def _write_fake_mimic(root, nan_rows=(3,)):
    """20 subjects x 2 studies; lead k of every record is the constant k+1 (mV)."""
    rows = []
    rng = np.random.default_rng(0)
    for s in range(N_SUBJECTS):
        subject_id = 10000000 + s
        for j in range(2):
            study_id = 40000000 + 10 * s + j
            rec_dir = (
                root
                / "files"
                / f"p{str(subject_id)[:4]}"
                / f"p{subject_id}"
                / f"s{study_id}"
            )
            rec_dir.mkdir(parents=True)
            signal = np.tile(
                np.arange(1, 13, dtype=np.float64), (SEQ, 1)
            ) + 0.01 * rng.standard_normal((SEQ, 12))
            wfdb.wrsamp(
                str(study_id),
                fs=500,
                units=["mV"] * 12,
                sig_name=list(MIMIC_LEAD_ORDER),
                p_signal=signal,
                fmt=["16"] * 12,
                adc_gain=[200.0] * 12,
                baseline=[0] * 12,
                write_dir=str(rec_dir),
            )
            row = {"subject_id": subject_id, "study_id": study_id}
            row.update({c: float(rng.integers(50, 900)) for c in MEASUREMENT_COLUMNS})
            rows.append(row)
    table = pd.DataFrame(rows)
    for i in nan_rows:
        table.loc[i, "qrs_axis"] = np.nan
    table.to_csv(root / "machine_measurements.csv", index=False)
    return table


@pytest.fixture(scope="module")
def mimic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic")
    _write_fake_mimic(root)
    return root


def test_mimic_splits_are_disjoint_by_subject(mimic_root):
    subjects = {
        split: set(
            MIMICIVECG(str(mimic_root), split=split, seq_length=SEQ).records[
                "subject_id"
            ]
        )
        for split in ("train", "val", "test")
    }
    assert subjects["train"] and subjects["val"] and subjects["test"]
    assert not (subjects["train"] & subjects["val"]) and not (
        subjects["train"] & subjects["test"]
    )


def test_mimic_normalizer_fit_on_train_for_every_split(mimic_root):
    train = MIMICIVECG(str(mimic_root), split="train", seq_length=SEQ)
    val = MIMICIVECG(str(mimic_root), split="val", seq_length=SEQ)
    assert val.feature_normalizer.to_dict() == train.feature_normalizer.to_dict()
    expected = FeatureNormalizer.fit(train.records, MEASUREMENT_COLUMNS)
    assert train.feature_normalizer.to_dict() == expected.to_dict()


def test_mimic_max_samples_does_not_change_normalizer(mimic_root):
    full = MIMICIVECG(str(mimic_root), seq_length=SEQ)
    small = MIMICIVECG(str(mimic_root), seq_length=SEQ, max_samples=3)
    assert len(small) == 3
    assert small.feature_normalizer.to_dict() == full.feature_normalizer.to_dict()


def test_mimic_lead_selection_by_name(mimic_root):
    ds = MIMICIVECG(str(mimic_root), seq_length=SEQ, leads="standard12", scale="none")
    x = ds[0]["signal"]
    assert ds.lead_names == list(resolve_leads("standard12", MIMIC_LEAD_ORDER))
    # File order stores aVF (5.0 mV) before aVL (6.0 mV); standard order swaps them
    means = x.mean(dim=1).round().tolist()
    assert (
        means[ds.lead_names.index("aVL")] == 6.0
        and means[ds.lead_names.index("aVF")] == 5.0
    )

    ds8 = MIMICIVECG(
        str(mimic_root), seq_length=SEQ, leads="independent8", scale="none"
    )
    assert ds8.lead_names == list(INDEPENDENT_8)
    assert ds8[0]["signal"].mean(dim=1).round().tolist() == [1, 2, 7, 8, 9, 10, 11, 12]


def test_mimic_default_matches_legacy_dataset(mimic_root):
    from ecgen.data.mimic_dataset import MIMICIVECGDataset

    legacy = MIMICIVECGDataset(str(mimic_root), split="train", seq_length=1000)
    new = MIMICIVECG(str(mimic_root), split="train", seq_length=1000)
    assert len(legacy) == len(new)
    for i in range(len(new)):
        ecg, _ = legacy[i]
        sample = new[i]
        assert torch.equal(ecg, sample["signal"])  # zero-padded to 1000, same z-score
        assert sample["meta"]["study_id"] == int(
            legacy.measurements.iloc[i]["study_id"]
        )


def test_mimic_missing_drop_vs_mask(mimic_root):
    n_total = 2 * N_SUBJECTS
    count = lambda **kw: sum(
        len(MIMICIVECG(str(mimic_root), split=s, seq_length=SEQ, **kw))
        for s in ("train", "val", "test")
    )  # noqa: E731
    assert count(missing="drop") == n_total - 1
    assert count(missing="mask") == n_total

    masked = [
        s
        for split in ("train", "val", "test")
        for ds in [
            MIMICIVECG(str(mimic_root), split=split, seq_length=SEQ, missing="mask")
        ]
        for s in (ds[i] for i in range(len(ds)))
        if s["cond_missing"]["measurements"].any()
    ]
    assert len(masked) == 1
    idx = list(MEASUREMENT_COLUMNS).index("qrs_axis")
    assert masked[0]["cond_missing"]["measurements"].tolist() == [
        i == idx for i in range(len(MEASUREMENT_COLUMNS))
    ]
    assert masked[0]["cond"]["measurements"][idx] == 0.0


def test_mimic_without_measurements(mimic_root):
    sample = MIMICIVECG(str(mimic_root), seq_length=SEQ, measurements=[])[0]
    assert sample["cond"] == {} and sample["cond_missing"] == {}


# --------------------------------------------------------- other components


def test_feature_normalizer_roundtrip():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [10.0, 10.0, 10.0, 10.0]})
    norm = FeatureNormalizer.fit(df, ["a", "b"])
    assert norm.mean["a"] == pytest.approx(2.0)
    values, missing = norm.transform([np.nan, 10.0])
    assert missing.tolist() == [True, False] and values.tolist() == [0.0, 0.0]
    assert FeatureNormalizer.from_dict(norm.to_dict()) == norm
    np.testing.assert_allclose(
        norm.inverse_transform(norm.transform([3.0, 10.0])[0]), [3.0, 10.0], rtol=1e-5
    )


def test_scale_signal():
    x = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(scale_signal(x, "none"), x)
    z = scale_signal(x, "zscore")
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
    np.testing.assert_allclose(scale_signal(x, 2), (x - 4.0) / 2)
    with pytest.raises(ValueError):
        scale_signal(x, "bogus")


def test_synthetic_is_deterministic_and_split_specific():
    a, b = (
        SyntheticECG(num_samples=4, seq_length=256),
        SyntheticECG(num_samples=4, seq_length=256),
    )
    assert torch.equal(a[2]["signal"], b[2]["signal"])
    assert not torch.equal(
        a[2]["signal"], SyntheticECG("val", num_samples=4, seq_length=256)[2]["signal"]
    )
    assert a[0]["signal"].shape == (12, 256)
    assert SyntheticECG(leads="independent8", seq_length=256)[0]["signal"].shape == (
        8,
        256,
    )


def test_collate_and_device_transfer():
    ds = SyntheticECG(num_samples=3, seq_length=128, leads="independent8")
    batch = ECGCollate(ds.lead_names, ds.fs)([ds[i] for i in range(3)])
    assert isinstance(batch, ECGBatch) and len(batch) == 3
    assert batch.signal.shape == (3, 8, 128)
    assert batch.cond["measurements"].shape == (3, 2)
    assert batch.cond_missing["measurements"].dtype == torch.bool
    assert batch.meta["index"] == [0, 1, 2] and batch.lead_names == list(INDEPENDENT_8)

    moved = move_data_to_device(batch, torch.device("cpu"))
    assert isinstance(moved, ECGBatch) and moved.lead_names == batch.lead_names


def test_get_signal_formats():
    x = torch.randn(2, 12, 10)
    batch = ECGBatch(signal=x, lead_names=[], fs=500)
    for b in (batch, {"signal": x}, {"ecg_signals": x}, (x, None), x):
        assert get_signal(b) is x
    with pytest.raises(KeyError):
        get_signal({"other": x})


def test_datamodule_shares_train_normalizer_and_saves_state(mimic_root):
    dm = ECGDataModule(
        dataset={
            "name": "mimic_iv_ecg",
            "params": {"root": str(mimic_root), "seq_length": SEQ},
        },
        batch_size=4,
        num_workers=0,
        split_params={"val": {"max_samples": 2}},
    )
    dm.setup("fit")
    assert (
        dm.datasets["val"].feature_normalizer is dm.datasets["train"].feature_normalizer
    )
    assert len(dm.datasets["val"]) == 2

    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, ECGBatch) and batch.signal.shape == (4, 12, SEQ)

    state = dm.state_dict()
    assert state["lead_names"] == list(MIMIC_LEAD_ORDER) and state["fs"] == 500
    assert (
        state["feature_normalizer"] == dm.datasets["train"].feature_normalizer.to_dict()
    )
