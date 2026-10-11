# Adding a Dataset

This guide covers adding a new ECG dataset (PTB-XL, a private cohort, pre-extracted NumPy arrays, ...) and training an existing model on it with `ecgen train`. There are three steps:

1. Subclass `BaseECGDataset` and return the dict sample contract.
2. Make it reachable from YAML: register a name with `@DATASETS.register("name")`, or refer to it by import path with `target:`.
3. Point the `data:` section of an experiment YAML at it.

No DataModule or training script is needed: the generic `ecgen.data.datamodule.ECGDataModule` builds the train/val/test splits of any dataset and the runner does the rest. See [MIMIC-IV-ECG](mimic.md) for the reference implementation (`ecgen.data.datasets.MIMICIVECG`), `ecgen.data.datasets.SyntheticECG` for a minimal one, and the [Data API reference](../api/data.md) for signatures.

## 1. The `BaseECGDataset` contract

`ecgen.data.datasets.base.BaseECGDataset` (`torch.utils.data.Dataset` + `abc.ABC`):

```python
class BaseECGDataset(Dataset, ABC):
    lead_names: List[str]          # names of the C output channels, in order
    fs: int                        # sampling rate in Hz
    seq_length: int                # samples per record (T)
    feature_normalizer: Optional[FeatureNormalizer] = None

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...

    @property
    def num_leads(self) -> int:    # len(self.lead_names)
        ...
```

`__getitem__` returns a dict:

| Key | Type | Notes |
|-----|------|-------|
| `"signal"` | `FloatTensor [C, T]` | `C == len(lead_names)`, `T == seq_length` |
| `"cond"` | `{field: Tensor}` | Condition fields, e.g. `"measurements": [F]`; `{}` for none |
| `"cond_missing"` | `{field: BoolTensor}` | Same keys as `cond`, `True` = missing; `{}` for none |
| `"meta"` | `dict` | Record ids etc.; collected into lists, never fed to a model |

`ECGCollate` stacks the samples into an `ecgen.types.ECGBatch` (`signal [B, C, T]`, `lead_names`, `fs`, `cond`, `cond_missing`, `meta`), taking `lead_names` and `fs` from the dataset. Every family reads the signal with `get_signal(batch)`, so a dataset that follows this contract works with all of them.

### Conventions

These are not enforced, but `ECGDataModule` and the runner rely on them:

- The constructor takes `split` (`"train"`, `"val"`, `"test"`); `ECGDataModule` fills it in for each split, and all other `data.dataset.params` are passed as keyword arguments.
- A deterministic, seeded split computed inside the dataset. MIMIC splits by **subject**, so if your data has patient IDs, split on them to avoid leakage.
- `lead_names`, `fs` and `seq_length` set in `__init__`. The runner compares `len(lead_names)` and `seq_length` with the model before training, and `ECGDataModule` stores them in every checkpoint. Use names from `ecgen.data.leads` (`STANDARD_12`, `INDEPENDENT_8`, `resolve_leads`) where they apply.
- Output `float32`. The VAE/VQ-VAE encoders downsample by `2 ** (len(channel_multipliers) - 1)`, so `seq_length` must divide evenly (5000 / 8 = 625).
- Amplitude scaling via `ecgen.data.datasets.base.scale_signal(ecg, scale)` (`"zscore"` per record, `"none"`, or a number), the same as MIMIC.
- If you provide numeric conditions, fit a `FeatureNormalizer` on the train split and accept a `feature_normalizer` argument: `ECGDataModule` passes the train dataset's normalizer to val/test when the constructor takes one.

## 2. Implement the dataset

The example reads one `(leads, time)` `.npy` file per record:

```python title="my_project/data/npy_ecg.py"
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ecgen.data.datasets.base import BaseECGDataset, Scale, scale_signal
from ecgen.data.leads import STANDARD_12, LeadSpec, resolve_leads
from ecgen.registry import DATASETS


@DATASETS.register("npy_ecg")
class NpyECGDataset(BaseECGDataset):
    """One ``.npy`` file of shape (12, time) per record, leads in standard order."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        leads: LeadSpec = None,
        seq_length: int = 5000,
        fs: int = 500,
        scale: Scale = "zscore",
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.lead_names = resolve_leads(leads, STANDARD_12)
        self._idx = [STANDARD_12.index(n) for n in self.lead_names]
        self.seq_length = seq_length
        self.fs = fs
        self.scale = scale

        files = sorted(Path(root).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {root}")

        # Deterministic split; the same seed gives disjoint train/val/test
        perm = np.random.default_rng(seed).permutation(len(files))
        n_test = int(round(len(files) * test_split))
        n_val = int(round(len(files) * val_split))
        idx = {
            "test": perm[:n_test],
            "val": perm[n_test:n_test + n_val],
            "train": perm[n_test + n_val:],
        }[split]
        self.files = [files[i] for i in sorted(idx)]
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ecg = np.load(self.files[idx]).astype(np.float32)[self._idx]   # (C, time)
        ecg = ecg[:, : self.seq_length]
        if ecg.shape[1] < self.seq_length:
            ecg = np.pad(ecg, ((0, 0), (0, self.seq_length - ecg.shape[1])))
        return {
            "signal": torch.from_numpy(scale_signal(ecg, self.scale)),
            "cond": {},
            "cond_missing": {},
            "meta": {"file": self.files[idx].name},
        }
```

Check it quickly:

```python
import torch
from my_project.data.npy_ecg import NpyECGDataset

ds = NpyECGDataset("data/raw/npy_ecg", split="val")
x = ds[0]["signal"]
assert x.shape == (ds.num_leads, ds.seq_length) and x.dtype == torch.float32
```

!!! tip "Write a test"
    Add a test that writes a few random arrays to `tmp_path`, then checks split disjointness, output shape and dtype, and that `ECGDataModule(dataset={"target": ..., "params": {...}}, batch_size=2)` yields an `ECGBatch`. Run it with `pytest tests/`.

## 3. Make it reachable from YAML

Two options:

=== "`target:` (no registration)"

    Refer to the class by import path. The module only has to be importable (installed, or on
    `PYTHONPATH`):

    ```yaml
    data:
      dataset:
        target: my_project.data.npy_ecg.NpyECGDataset
        params: {root: data/raw/npy_ecg}
    ```

=== "Registered name"

    `@DATASETS.register("npy_ecg")` adds the class to `ecgen.registry.DATASETS` when its module is
    imported. Built-in datasets live in `src/ecgen/data/datasets/` and are imported by
    `ecgen/data/datasets/__init__.py`, which `ecgen.registry.load_builtins()` imports; add yours
    there to make `name: npy_ecg` work everywhere (`ecgen list` shows it):

    ```yaml
    data:
      dataset:
        name: npy_ecg
        params: {root: data/raw/npy_ecg}
    ```

## 4. Experiment YAML

Copy an existing experiment (for example `configs/experiments/vqvae_mimic.yaml`) to `configs/experiments/vqvae_npy_ecg.yaml` and replace the `data:` section:

```yaml
experiment:
  name: vqvae_npy_ecg
  seed: 42
  runs_root: runs

data:
  dataset:
    name: npy_ecg                        # or target: my_project.data.npy_ecg.NpyECGDataset
    params:
      root: data/raw/npy_ecg
      max_samples: null
      val_split: 0.1
      test_split: 0.1
      leads: null                        # 12 leads
      seq_length: 5000
  split_params:                          # optional per-split overrides
    val: {max_samples: 500}
  batch_size: 32
  num_workers: 4
```

Run it:

```bash
ecgen train configs/experiments/vqvae_npy_ecg.yaml
ecgen train configs/experiments/vqvae_mimic.yaml \
    data.dataset.name=npy_ecg 'data.dataset.params={root: data/raw/npy_ecg}'   # without a new file
```

Outputs go to `runs/<experiment.name>/seed_<seed>/`, as described in [Training with `ecgen train`](../training/runner.md#run-directory).

!!! tip "Model compatibility checklist"
    - The model's `in_channels` (VAE, VQ-VAE) or `num_channels` (Pulse2Pulse) must equal `len(lead_names)`; the runner fails early otherwise.
    - `seq_length` must be divisible by the encoder's total downsampling factor (8 with the default `channel_multipliers: [1, 2, 4, 4]`), and equal Pulse2Pulse's `seq_length` (a multiple of 2500).
    - Pulse2Pulse uses 8 leads (`num_channels: 8`, `leads: independent8`). See [Pulse2Pulse GAN](../models/gan.md).

`configs/dataset/*.yaml` fragments are reference values only; the runner reads the experiment YAML.

## Checklist

- [ ] `BaseECGDataset` subclass implementing `__len__` and `__getitem__`, setting `lead_names`, `fs` and `seq_length`
- [ ] `__getitem__` returns `{"signal", "cond", "cond_missing", "meta"}`
- [ ] Constructor takes `split`; split is seeded and leakage-free
- [ ] Registered with `@DATASETS.register("<name>")` (imported by `ecgen.data.datasets`) or referenced with `target:`
- [ ] `configs/experiments/<model>_<name>.yaml` with a `data.dataset` section
- [ ] A test under `tests/`
