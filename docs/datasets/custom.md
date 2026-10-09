# Adding a Dataset

This guide covers adding a new ECG dataset (PTB-XL, a private cohort, pre-extracted NumPy arrays, ...) and training an existing model on it. There are four steps:

1. Subclass `BaseECGDataset` in `data/datasets/<name>/`.
2. Write a `LightningDataModule` around it.
3. Add a config fragment under `configs/dataset/` and the matching `data:` section in an experiment YAML.
4. Wire it into a training script.

See [MIMIC-IV-ECG](mimic.md) for the reference implementation and the [Data API reference](../api/data.md) for signatures.

## 1. The `BaseECGDataset` contract

`data/datasets/base.py` defines the abstract base class:

```python
class BaseECGDataset(Dataset, ABC):          # torch.utils.data.Dataset + abc.ABC

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Should return a dictionary with at least:
        - 'ecg_signals': torch.Tensor of shape (num_leads, seq_length)"""

    @property
    @abstractmethod
    def num_leads(self) -> int:
        """Return the number of ECG leads."""

    @property
    @abstractmethod
    def seq_length(self) -> int:
        """Return the sequence length."""
```

You must implement all four members, or instantiation fails with `TypeError: Can't instantiate abstract class ...`.

| Member | Kind | Must return |
|--------|------|-------------|
| `__len__()` | method | Number of samples in **this split** |
| `__getitem__(idx)` | method | `dict` with at least `"ecg_signals"`: a `torch.Tensor` of shape `(num_leads, seq_length)` |
| `num_leads` | read-only **property** | Lead count of every returned tensor |
| `seq_length` | read-only **property** | Time steps of every returned tensor |

The base class has no constructor, splitting, normalization or I/O logic. All of that is up to the subclass.

!!! warning "`num_leads` / `seq_length` are properties without setters"
    `self.num_leads = 12` in your `__init__` raises `AttributeError` because the base class declares them as properties. Store the values under private names (`self._num_leads`) and return them from `@property` methods, as in the example below.

!!! info "Dict vs. tuple: what the existing code consumes"
    The base class asks for a **dict**, but the MIMIC dataset (which does not subclass `BaseECGDataset`) returns a **tuple** `(ecg, features)`, and some code assumes one form:

    | Consumer | `{"ecg_signals": ...}` | `(ecg, cond)` |
    |----------|:---:|:---:|
    | `models.vqvae.VQVAELightning`, `PriorLightning` | yes | yes |
    | `models.vae.VAELightning` | yes | no |
    | Pulse2Pulse GAN (`models/gan/pulse2pulse.py`) | yes | no |
    | `training.callbacks.visualization.VAEVisualizationCallback` | no (does `ecg, _ = batch`) | yes |
    | `VAEDataWrapper` inside `scripts/train/train_vae_mimic.py` | no | yes |

    Follow the base-class contract (dict) for new code. If you want to reuse a training script that currently runs on MIMIC unchanged, including its visualization callback, also offer a tuple (see `return_tuple` in the example).

### Conventions used elsewhere in the repo

These are not enforced by the base class, but following them makes a dataset a drop-in replacement for MIMIC:

- Constructor arguments `split` (`"train"`, `"val"`, `"test"`), `val_split`, `test_split`, `seed`, `max_samples`, `num_leads`, `seq_length`.
- A deterministic, seeded split computed inside the dataset. MIMIC splits by **subject**, so if your data has patient IDs, split on them to avoid leakage.
- Output `float32`, shape `(num_leads, seq_length)`, default `12 × 5000` (10 s at 500 Hz). The models' default `in_channels` is 12, and the VQ-VAE/VAE encoders downsample by `2 ** (len(channel_multipliers) - 1)`, so `seq_length` must divide evenly (5000 / 8 = 625).
- Per-sample normalization `(x - x.mean()) / max(x.std(), eps)` over all leads, the same as MIMIC.
- Raw data in `data/raw/<name>/`, processed data in `data/processed/<name>/` (both gitignored).

## 2. Implement the dataset

Create `data/datasets/npy_ecg/` with an `__init__.py` and a `dataset.py`. The example below reads one `(leads, time)` `.npy` file per record. It runs as-is against the real base class:

```python title="data/datasets/npy_ecg/dataset.py"
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from data.datasets.base import BaseECGDataset


class NpyECGDataset(BaseECGDataset):
    """One ``.npy`` file of shape (leads, time) per record."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        num_leads: int = 12,
        seq_length: int = 5000,
        return_tuple: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(root)
        self.split = split
        self.return_tuple = return_tuple
        self._num_leads = num_leads      # backing fields for the abstract properties
        self._seq_length = seq_length

        files = sorted(self.root.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {self.root}")

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

    # --- abstract API -------------------------------------------------------
    @property
    def num_leads(self) -> int:
        return self._num_leads

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ecg = np.load(self.files[idx]).astype(np.float32)   # (leads, time)
        ecg = self._fix_shape(ecg)
        ecg = (ecg - ecg.mean()) / max(float(ecg.std()), 1e-6)
        ecg = torch.from_numpy(ecg)
        if self.return_tuple:                                # MIMIC-style (ecg, cond)
            return ecg, torch.zeros(0)
        return {"ecg_signals": ecg}

    # --- helpers ------------------------------------------------------------
    def _fix_shape(self, x: np.ndarray) -> np.ndarray:
        """Truncate/zero-pad to (num_leads, seq_length)."""
        x = x[: self.num_leads]
        if x.shape[0] < self.num_leads:
            pad = np.zeros((self.num_leads - x.shape[0], x.shape[1]), np.float32)
            x = np.vstack([x, pad])
        x = x[:, : self.seq_length]
        if x.shape[1] < self.seq_length:
            x = np.pad(x, ((0, 0), (0, self.seq_length - x.shape[1])))
        return x
```

```python title="data/datasets/npy_ecg/__init__.py"
"""NumPy-file ECG dataset."""

from data.datasets.npy_ecg.dataset import NpyECGDataset

__all__ = ["NpyECGDataset"]
```

Check it quickly:

```python
from data.datasets.npy_ecg import NpyECGDataset

ds = NpyECGDataset("data/raw/npy_ecg", split="val")
x = ds[0]["ecg_signals"]
assert x.shape == (ds.num_leads, ds.seq_length) and x.dtype.is_floating_point
```

!!! tip "Write a test"
    Add `tests/test_data/test_npy_ecg.py` that writes a few random arrays to `tmp_path`, then checks split disjointness, output shape and dtype. Run it with `pytest tests/`.

## 3. Create a LightningDataModule

The repo has no generic datamodule. `data/datasets/mimic/datamodule.py::ECGDataModule` is a plain dataclass, not a `LightningDataModule`. Each training script defines its own. Put yours next to the dataset so scripts can import it:

```python title="data/datasets/npy_ecg/datamodule.py"
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data.datasets.npy_ecg.dataset import NpyECGDataset


class NpyECGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
        seed: int = 42,
        num_leads: int = 12,
        seq_length: int = 5000,
        return_tuple: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_kwargs = dict(
            root=data_dir, val_split=val_split, test_split=test_split,
            max_samples=max_samples, seed=seed, num_leads=num_leads,
            seq_length=seq_length, return_tuple=return_tuple,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = NpyECGDataset(split="train", **self.ds_kwargs)
            self.val_dataset = NpyECGDataset(split="val", **self.ds_kwargs)
        if stage in ("test", None):
            self.test_dataset = NpyECGDataset(split="test", **self.ds_kwargs)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
```

With the default dict output, a batch is `{"ecg_signals": Tensor[B, num_leads, seq_length]}`. With `return_tuple=True` it is `(Tensor[B, num_leads, seq_length], Tensor[B, 0])`.

## 4. Add configuration

### Dataset fragment

Add `configs/dataset/npy_ecg.yaml`, following `configs/dataset/mimic.yaml`:

```yaml title="configs/dataset/npy_ecg.yaml"
dataset:
  name: npy_ecg
  data_dir: data/raw/npy_ecg
  num_leads: 12
  seq_length: 5000
  sampling_rate: 500  # Hz
  batch_size: 32
  num_workers: 4
  val_split: 0.1
  test_split: 0.1
  max_samples: null
```

!!! warning "Fragments are documentation, not inputs"
    No script loads `configs/dataset/*.yaml`. The training scripts read only `configs/experiments/*.yaml` and only the keys they explicitly look up (see [Configuration](../getting_started/configuration.md)). The fragment records your defaults. The experiment YAML is what takes effect.

### Experiment YAML

Copy an existing experiment (for example `configs/experiments/vqvae_mimic.yaml`) to `configs/experiments/vqvae_npy_ecg.yaml` and replace the `data:` section:

```yaml
experiment:
  name: vqvae_npy_ecg
  seed: 42
  runs_root: runs

data:
  target: data.datasets.npy_ecg.NpyECGDataModule   # informational; scripts don't instantiate it
  params:
    data_dir: data/raw/npy_ecg
    batch_size: 32
    num_workers: 4
    max_samples: null
    val_split: 0.1
    test_split: 0.1
    seq_length: 5000
```

## 5. Wire it into a training script

The `target:` fields are not instantiated generically. Each script maps specific YAML keys onto argparse arguments. A new dataset therefore needs a script, or a change to an existing one, that:

1. imports your datamodule,
2. reads your `data.params` keys,
3. builds the datamodule instead of the inline MIMIC one.

Start from `scripts/train/train_template.py`, or copy the closest existing script (for example `scripts/train/train_vqvae_mimic.py`) to `scripts/train/train_vqvae_npy_ecg.py` and change the data parts:

```python title="scripts/train/train_vqvae_npy_ecg.py (excerpt)"
from data.datasets.npy_ecg.datamodule import NpyECGDataModule   # replaces the inline MIMIC datamodule

# ... inside main(), where the YAML 'data' section is parsed:
if "data" in yaml_config and "params" in yaml_config["data"]:
    data_cfg = yaml_config["data"]["params"]
    args.data_dir = data_cfg.get("data_dir", args.data_dir)      # was 'mimic_path'
    args.batch_size = data_cfg.get("batch_size", args.batch_size)
    args.num_workers = data_cfg.get("num_workers", args.num_workers)
    args.max_samples = data_cfg.get("max_samples", args.max_samples)
    args.val_split = data_cfg.get("val_split", args.val_split)
    args.test_split = data_cfg.get("test_split", args.test_split)
    args.seq_length = data_cfg.get("seq_length", args.seq_length)

# ... where the datamodule is built:
datamodule = NpyECGDataModule(
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    val_split=args.val_split,
    test_split=args.test_split,
    max_samples=args.max_samples,
    seed=args.seed,
    num_leads=args.in_channels,       # keep data and model lead counts in sync
    seq_length=args.seq_length,
    return_tuple=True,                # VAEVisualizationCallback unpacks (ecg, _)
)
```

Run it:

```bash
python scripts/train/train_vqvae_npy_ecg.py --config configs/experiments/vqvae_npy_ecg.yaml
```

!!! note "The one generic builder"
    `training/trainers/gan_trainer.py` (`--config <yaml>`) does instantiate `data.target(**data.params)` generically, so pointing `target` at `data.datasets.npy_ecg.datamodule.NpyECGDataModule` would work there without code changes. However, it imports `ecgen.utils.*` from the legacy `src/ecgen` package, and its model (Pulse2Pulse) needs dict batches.

Optionally add a wrapper in `scripts/shell/`, like the existing `run_train_*_mimic.sh` scripts. Outputs go to `runs/<experiment.name>/seed_<seed>/`, as described in the [Training guide](../training/guide.md).

!!! tip "Model compatibility checklist"
    - `model.params.config.in_channels` must equal the dataset's `num_leads`.
    - `seq_length` must be divisible by the encoder's total downsampling factor (8 with the default `channel_multipliers: [1, 2, 4, 4]`).
    - Pulse2Pulse uses 8 leads by default (`num_channels: 8`). See [Pulse2Pulse GAN](../models/gan.md).

## Checklist

- [ ] `data/datasets/<name>/dataset.py`: subclass of `BaseECGDataset` implementing `__len__`, `__getitem__`, and the `num_leads`/`seq_length` properties
- [ ] `data/datasets/<name>/__init__.py` exports the class
- [ ] `data/datasets/<name>/datamodule.py`: `LightningDataModule` with `setup` and `*_dataloader`
- [ ] `configs/dataset/<name>.yaml` (reference defaults)
- [ ] `configs/experiments/<model>_<name>.yaml` with a `data.params` section
- [ ] Training script reads those keys and builds your datamodule
- [ ] Optional: `data/datasets/<name>/README.md` and a test under `tests/`
