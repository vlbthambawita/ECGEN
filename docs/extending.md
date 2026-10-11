# Extending ECGEN

Each extension point is a class plus one registration line. Code outside the
package can skip registration and be referenced with `target: my_pkg.MyClass`.

## A new family

```python
from dataclasses import dataclass
from typing import Any, Optional

from ecgen.families.base import Capabilities, GenerativeModule
from ecgen.registry import FAMILIES
from ecgen.utils.config import resolve_config


@dataclass
class MyConfig:
    in_channels: int = 12
    lr: float = 1e-4
    conditioning: Optional[dict] = None


@FAMILIES.register("my_model")
class MyModel(GenerativeModule):
    supports = Capabilities(conditional=True)
    accepts = frozenset({"vector"})

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__()
        config = resolve_config(MyConfig, config, kwargs)   # checkpoint-safe
        self.save_hyperparameters(config.__dict__)
        self.config = config
        self.init_conditioning(config.conditioning)
        ...

    def training_step(self, batch, batch_idx):
        x = self.get_signal(batch)          # [B, C, T]
        c = self.condition(batch)           # ConditionOutput or None
        ...

    def sample(self, n_samples=16, cond=None, **kwargs):
        c = self.encode_condition(n_samples, cond)   # raw values -> ConditionOutput
        ...
```

Then a config with `model: {name: my_model, params: {...}}`, a smoke test with
the `synthetic` dataset, and a docs page. Data, conditioning, the runner,
callbacks, `ecgen sample` and the evaluation tools work unchanged.

## A new diffusion / flow method

Subclass `ecgen.processes.Process` — `training_pair(x0, noise)` returns
`TrainingPair(xt, t, target, weight)`, `sample(model_fn, shape, steps, device,
generator, inpaint)` integrates — and register it with
`@PROCESSES.register("name")`. Use it with
`model: {name: diffusion, params: {process: {name: name}}}`. No family code.

## A new backbone

Implement `forward(x, t, cond)` (`cond` is a `ConditionOutput` or `None`), set
`accepts` to the slots it consumes, take `in_channels`, `cond_dim`,
`context_dim`, `dense_channels` in `__init__`, and register it with
`@BACKBONES.register("name")`. Zero-initialise the output layer so training
starts from a neutral prediction.

## A new kind of condition

Dataset side: emit `cond["field"]` (and `cond_missing["field"]` if values can be
missing) in every sample; numeric fields should expose a `FeatureNormalizer` in
`field_normalizers` so raw values work at sampling time. Model side: subclass
`ecgen.conditioning.FieldEncoder`, set `slot` (`vector`, `tokens`, or `dense`)
and `out_dim`, implement `forward(value, missing)`, and register it with
`@CONDITION_ENCODERS.register("name")`.

## A new dataset

Subclass `ecgen.data.datasets.BaseECGDataset`: return the dict contract from
`__getitem__`, set `lead_names`, `fs`, `seq_length`, accept `split=`, fit
normalizers on the training split, and register it with
`@DATASETS.register("name")`. See [Adding a dataset](datasets/custom.md).

## A new mask, guidance, or callback

Masks subclass `ecgen.data.masks.MaskGenerator` (`@MASKS.register`); guidance
is a function of model outputs like `ecgen.processes.cfg_model_fn`; callbacks
are Lightning callbacks registered in `CALLBACKS` (a `save_dir` argument defaults
to the run's `samples/` folder).
