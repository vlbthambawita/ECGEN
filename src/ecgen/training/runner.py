"""
YAML-driven training runner shared by all families.

Config sections::

    experiment: {name, seed, runs_root}
    model:      {name | target, params}          # registry name, e.g. "vae"
    data:       {dataset: {name | target, params}, batch_size, num_workers,
                 pin_memory, split_params}
    checkpoint: {monitor, mode, save_top_k, save_last, filename}
    callbacks:  [{name | target, params}, ...]
    trainer:    {...}                            # pl.Trainer kwargs
    wandb:      {enabled, project, entity, run_name, tags, notes, log_model}
    resume:     path/to/checkpoint.ckpt | null

Outputs go to ``<runs_root>/<name>/seed_<seed>/`` (``checkpoints/``, ``samples/``,
``metrics/``, ``tb/``, ``config_resolved.yaml``, ``metadata.json``). String values
may use ``${run_dir}``, ``${name}`` and ``${seed}``; a callback that takes a
``save_dir`` and gets none writes to ``${run_dir}/samples``.
"""

from __future__ import annotations

import copy
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ecgen.data.datamodule import ECGDataModule
from ecgen.registry import CALLBACKS, FAMILIES, load_builtins
from ecgen.utils.io import read_yaml, write_json, write_yaml
from ecgen.utils.logging import get_logger
from ecgen.utils.metadata import collect_run_metadata
from ecgen.utils.seed import set_global_seed

logger = get_logger("ecgen.train")

KNOWN_SECTIONS = {
    "experiment",
    "model",
    "data",
    "checkpoint",
    "callbacks",
    "trainer",
    "wandb",
    "resume",
}
DEFAULT_CHECKPOINT = {
    "monitor": "val_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "filename": "epoch{epoch:03d}-step{step:06d}",
}


# --------------------------------------------------------------------- config

# PyYAML (YAML 1.1) reads "1e-3" as a string; treat it as a float
_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def parse_value(raw: str) -> Any:
    """Parse an override value as YAML, reading exponent notation as a float."""
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
        return float(value)
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """
    Apply ``dotted.key=value`` overrides; values are parsed as YAML.

    ``model.params.lr=1e-3``, ``trainer.devices=[1]``, ``callbacks.0.params.num_samples=2``.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = parse_value(raw)
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if isinstance(node, list):
                node = node[int(part)]
            else:
                node = node.setdefault(part, {})
                if node is None:
                    raise ValueError(f"Cannot set {key!r}: {part!r} is null")
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return cfg


def _interpolate(obj: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        for k, v in variables.items():
            obj = obj.replace("${" + k + "}", v)
        return obj
    if isinstance(obj, dict):
        return {k: _interpolate(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, variables) for v in obj]
    return obj


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints: Path
    samples: Path
    metrics: Path


def resolve_config(cfg: Mapping[str, Any]) -> tuple[Dict[str, Any], RunPaths]:
    """Validate sections, fill defaults, and interpolate ``${...}`` variables."""
    unknown = set(cfg) - KNOWN_SECTIONS
    if unknown:
        raise ValueError(
            f"Unknown config sections: {sorted(unknown)}; expected {sorted(KNOWN_SECTIONS)}"
        )
    for section in ("model", "data"):
        if section not in cfg:
            raise ValueError(f"Config needs a '{section}' section")

    cfg = copy.deepcopy(dict(cfg))
    exp = cfg.setdefault("experiment", {})
    exp.setdefault("name", "ecg_experiment")
    exp["seed"] = int(exp.get("seed", 42))
    exp.setdefault("runs_root", "runs")

    run_dir = Path(exp["runs_root"]) / exp["name"] / f"seed_{exp['seed']}"
    cfg = _interpolate(
        cfg, {"run_dir": str(run_dir), "name": exp["name"], "seed": str(exp["seed"])}
    )
    cfg["checkpoint"] = {**DEFAULT_CHECKPOINT, **(cfg.get("checkpoint") or {})}
    cfg.setdefault("callbacks", [])
    cfg["callbacks"] = cfg["callbacks"] or []
    cfg.setdefault("trainer", {})
    cfg.setdefault("wandb", {"enabled": False})
    cfg.setdefault("resume", None)

    paths = RunPaths(
        run_dir, run_dir / "checkpoints", run_dir / "samples", run_dir / "metrics"
    )
    return cfg, paths


# -------------------------------------------------------------------- objects


def build_model(spec: Mapping[str, Any]) -> pl.LightningModule:
    load_builtins()
    return FAMILIES.build(spec)


def build_datamodule(spec: Mapping[str, Any]) -> ECGDataModule:
    load_builtins()
    spec = dict(spec)
    if "dataset" not in spec:
        raise ValueError("data section needs a 'dataset: {name|target, params}' entry")
    return ECGDataModule(**spec)


def build_callbacks(
    specs: Sequence[Mapping[str, Any]], samples_dir: Path
) -> List[Callback]:
    load_builtins()
    callbacks = []
    for spec in specs:
        cls = CALLBACKS.resolve(spec)
        params = dict(spec.get("params") or {})
        if "save_dir" in inspect.signature(cls).parameters and "save_dir" not in params:
            params["save_dir"] = str(samples_dir)
        callbacks.append(cls(**params))
    return callbacks


def _model_io(model: pl.LightningModule) -> Dict[str, Optional[int]]:
    """Channel count and length a model expects, where it declares them."""
    config = getattr(model, "config", None)
    vqvae = getattr(model, "vqvae", None)
    if vqvae is not None and not hasattr(config, "in_channels"):
        config = getattr(
            vqvae, "config", config
        )  # the prior trains on its VQ-VAE's input
    channels = getattr(config, "in_channels", None) or getattr(
        config, "num_channels", None
    )
    seq_length = getattr(model, "fixed_seq_length", getattr(config, "seq_length", None))
    return {"channels": channels, "seq_length": seq_length}


def check_compatibility(model: pl.LightningModule, datamodule: ECGDataModule) -> None:
    """Fail early if the model and dataset disagree on leads, length, or conditions."""
    ds = datamodule.reference_dataset
    conditioner = getattr(model, "conditioner", None)
    if conditioner is not None:
        sample = ds[0]
        available = set(sample.get("cond", {}))
        missing = [f for f in conditioner.cond_fields if f not in available]
        if missing:
            raise ValueError(
                f"Conditioning uses fields {missing} that the dataset does not provide "
                f"(it provides {sorted(available)})"
            )
        if conditioner.uses_obs and datamodule.mask_spec is None:
            raise ValueError(
                "Conditioning reads the observed ECG ('obs' field) but data.masks is not set"
            )
    io = _model_io(model)
    if io["channels"] is not None and io["channels"] != ds.num_leads:
        raise ValueError(
            f"Model expects {io['channels']} channels but the dataset yields {ds.num_leads} "
            f"leads {ds.lead_names}; set data.dataset.params.leads or the model's channel count"
        )
    if io["seq_length"] is not None and io["seq_length"] != ds.seq_length:
        raise ValueError(
            f"Model expects seq_length={io['seq_length']} but the dataset yields {ds.seq_length}"
        )


def _loggers(cfg: Mapping[str, Any], run_dir: Path) -> list:
    loggers: list = [TensorBoardLogger(save_dir=str(run_dir), name="tb")]
    wandb_cfg = cfg.get("wandb") or {}
    if not wandb_cfg.get("enabled", False):
        return loggers
    try:
        from pytorch_lightning.loggers import WandbLogger

        exp = cfg["experiment"]
        loggers.append(
            WandbLogger(
                project=wandb_cfg.get("project", "ecg-generation"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("run_name") or f"{exp['name']}_seed{exp['seed']}",
                save_dir=str(run_dir),
                tags=wandb_cfg.get("tags", []),
                notes=wandb_cfg.get("notes", ""),
                log_model=wandb_cfg.get("log_model", False),
                config=dict(cfg),
            )
        )
    except ImportError:
        logger.warning("wandb not installed; continuing without it (pip install wandb)")
    return loggers


# ------------------------------------------------------------------------ run


def train(
    cfg: Mapping[str, Any],
    overrides: Sequence[str] = (),
    config_path: Optional[str | Path] = None,
) -> pl.Trainer:
    """
    Train a model from a config dict; returns the fitted ``pl.Trainer``.

    Args:
        cfg: Parsed experiment config.
        overrides: ``dotted.key=value`` strings applied on top of ``cfg``.
        config_path: Source YAML, recorded in the run metadata.
    """
    cfg, paths = resolve_config(apply_overrides(dict(cfg), overrides))
    seed = cfg["experiment"]["seed"]
    set_global_seed(seed)

    model = build_model(cfg["model"])
    datamodule = build_datamodule(cfg["data"])
    datamodule.setup("fit")
    check_compatibility(model, datamodule)
    if hasattr(model, "set_data_info"):
        model.set_data_info(datamodule.state_dict())

    for d in (paths.checkpoints, paths.samples, paths.metrics):
        d.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, paths.run_dir / "config_resolved.yaml")
    write_json(
        collect_run_metadata(
            config_path=Path(config_path) if config_path else None,
            cfg=cfg,
            run_dir=paths.run_dir,
            argv=None,
        ),
        paths.run_dir / "metadata.json",
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(paths.checkpoints),
        auto_insert_metric_name=False,
        **cfg["checkpoint"],
    )
    callbacks = [checkpoint] + build_callbacks(cfg["callbacks"], paths.samples)

    trainer = pl.Trainer(
        default_root_dir=str(paths.run_dir),
        logger=_loggers(cfg, paths.run_dir),
        callbacks=callbacks,
        **cfg["trainer"],
    )
    logger.info(
        f"Training '{cfg['experiment']['name']}' ({type(model).__name__}) in {paths.run_dir}"
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg["resume"])

    write_json(
        {
            "callback_metrics": {
                k: float(v) for k, v in trainer.callback_metrics.items()
            },
            "logged_metrics": {k: float(v) for k, v in trainer.logged_metrics.items()},
            "best_model_path": checkpoint.best_model_path,
        },
        paths.metrics / "train_val_metrics.json",
    )
    logger.info(f"Done. Best checkpoint: {checkpoint.best_model_path or '<none>'}")
    return trainer


def train_from_file(path: str | Path, overrides: Sequence[str] = ()) -> pl.Trainer:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return train(read_yaml(path), overrides=overrides, config_path=path)
