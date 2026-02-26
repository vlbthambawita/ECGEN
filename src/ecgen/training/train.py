import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from ecgen.utils.io import read_yaml, write_yaml, write_json
from ecgen.utils.logging import get_logger
from ecgen.utils.metadata import collect_run_metadata
from ecgen.utils.seed import set_global_seed


def _import_from_string(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path: {path}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG model with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def build_objects(cfg: Dict[str, Any]) -> tuple[pl.LightningModule, pl.LightningDataModule, Dict[str, Any], List[Callback]]:
    # Experiment section
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))
    set_global_seed(seed)

    # Model
    model_cfg = cfg["model"]
    model_target = model_cfg["target"]
    model_params = model_cfg.get("params", {})
    model_cls = _import_from_string(model_target)
    model = model_cls(**model_params)

    # Data
    data_cfg = cfg["data"]
    data_target = data_cfg["target"]
    data_params = data_cfg.get("params", {})
    data_cls = _import_from_string(data_target)
    datamodule = data_cls(**data_params)

    # Callbacks
    callbacks_list: List[Callback] = []
    callbacks_cfg = cfg.get("callbacks", [])
    for cb_cfg in callbacks_cfg:
        cb_target = cb_cfg["target"]
        cb_params = cb_cfg.get("params", {})
        cb_cls = _import_from_string(cb_target)
        callbacks_list.append(cb_cls(**cb_params))

    # Trainer
    trainer_cfg = cfg.get("trainer", {})
    return model, datamodule, trainer_cfg, callbacks_list


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = read_yaml(config_path)
    logger = get_logger("train")

    model, datamodule, trainer_cfg, custom_callbacks = build_objects(cfg)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "ecg_experiment")
    seed = int(exp_cfg.get("seed", 42))

    # Standardized run directory:
    # runs/<experiment_name>/seed_<N>/
    runs_root = Path(exp_cfg.get("runs_root", "runs"))
    run_dir = runs_root / exp_name / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    preds_dir = run_dir / "preds"
    samples_dir = run_dir / "samples"

    for d in (checkpoints_dir, metrics_dir, preds_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save resolved config and metadata for this run
    write_yaml(cfg, run_dir / "config_resolved.yaml")
    metadata = collect_run_metadata(
        config_path=config_path,
        cfg=cfg,
        run_dir=run_dir,
        argv=None,
    )
    write_json(metadata, run_dir / "metadata.json")

    # Setup loggers
    loggers_list = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="tb",
    )
    loggers_list.append(tb_logger)
    
    # Weights & Biases logger (optional)
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            import wandb
            
            wandb_project = wandb_cfg.get("project", "ecg-generation")
            wandb_entity = wandb_cfg.get("entity", None)
            wandb_run_name = wandb_cfg.get("run_name", None) or f"{exp_name}_seed{seed}"
            wandb_tags = wandb_cfg.get("tags", [])
            wandb_notes = wandb_cfg.get("notes", "")
            
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                save_dir=str(run_dir),
                tags=wandb_tags,
                notes=wandb_notes,
                log_model=wandb_cfg.get("log_model", True),
                config=cfg,
            )
            loggers_list.append(wandb_logger)
            logger.info(f"Weights & Biases logging enabled: {wandb_project}/{wandb_run_name}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    else:
        logger.info("Weights & Biases logging disabled")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch{epoch:03d}-step{step:06d}",
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )

    all_callbacks = [checkpoint_callback] + custom_callbacks

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        logger=loggers_list,
        callbacks=all_callbacks,
        **trainer_cfg,
    )

    logger.info(f"Starting training for experiment '{exp_name}' with config: {config_path}")
    trainer.fit(model=model, datamodule=datamodule)
    logger.info("Training finished.")

    # Persist final metrics snapshot into the run directory
    callback_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logged_metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
    metrics_payload = {
        "callback_metrics": callback_metrics,
        "logged_metrics": logged_metrics,
    }
    write_json(metrics_payload, metrics_dir / "train_val_metrics.json")


if __name__ == "__main__":
    main()

