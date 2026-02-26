import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ecgen.utils.io import read_yaml
from ecgen.utils.logging import get_logger
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


def build_objects(cfg: Dict[str, Any]) -> tuple[pl.LightningModule, pl.LightningDataModule, Dict[str, Any]]:
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

    # Trainer
    trainer_cfg = cfg.get("trainer", {})
    return model, datamodule, trainer_cfg


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = read_yaml(config_path)
    logger = get_logger("train")

    model, datamodule, trainer_cfg = build_objects(cfg)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "ecg_experiment")
    output_dir = Path(exp_cfg.get("output_dir", "outputs"))

    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name=exp_name,
    )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=tb_logger,
        **trainer_cfg,
    )

    logger.info(f"Starting training for experiment '{exp_name}' with config: {config_path}")
    trainer.fit(model=model, datamodule=datamodule)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()

