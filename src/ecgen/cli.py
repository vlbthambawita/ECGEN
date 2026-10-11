"""
Command-line interface: ``ecgen train <config.yaml> [key=value ...]``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecgen", description="ECG generation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model from an experiment YAML")
    train.add_argument(
        "config", help="Experiment config YAML (configs/experiments/*.yaml)"
    )
    train.add_argument(
        "overrides",
        nargs="*",
        help="dotted.key=value overrides applied after the YAML, e.g. trainer.max_epochs=5",
    )
    train.add_argument(
        "--resume", help="Checkpoint to resume from (same as resume=<path>)"
    )
    train.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit without training",
    )

    sub.add_parser("list", help="List registered families, datasets, and callbacks")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    # parse_known_args so overrides may also follow flags: `train c.yaml --resume x a.b=1`
    args, extra = parser.parse_known_args(argv)
    bad = [
        x for x in extra if args.command != "train" or "=" not in x or x.startswith("-")
    ]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")
    if extra:
        args.overrides = list(args.overrides) + extra

    if args.command == "list":
        from ecgen.registry import CALLBACKS, DATASETS, FAMILIES, load_builtins

        load_builtins()
        for reg in (FAMILIES, DATASETS, CALLBACKS):
            print(f"{reg.kind}: {', '.join(reg)}")
        return 0

    if args.command == "train":
        from ecgen.training import runner
        from ecgen.utils.io import read_yaml

        overrides = list(args.overrides)
        if args.resume:
            overrides.append(f"resume={args.resume}")
        if args.print_config:
            cfg, _ = runner.resolve_config(
                runner.apply_overrides(read_yaml(args.config), overrides)
            )
            print(yaml.safe_dump(cfg, sort_keys=False))
            return 0
        runner.train_from_file(args.config, overrides)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
