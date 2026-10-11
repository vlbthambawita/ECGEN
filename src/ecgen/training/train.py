"""Deprecated: use ``ecgen train <config.yaml>`` (:mod:`ecgen.training.runner`)."""

import argparse

from ecgen.training.runner import train_from_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ECG model with PyTorch Lightning."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML."
    )
    args, overrides = parser.parse_known_args()
    train_from_file(args.config, overrides)


if __name__ == "__main__":
    main()
