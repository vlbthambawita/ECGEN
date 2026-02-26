import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # TODO: wire up config loading, datamodule, model, and trainer.
    print(f"[train] Would run training with config: {config_path}")


if __name__ == "__main__":
    main()
