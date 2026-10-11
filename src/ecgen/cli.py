"""
Command-line interface:

- ``ecgen train <config.yaml> [key=value ...]``
- ``ecgen sample <checkpoint> -n 16 -o out --format csv [--cond field=value ...]``
- ``ecgen evaluate <samples.npy> <config.yaml>``
- ``ecgen list``
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

    smp = sub.add_parser("sample", help="Generate ECGs from a checkpoint")
    smp.add_argument("checkpoint", help=".ckpt path or hf://<repo>/<path>")
    smp.add_argument("-n", type=int, default=16, help="number of ECGs")
    smp.add_argument("-o", "--output-dir", default="samples")
    smp.add_argument("--format", default="npy", choices=["npy", "csv", "wfdb", "none"])
    smp.add_argument(
        "--cond",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="condition, repeatable: numbers comma-separated (nan = unknown), "
        "otherwise text, e.g. --cond measurements=800,nan,... --cond report='Sinus rhythm'",
    )
    smp.add_argument("--guidance", type=float)
    smp.add_argument("--steps", type=int)
    smp.add_argument("--seed", type=int)
    smp.add_argument(
        "--family", help="registry name, for checkpoints that do not record it"
    )
    smp.add_argument("--device")
    smp.add_argument("--batch-size", type=int, default=32)
    smp.add_argument(
        "--derive-12", action="store_true", help="complete 8 independent leads to 12"
    )
    smp.add_argument(
        "--plot", type=int, default=0, help="render the first N samples as HTML"
    )

    ev = sub.add_parser(
        "evaluate", help="Compare generated ECGs with real ones (distribution metrics)"
    )
    ev.add_argument("samples", help="generated samples (.npy, [N, C, T])")
    ev.add_argument(
        "config", help="experiment YAML whose data.dataset gives the real ECGs"
    )
    ev.add_argument("--split", default="test")
    ev.add_argument("-n", type=int, default=256, help="real ECGs to load")
    ev.add_argument("--fs", type=int, help="sampling rate (default: the dataset's)")
    return parser


def parse_cond(items: Sequence[str]) -> dict:
    """``["a=1,2,nan", "report=Sinus rhythm"]`` -> ``{"a": [1.0, 2.0, nan], "report": "Sinus rhythm"}``."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--cond must look like FIELD=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        raw = raw.strip().strip("'\"")
        parts = [x.strip() for x in raw.split(",")]
        try:
            values = [float(x) for x in parts]
            out[key] = values if len(values) > 1 else values[0]
        except ValueError:
            out[key] = raw
    return out


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

    if args.command == "sample":
        from ecgen.api import sample

        x = sample(
            args.checkpoint,
            args.n,
            family=args.family,
            cond=parse_cond(args.cond) or None,
            guidance=args.guidance,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            derive_12=args.derive_12,
            output_dir=args.output_dir,
            format=None if args.format == "none" else args.format,
            plot=args.plot,
        )
        print(f"Generated {x.shape[0]} ECGs {tuple(x.shape[1:])} -> {args.output_dir}")
        return 0

    if args.command == "evaluate":
        import json

        import numpy as np

        from ecgen.data.collate import ECGCollate
        from ecgen.evaluation import distribution_metrics, inter_lead_correlation_error
        from ecgen.registry import DATASETS, load_builtins
        from ecgen.utils.io import read_yaml

        load_builtins()
        fake = np.load(args.samples)
        spec = read_yaml(args.config)["data"]["dataset"]
        ds = DATASETS.build(
            {**spec, "params": {**spec.get("params", {}), "split": args.split}}
        )
        n = min(args.n, len(ds))
        real = ECGCollate(ds.lead_names, ds.fs)(
            [ds[i] for i in range(n)]
        ).signal.numpy()
        if real.shape[1:] != fake.shape[1:]:
            raise SystemExit(
                f"Shape mismatch: real {real.shape[1:]} vs generated {fake.shape[1:]}"
            )
        result = distribution_metrics(real, fake, fs=args.fs or ds.fs)
        result["inter_lead_correlation_error"] = inter_lead_correlation_error(
            real, fake
        )
        result.update({"n_real": int(n), "n_generated": int(len(fake))})
        print(json.dumps(result, indent=2))
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
