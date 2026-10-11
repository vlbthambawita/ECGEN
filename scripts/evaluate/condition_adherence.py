#!/usr/bin/env python3
"""
Heart-rate adherence of a conditional model: sample ECGs at requested heart
rates and measure the heart rate of what comes out.

    python scripts/evaluate/condition_adherence.py <ckpt> --family vae \
        --field measurements --columns heart_rate,amplitude --fixed amplitude=1.0 \
        --rates 50,70,90,110,130 --n 16

For MIMIC (field ``measurements``, RR interval in ms) use ``--target rr_interval``
and the RR interval is set to 60000 / rate. Unconditional checkpoints are
sampled without conditions (a baseline: their r should be ~0).
"""

from __future__ import annotations

import argparse
import json

import torch

from ecgen.evaluation.measure import adherence, heart_rate
from ecgen.registry import FAMILIES, load_builtins


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoint")
    p.add_argument(
        "--family",
        required=True,
        help="registry name, e.g. vae, pulse2pulse, vqvae_prior",
    )
    p.add_argument("--field", default="measurements")
    p.add_argument(
        "--target",
        default="heart_rate",
        help="column that sets the rate (heart_rate or rr_interval)",
    )
    p.add_argument(
        "--fixed",
        default="",
        help="other columns, e.g. amplitude=1.0 (others are NaN = unknown)",
    )
    p.add_argument("--rates", default="50,70,90,110,130")
    p.add_argument("--n", type=int, default=16, help="samples per rate")
    p.add_argument("--seq-length", type=int, default=5000)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--steps", type=int, help="sampling steps (diffusion / flow)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    load_builtins()
    model = (
        FAMILIES.get(args.family)
        .load_from_checkpoint(args.checkpoint, map_location=args.device)
        .eval()
    )
    fs = int(model.data_info.get("fs", 500))
    fixed = dict(kv.split("=") for kv in args.fixed.split(",") if kv)
    torch.manual_seed(args.seed)

    requested, measured = [], []
    for rate in [float(r) for r in args.rates.split(",")]:
        kwargs = {"seq_length": args.seq_length} if args.family != "pulse2pulse" else {}
        if args.guidance != 1.0:
            kwargs["guidance"] = args.guidance
        if args.steps:
            kwargs["steps"] = args.steps
        if model.is_conditional:
            columns = model.data_info["normalizers"][args.field]["columns"]
            value = {c: float(fixed.get(c, "nan")) for c in columns}
            value[args.target] = rate if args.target == "heart_rate" else 60000.0 / rate
            kwargs["cond"] = {args.field: [value[c] for c in columns]}
        with torch.no_grad():
            x = model.sample(args.n, **kwargs)
        requested += [rate] * args.n
        measured += list(heart_rate(x, fs))

    result = adherence(requested, measured)
    result["conditional"] = model.is_conditional
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
