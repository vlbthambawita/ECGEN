#!/usr/bin/env python3
"""
Lead-reconstruction error of a model: observe some leads of real test ECGs,
generate the rest, and compare with the truth on the unobserved leads.

    python scripts/evaluate/lead_reconstruction.py <ckpt> --family diffusion \
        --keep I --n 64 [--inpaint repaint] [--steps 50]

Reports MAE, RMSE and the mean per-lead Pearson r on the unobserved leads, for
the model and for the trivial "all zeros" reconstruction (z-scored data).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ecgen.data.collate import ECGCollate
from ecgen.registry import DATASETS, FAMILIES, load_builtins
from ecgen.utils.io import read_yaml


def metrics(pred: np.ndarray, true: np.ndarray, unobserved: np.ndarray) -> dict:
    err = (pred - true)[:, unobserved]
    rs = []
    for i in range(pred.shape[0]):
        for c in np.where(unobserved)[0]:
            p, t = pred[i, c], true[i, c]
            if p.std() > 0 and t.std() > 0:
                rs.append(np.corrcoef(p, t)[0, 1])
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "pearson_r": float(np.mean(rs)) if rs else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoint")
    p.add_argument("--family", required=True)
    p.add_argument(
        "--config",
        help="experiment YAML whose data.dataset is used (default: run dir's config_resolved.yaml)",
    )
    p.add_argument("--keep", default="I", help="comma-separated observed leads")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--inpaint", default=None, help="replace | repaint (unconditional models)"
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    load_builtins()
    model = (
        FAMILIES.get(args.family)
        .load_from_checkpoint(args.checkpoint, map_location=args.device)
        .eval()
    )
    from pathlib import Path

    cfg_path = args.config or str(
        Path(args.checkpoint).parents[1] / "config_resolved.yaml"
    )
    spec = read_yaml(cfg_path)["data"]["dataset"]
    ds = DATASETS.build({**spec, "params": {**spec.get("params", {}), "split": "test"}})
    collate = ECGCollate(ds.lead_names, ds.fs)
    keep = [ds.lead_names.index(name) for name in args.keep.split(",")]
    unobserved = np.ones(len(ds.lead_names), dtype=bool)
    unobserved[keep] = False

    preds, trues = [], []
    for start in range(0, min(args.n, len(ds)), args.batch):
        batch = collate(
            [ds[i] for i in range(start, min(start + args.batch, args.n, len(ds)))]
        )
        x = batch.signal.to(args.device)
        mask = torch.zeros_like(x)
        mask[:, keep] = 1
        kwargs = {"obs": x, "obs_mask": mask}
        if args.family in ("diffusion", "flow_matching"):
            kwargs.update(
                steps=args.steps, seed=args.seed + start, inpaint=args.inpaint
            )
        with torch.no_grad():
            out = model.sample(x.shape[0], seq_length=x.shape[-1], **kwargs)
        preds.append(out.cpu().numpy())
        trues.append(x.cpu().numpy())
    pred, true = np.concatenate(preds), np.concatenate(trues)
    print(
        json.dumps(
            {
                "model": metrics(pred, true, unobserved),
                "zeros": metrics(np.zeros_like(true), true, unobserved),
                "n": int(pred.shape[0]),
                "observed": args.keep,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
