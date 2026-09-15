#!/usr/bin/env python3
"""Recompute AUC and compact ROC curves from saved class-score arrays.

The input directory contains n{8,16,32,64}/{precision}-s{1,2,3}.npz.
Each archive must contain y, score, and JSON metadata in meta. Exact AUC
values are calculated from all events. Interpolated ROC curves are for
visualization only and must not be integrated to recover those AUC values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CLASSES = ("g", "q", "W", "Z", "t")
PRECISIONS = ("FP32", "W8A8", "W1A8", "W1A6", "W1A4")


def summarize(predictions: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows, curves, sources = [], [], []
    grid = np.geomspace(1e-5, 1, 300)
    for n in (8, 16, 32, 64):
        for precision in PRECISIONS:
            for seed in (1, 2, 3):
                path = predictions / f"n{n}" / f"{precision}-s{seed}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    y, score = saved["y"], saved["score"]
                    meta = json.loads(str(saved["meta"]))
                if y.shape != (260_000, 5) or score.shape != y.shape:
                    raise ValueError(f"Unexpected prediction shape: {path.name}")
                if not np.isfinite(score).all() or not np.all(y.sum(axis=1) == 1):
                    raise ValueError(f"Invalid labels or scores: {path.name}")
                aucs = [roc_auc_score(y[:, i], score[:, i]) for i in range(5)]
                auc = float(np.mean(aucs))
                if not np.isclose(auc, meta["auc"], rtol=0, atol=1e-12):
                    raise ValueError(f"AUC does not match archived metadata: {path.name}")
                rows.append({"n_constituents": n, "precision": precision, "seed": seed,
                             "macro_ovr_auc": auc,
                             **{f"auc_{c}": float(v) for c, v in zip(CLASSES, aucs)}})
                # Seed one is used consistently for the illustrative class-wise curves.
                if seed == 1:
                    for i, label in enumerate(CLASSES):
                        fpr, tpr, _ = roc_curve(y[:, i], score[:, i])
                        # Last point of each FPR plateau gives maximal attainable TPR.
                        keep = np.r_[np.diff(fpr) != 0, True]
                        values = np.interp(grid, fpr[keep], tpr[keep])
                        curves.extend({"n_constituents": n, "precision": precision,
                                       "seed": seed, "class": label, "fpr": float(x),
                                       "tpr": float(z)} for x, z in zip(grid, values))
                sources.append({"configuration": f"n{n}-{precision}-seed{seed}",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "n_events": len(y)})
        print(f"Verified 15 prediction archives for N={n}", flush=True)
    for filename, values in (("auc_per_seed.csv", rows), ("roc_curves.csv", curves)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=values[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    (output / "prediction_integrity.json").write_text(json.dumps({
        "definition": "Arithmetic mean of five one-vs-rest ROC AUC values.",
        "evaluation_split": "Zenodo validation archive, held out from training",
        "metadata_agreement_atol": 1e-12,
        "roc_sampling": "300 logarithmic FPR points, seed 1; visualization only",
        "sources": sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/pre_conference"))
    args = parser.parse_args()
    summarize(args.predictions, args.output)
