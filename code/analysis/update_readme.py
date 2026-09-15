#!/usr/bin/env python3
"""Update marked README result tables from the versioned scientific records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "tensor_quantization": "Tensor-wise baseline",
    "channel_quantization": "Channel-wise quantization",
    "reduced_feedforward": "Reduced feed-forward width (32)",
    "attention_probability_8bit": "8-bit attention probabilities",
    "gradual_budget": "Gradual budget schedule",
    "fixed_width_recovery": "Fixed-width recovery",
    "knowledge_distillation": "Knowledge distillation",
}


def tables():
    metrics = json.loads((ROOT / "results/post_conference/ablation_metrics.json").read_text())
    recent = [f"Evaluation snapshot: **{metrics['evaluation_date']}**. All experiments use eight constituents and one training seed.", "",
              "| Experiment | EBOPs | Validation AUC | Held-out accuracy | Status |",
              "|---|---:|---:|---:|---|"]
    for row in metrics["runs"]:
        label = LABELS[row["arm"]]
        if "ebops" not in row:
            recent.append(f"| {label} | — | — | — | No feasible checkpoint |")
            continue
        recent.append(f"| {label} | {row['ebops']:,} | {row['selected_val_macro_auc']:.4f} | "
                      f"{row['test_accuracy']:.2%} | {'Completed' if row['status'] == 'finished' else 'Interim'} |")
    with (ROOT / "results/pre_conference/auc_per_seed.csv").open() as handle:
        records = list(csv.DictReader(handle))
    precisions = ("FP32", "W8A8", "W1A8", "W1A6", "W1A4")
    baseline = ["| Constituents | FP32 | W8A8 | W1A8 | W1A6 | W1A4 |",
                "|---:|---:|---:|---:|---:|---:|"]
    for n in (8, 16, 32, 64):
        cells = []
        for precision in precisions:
            values = [float(r["macro_ovr_auc"]) for r in records
                      if int(r["n_constituents"]) == n and r["precision"] == precision]
            cells.append(f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}")
        baseline.append(f"| {n} | " + " | ".join(cells) + " |")
    return {"RECENT_RESULTS": "\n".join(recent), "PRE_CONFERENCE": "\n".join(baseline)}


def main():
    path = ROOT / "README.md"
    text = path.read_text()
    for name, table in tables().items():
        begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f"Expected one pair of README markers: {name}")
        left, remaining = text.split(begin)
        _, right = remaining.split(end)
        text = left + begin + "\n\n" + table + "\n\n" + end + right
    path.write_text(text)
    print("Updated README result tables")


if __name__ == "__main__":
    main()
