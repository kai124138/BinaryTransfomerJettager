#!/usr/bin/env python3
"""Validate source syntax, configurations, result invariants, and README links."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/hgq2"))
from bnhgq2.config import load_config  # noqa: E402


def main():
    sources = list((ROOT / "code").rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    configs = list((ROOT / "code/hgq2/configs").glob("*.json"))
    for path in configs:
        data = json.loads(path.read_text())
        if "arch" in data:
            config = load_config(path)
            assert config["name"] == path.stem, path
            assert config["arch"]["n_feat"] == len(config["arch"]["features"]), path
    for path in (ROOT / "results").rglob("*.json"):
        json.loads(path.read_text())
    with (ROOT / "results/pre_conference/auc_per_seed.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    keys = {(int(r["n_constituents"]), r["precision"], int(r["seed"])) for r in rows}
    assert len(keys) == 60
    for row in rows:
        aucs = [float(row[f"auc_{c}"]) for c in ("g", "q", "W", "Z", "t")]
        assert all(0 <= value <= 1 for value in aucs)
        assert abs(sum(aucs) / 5 - float(row["macro_ovr_auc"])) < 1e-12
    data = json.loads((ROOT / "results/post_conference/ablation_metrics.json").read_text())
    assert data["n_test"] == 260000 and data["n_validation"] == 124000
    assert len({r["arm"] for r in data["runs"]}) == 7
    for row in data["runs"]:
        if "ebops" not in row:
            continue
        assert row["ebops"] <= data["budget_ebops"]
        assert 0 <= row["selected_val_macro_auc"] <= 1
        assert 0 <= row["test_accuracy"] <= 1
        assert re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_sha256"])
        assert row["selected_epoch_one_based"] == row["selected_epoch_zero_based"] + 1
    final_training = json.loads((ROOT / "docs/current-work/training-results-20260923.json").read_text())
    assert final_training["summary"] == {
        "runs": 27, "training_complete": 27,
        "architecture_runs": 12, "architecture_feasible": 5,
        "attention_runs": 15, "attention_feasible": 0,
    }
    constituent = json.loads((ROOT / "results/constituent_study/results-20260923.json").read_text())
    assert constituent["summary"]["intended_cases"] == 38
    assert constituent["summary"]["complete_50_epochs"] == 28
    assert constituent["summary"]["feasible_checkpoints"] == 0
    engram = json.loads((ROOT / "results/engram/status-20260923.json").read_text())
    assert engram["summary"]["training_loops_complete"] == 4
    assert engram["summary"]["feasible_checkpoints"] == 0
    registry = json.loads((ROOT / "results/wandb-project-inventory-20260923.json").read_text())
    assert registry["summary"] == {
        "projects": 3, "wandb_run_records": 86,
        "scientific_training_records": 81, "diagnostic_attempts": 5,
        "statically_rejected_cases_not_in_wandb": 1,
    }
    assert {row["wandb_run_records"] for row in registry["projects"]} == {13, 27, 46}
    standalone = json.loads((ROOT / "results/post_conference/standalone_long_budget.json").read_text())
    assert standalone["result"]["completed_epochs"] == 1000
    assert standalone["result"]["selected_ebops"] <= standalone["configuration"]["target_ebops"]
    for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / target.split("#")[0]).exists(), f"Broken README link: {target}"
    print(f"Validated {len(sources)} Python sources, {len(configs)} configurations, and result records")


if __name__ == "__main__":
    main()
