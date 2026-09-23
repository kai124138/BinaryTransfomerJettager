#!/usr/bin/env python3
"""Validate the frozen N8/N64 source bundle and published result counts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code/constituent-study-20260922"
RESULTS = ROOT / "results/constituent_study"


def read(path: Path):
    return json.loads(path.read_text())


def main():
    manifest = read(RESULTS / "source_manifest.json")
    assert len(manifest["files"]) == 152
    for name, expected in manifest["files"].items():
        path = CODE / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

    preflight = read(RESULTS / "preflight.json")
    assert preflight["status"] == "PASS"
    assert preflight["bundle_sha256"] == manifest["sha256"]
    assert (preflight["cases"], preflight["training_arms"], preflight["static_infeasible_cases"]) == (38, 37, 1)

    result = read(RESULTS / "results-20260923.json")
    summary = result["summary"]
    assert summary == {
        "intended_cases": 38,
        "gpu_training_runs": 37,
        "complete_50_epochs": 28,
        "partial_crashed": 7,
        "canary_only": 2,
        "static_infeasible": 1,
        "feasible_checkpoints": 0,
    }
    assert Counter(row["execution_status"] for row in result["runs"]) == {
        "complete_50_epochs": 28,
        "partial_crashed": 7,
        "canary_only": 2,
    }
    assert len(result["static_infeasible_cases"]) == 1
    assert all(row["selected_feasible_checkpoint"] is None for row in result["runs"])
    print("Validated 152 source files, preflight evidence and 38 constituent-study outcomes")


if __name__ == "__main__":
    main()
