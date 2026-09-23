#!/usr/bin/env python3
"""Capture the public experiment registry from the three September W&B projects."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import wandb

ENTITY = "kayamaguchi-uc-san-diego"
PROJECTS = (
    "BNJetTag-EBOPs-N8",
    "BNJetTag-Batch20260917",
    "BNJetTag-Engram-Experimental",
)
METRICS = (
    "epoch",
    "best_epoch",
    "best_val_macro_auc",
    "checkpoint_ebops",
    "budget_met",
    "val_categorical_accuracy",
    "val_macro_auc",
    "ebops",
    "target_ebops",
    "categorical_accuracy",
)


def scope(project: str, group: str | None) -> tuple[str, str]:
    if project == "BNJetTag-EBOPs-N8":
        return {
            "ebops-n8-20260910": ("initial_budget_pilot", "scientific_training"),
            "ebops-n8-20260910-costfirst": ("resource_priority_retry", "scientific_training"),
            "ebops-n8-20260911-long-budget": ("standalone_long_budget", "scientific_training"),
            "ebops-n8-20260912-ablation": ("seven_arm_ablation", "scientific_training"),
        }[group]
    if project == "BNJetTag-Batch20260917":
        return ("architecture_a00_a11" if group == "batch20260917" else "attention_b00_b04", "scientific_training")
    if group == "engram-screen":
        return "original_engram_e00_e03", "scientific_training"
    if group == "constituent-20260922-diagnostic-tf32":
        return "constituent_tf32_diagnostic", "diagnostic_attempt"
    return "constituent_n8_n64_screen", "scientific_training"


def clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    records = []
    project_summary = []
    for project in PROJECTS:
        runs = list(api.runs(f"{ENTITY}/{project}", per_page=100))
        states = Counter(run.state for run in runs)
        groups = Counter(run.group for run in runs)
        project_summary.append({
            "project": project,
            "wandb_run_records": len(runs),
            "states": dict(sorted(states.items())),
            "groups": dict(sorted(groups.items())),
        })
        for run in runs:
            config = dict(run.config)
            summary = dict(run.summary)
            experiment_scope, role = scope(project, run.group)
            records.append({
                "project": project,
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "group": run.group,
                "job_type": run.job_type,
                "created_at": run.created_at,
                "experiment_scope": experiment_scope,
                "record_role": role,
                "configuration": {
                    "seed": config.get("seed"),
                    "n_constituents": config.get("arch_n_part"),
                    "variant": config.get("variant"),
                    "config_hash": config.get("config_hash"),
                },
                "summary_metrics": {
                    key: clean(summary[key]) for key in METRICS if key in summary
                },
                "url": run.url,
            })
    records.sort(key=lambda row: (PROJECTS.index(row["project"]), row["created_at"], row["run_id"]))
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "entity": ENTITY,
        "scope": "All run records in the three September W&B projects shown in the project dashboard.",
        "counting_note": "W&B records are execution records, not always unique scientific comparisons. Diagnostic attempts and the statically rejected E07/N64 case are identified separately.",
        "summary": {
            "projects": len(PROJECTS),
            "wandb_run_records": len(records),
            "scientific_training_records": sum(row["record_role"] == "scientific_training" for row in records),
            "diagnostic_attempts": sum(row["record_role"] == "diagnostic_attempt" for row in records),
            "statically_rejected_cases_not_in_wandb": 1,
        },
        "projects": project_summary,
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Captured {len(records)} W&B run records in {args.output}")


if __name__ == "__main__":
    main()
