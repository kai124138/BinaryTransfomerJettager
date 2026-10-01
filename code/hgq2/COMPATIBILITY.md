# Research compatibility

The public entry points remain `run_stage.py`, `run_ablation.py`, `convert_binary.py`,
`evaluate_roc.py`, `fold_binary_transformer.py`, `measure_pre_conference_ebops.py`,
`estimate_auc_uncertainty.py` and `record_synthesis_metrics.py`. Public output paths,
explicit checkpoint and teacher arguments, and opt-in experiment tracking retain
their existing behavior.

`train.pt_weights` enables the historical optional jet-pT weighting implementation.
Absent configuration or `enable: false` leaves training unweighted. Enabled weights
are derived from the training partition; validation and checkpoint selection remain
unweighted. Historical capped and uncapped settings remain explicit config choices.

Historical model JSON files are retained under their original names and unchanged
bytes. Names, data paths and tracking-project fields contribute to their config
identities. Renamed public configurations are separate identities even when the
architecture matches. `run_stage.py` and `bnhgq2.train` accept `BNHGQ2_TRAIN_DATA`;
`run_ablation.py` reads prepared arrays under `--root/data`. The pT evaluator
accepts `--train-dir` and `--val-dir`. Use explicit output/checkpoint paths when
replaying a historical configuration in another environment.

The `configs/legacy/` generators preserve historical EBOP names and configuration
fields. Their qualified imports keep them separate from the public generators.
`gen_r14.py`, `gen_r15_gamma.py` and `gen_ptw.py` preserve the remaining historical
generators. `check_legacy_ebops_ablation.py` and `check_legacy_ebops_target.py` select
historical configs explicitly. These check scripts can contain training operations;
their presence is not authorization to run them locally.

| Historical entry point | Canonical implementation |
| --- | --- |
| `convert_final.py` | `convert_binary.py`, including the `run_convert_final` callable alias |
| `roc_final.py` | `evaluate_roc.py` |
| `fold_r14n8.py` | `fold_binary_transformer.py` |
| `ebops_r14.py` | `measure_pre_conference_ebops.py` |
| `uncertainty_r14.py` | `estimate_auc_uncertainty.py` |
| `log_hls_wandb.py` | `record_synthesis_metrics.py` |
| `bnhgq2.convert.pack_for_mulder` | `package_hls_project` |

Aliases preserve callable forwarding and command names. They use public defaults;
they do not recreate old machine paths, tracking accounts or artifact identities.
The pT evaluation scripts under `sample_weighting/` accept explicit checkpoint,
input and output paths. Their default outputs follow the public repository layout.

Frozen Engram, Chang and Delta campaign implementations are not integrated by this
change. Their recorded code bundles remain authoritative. Local compatibility
checks do not establish historical metric reproduction, original calibration-width
remeasurement, production readiness or FPGA synthesis. Missing historical evidence
must remain explicit; uncovered historical entry points are not retired.
