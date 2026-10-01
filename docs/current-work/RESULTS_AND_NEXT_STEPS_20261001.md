# Results and next steps — 1 October 2026

The completed pT-weighting experiment gives a negative result for its tested recipe. The earlier architecture campaign produced feasible candidates. The separate twelve-run confirmation queue has no final result in the saved record. Chang remains blocked at its pilot decision; Delta has resource-canary evidence but no screen result.

This update uses saved records through 29 September; live cluster status on 1 October is unknown. The [provenance JSON](results-status-20261001.json) records source dates, evidence status and SHA256 hashes.

## Completed training and held-out results

The pT-weighting campaign completed 24 runs: unweighted BASE, capped weighting PTW5 and uncapped weighting PTWNC, each with seeds 1–8. All use N=8, `l1x3` inputs and W1A8, with a 101-epoch maximum and early stopping. The following macro one-vs-rest AUC values use the same 260,000-jet held-out archive. Spreads are sample standard deviations across eight seeds.

| Training weights | Held-out macro AUC, mean ± sd | Paired change from BASE, 95% t-interval |
|---|---:|---:|
| Unweighted BASE | 0.8711 ± 0.0013 | Reference |
| pT weighting, cap 5 | 0.8603 ± 0.0019 | −0.0108 [−0.0124, −0.0093] |
| pT weighting, uncapped | 0.8587 ± 0.0020 | −0.0125 [−0.0148, −0.0101] |

Both weighted arms are lower on all eight matched seeds. The intervals use seven degrees of freedom. The primary metric, paired comparison rule and pT-sextile analysis were post hoc. These values quote the 26 September verification from saved predictions; no new recomputation was performed. Source: `pt_weighting_verify` in the [source record](results-status-20261001.json).

The [architecture and attention continuations](TRAINING_RESULTS_20260923.md) completed all 27 training loops at 1,000 epochs. Five of the 12 architecture runs recorded checkpoints within their configured budgets; none of the 15 attention runs met the 350k eBOP target. Feasibility is trainer-recorded; the architecture study used one seed.

The [24 September record](CONFIRMATION_RUNS_20260924.md) reports deterministic reloads for A02 and A11. These are previously reported seed-1 measurements on 260,000 held-out jets, not freshly verified results.

| Candidate | Budget | Selected checkpoint eBOPs | Held-out accuracy | Held-out macro-OvR AUC |
|---|---:|---:|---:|---:|
| A02, N=8, seed 1 | 350,000 | 349,298 | 60.7850% | 0.860432 |
| A11, N=8, seed 1 | 500,000 | 479,462 | 62.3335% | 0.873813 |

These candidates have different budgets. Their values do not establish a same-budget advantage or training-seed robustness.

All four N16 [Engram loops](ENGRAM_STUDY.md) reached 1,000 epochs. E00 passed outer validation; E01–E03 failed selected-checkpoint metric reproduction. None had a feasible checkpoint under the augmented 350k target. The exploratory [N8/N64 screen](CONSTITUENT_SCREEN_20260923.md) produced no feasible checkpoint under its tested selection rule.

## Incomplete campaigns and operational evidence

| Workstream | Latest saved evidence | Remaining gate |
|---|---|---|
| Twelve-run confirmation queue | The 24 September snapshot showed training in progress. A later handoff reports failure around 26 September, diagnosed on 27 September, with no relaunch at the handoff. It reports N8 epochs 281–300 and N64 epochs 240–280 of 1,000. | Recover current status and exact checkpoint state. No final confirmation results are available in the saved record. |
| Chang regime-B pilot | The three-arm `b3` pilot reached its epoch-500 pause. The readout certified the primary and sensitivity checkpoints for C-s1 and F-s1; A07-350-s1 had no feasible checkpoint. The K1 PID-input rule fired. | Full terminal `b5` readout and Kai's pending (c)/(d) decision. Production remains blocked. |
| Delta | Resource canaries ran. E at K=4 reached roughly 100 epochs with finite training and acceptable recorded RSS before stopping to yield the GPU. A07 at K=3 ran out of GPU memory on A10; its two surviving arms completed 110 epochs and passed the host-RSS gate. | No completed Delta screen result. Product, packing and the frozen study's resource conditions require resolution before launch. |
| GPU-product benchmark | Of 16 Jobs, nine completed and seven did not schedule within the allowed window. All 70 executed arm-runs completed 21 epochs and passed checkpoint verification. | Product choice is pending. Actual schedulable GPU counts were not measured; the chosen product still needs its 110-epoch canary and cost certification. |

The confirmation failure is a local report, not a fresh cluster observation. Its epoch ranges require confirmation from per-run checkpoints. The report attributes the whole-pod exit to one arm failing.

Chang's readout is a pilot diagnostic. Option (c) feeds traced eBOPs to the PID; option (d) scales each arm's setpoint by its measured ratio. Neither choice is approved. Changes require a new frozen record and the applicable preflight and pilot checks.

The GPU benchmark is operational telemetry. Its projections assume available GPU counts and use one node per Job shape; they do not establish the fastest deployable campaign. Source records: `confirmation_later_status`, `chang_run`, `chang_b3_rules`, `delta_run` and `gpu_benchmark_verify` in the [provenance JSON](results-status-20261001.json).

## Work remaining, in dependency order

1. Recover terminal `b5` artifacts and confirmation state through authorized access. Preserve logs and checkpoint identities before any resume decision.
2. Complete the Chang readout, resolve Kai's (c)/(d) decision and repeat the required freeze, CPU checks and pilot for the selected change. K1 fired; its required decision remains pending.
3. Choose the GPU product and packing from measured throughput and current capacity. Complete the selected product's 110-epoch memory/RSS canary, cost certification and decision-time checks. Resolve Delta's resource conditions and remaining preflight and fingerprint gates.
4. Launch or resume only with explicit action-time authority, a validated immutable run handoff, a passing manifest check and cleared scientific gates. Saved pilot progress alone does not authorize production.
5. After full schedules finish, validate selected checkpoints and report matched-seed comparisons with intervals. Evaluate fixed selections on the held-out archive; retain separate N8/N64 budgets and native versus augmented cost conventions. Diagnose Engram's metric-reproduction failure before using its final results.
6. Complete the code-line merge with CPU build and checkpoint-reload gates for the existing configurations. Synthesize the selected, validated candidates separately from training. The earlier [R4 synthesis attempt](R4_HARDWARE_SYNTHESIS.md) stopped at a Vitis frontend failure; it supplies no current-candidate hardware result.

The completed [60-model fixed-precision baseline](../../README.md#pre-conference-fixed-precision-study) remains the reference. FPGA resources, latency, initiation interval and timing closure for current candidates still require hardware measurements.

[Current-work index](README.md) · [Historical experiment index](EXPERIMENT_INDEX_20260923.md)
