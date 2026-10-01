# Results and next steps — 1 October 2026

The completed pT-weighting experiment gives a negative result for its tested recipe. The earlier architecture campaign produced feasible candidates. The separate twelve-run confirmation queue remains incomplete, with its saved checkpoint generations now recovered. Chang will use option (c), traced-cost PID, subject to the missing b5 readout and a new pilot; Delta has resource-canary evidence but no screen result.

This update preserves the scientific records through 29 September and adds reviewed cluster observations, a verified 253-file storage recovery, the option-(c) decision and bounded code-compatibility checks. The [provenance JSON](results-status-20261001.json) records source dates and hashes alongside operational observations. Recovered logs and state metadata are diagnostic evidence; no new tagging-performance measurement is promoted.

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
| Twelve-run confirmation queue | The Job has `Failed=True`, reason `FailedIndexes`, at `2026-09-26T07:24:11Z`. All twelve current checkpoint generations, including model, optimizer and state files, are recovered. Saved progress is 280–318 of 1,000 epochs. | Diagnose selected-checkpoint reload failures and verify resumability. The exact final queue trigger is not established. `status.failed=5` is a Kubernetes counter, not five failed scientific arms. |
| Chang regime-B pilot | The `b3` readout certified four primary/sensitivity checkpoints for C-s1 and F-s1; A07-350-s1 had no feasible checkpoint. K1 fired. All five `b5` snapshot states record 500 completed epochs; final logs contain stored-cost reload verification passes. | The named certification and entropy outputs are missing. Analyze the recovered controller histories and complete the full rule analysis. Option (c) is selected, but the amendment, new freeze and replacement pilot remain required; production stays blocked. |
| Delta | Resource canaries ran. E at K=4 reached roughly 100 epochs with finite training and acceptable recorded RSS before stopping to yield the GPU. A07 at K=3 ran out of GPU memory on A10; its two surviving arms completed 110 epochs and passed the host-RSS gate. | No completed Delta screen result. Product, packing and the frozen study's resource conditions require resolution before launch. |
| GPU-product benchmark | Of 16 Jobs, nine completed and seven did not schedule within the allowed window. All 70 executed arm-runs completed 21 epochs and passed checkpoint verification. | Product choice is pending. Actual schedulable GPU counts were not measured; the chosen product still needs its 110-epoch canary and cost certification. |

The original Job-specific pod queries were empty, so the cluster capture recovered no container logs. The subsequent storage reader recovered 37 saved arm logs. Some logs contain multiple process attempts, earlier CUDA/nonfinite errors and later reload assertions without absolute timestamps. They support failure diagnoses but do not uniquely identify the scientific arm that caused the final Job failure. The recovered current generations supersede the earlier handoff's approximate progress ranges; no automatic resume is implied.

NRP authentication succeeded. One authorized CPU-only reader accessed `kai-data` read-only and transferred 253 files totaling 63,514,020 bytes. Independent review checked every accepted file through the transport and export hash chain. The reader exited zero without a restart at `2026-10-01T05:54:10Z`; its Job completed at `05:54:59Z`. All seventeen run pointers and their selected current generations are preserved. The 150 missing entries include optional checkpoint variants and are not a failed-run count. Cache metadata matches the three historical identities; dataset arrays were not rehashed and models were not loaded in this recovery.

Both b5 readout JSON files are missing from the expected output path. The exact readout Job query also returned no Job; neither observation proves that the readout never ran elsewhere. The five recovered snapshots have feasible-selected model records for A-s2 and D-s1, while A-s1, C′-s1 and E1-s1 have no such record. These are stored selection states, not fresh certification. The initial recovery scope did not include `activation_widths.jsonl`. The five historical files have now been recovered separately and remain to be analyzed. Saved eBOP stdout must not be substituted for actual in-training cost. Legacy training Jobs still lack immutable-handoff context; new diagnostic provenance cannot retroactively supply it.

At `2026-10-01T06:57:17.233207+00:00`, the new historical b5 diagnostic `kai-chang1001-readoutb5-42abed-r1` was Running. It uses 8 CPU, 24 GiB RAM and no GPU, with a four-hour deadline and no retry. A partial certification log has appeared; complete certification, entropy and rule analysis remain pending. The five controller histories exceed the wrapper's 8 MiB per-file export bound. Separate read-only recovery copied all five files, totaling 880,343,029 bytes, and matched each against a fresh source SHA-256 and unchanged file metadata. The running wrapper's incomplete telemetry status remains explicit; the separate recovery does not turn its final result into an automatic PASS.

The one-A10 confirmation inference diagnostic `kai-confirm1001-replay-a02s3-r1` failed before input checks, model loading or inference. Its output directory's parent was absent. The handoff init passed; the main container exited1 at `2026-10-01T06:47:50Z`, with zero restarts and no retry. This startup defect is separate from the original checkpoint-metric failures, whose cause remains unresolved. A startup fix, corresponding test and reviewed new attempt are required.

Eight additional A02 seed-1 artifacts were recovered, including the historical selected model and its epoch-1000 state. Model SHA-256 `b2d5e8940b5132d782de20bc9aab09ec7e37d1543472e7566627b33d66ec6e45`, config bytes and stored final source/config/data identities match the historical evaluation association. No model inference, array rehash or hardware validation was performed. A converter that preserves the selected model's channel precisions and fresh C simulation remain prerequisites. Sources for these operational updates are under `operational_evidence.diagnostic_followup` in the [provenance JSON](results-status-20261001.json).

Chang's readout is a pilot diagnostic. On 1 October, Kai selected **option (c): feed traced eBOPs directly to the PID**. The decision is recorded separately from historical runs. It requires a dated amendment, a new frozen revision and the applicable preflight and replacement-pilot checks. It does not clear K1 or production.

The GPU benchmark is operational telemetry. Its projections assume available GPU counts and use one node per Job shape; they do not establish the fastest deployable campaign. Source records: `confirmation_later_status`, `chang_run`, `chang_b3_rules`, `delta_run` and `gpu_benchmark_verify` in the [provenance JSON](results-status-20261001.json).

The [code-compatibility update at the reviewed commit](https://github.com/kai124138/BinaryTransfomerJettager/blob/e56fd7361f0299c1721aa053506d1eabc092cdba/docs/current-work/CODE_COMPATIBILITY_20261001.md) records a bounded engineering PASS: 217 passing comparisons comprising 21 interface/behavior contracts, 100 configuration builds and 96 checkpoint-reload combinations. The work is available in [draft pull request #1](https://github.com/kai124138/BinaryTransfomerJettager/pull/1) and is not merged into main. Synthetic reload differences were zero within the `1e-7` tolerance. Historical metric reproduction, calibrated-width remeasurement and attribution of 74 checkpoint paths remain pending. This does not establish full historical compatibility or hardware readiness; uncovered historical entry points remain retained.

## Work remaining, in dependency order

1. Repair and review the confirmation diagnostic startup path, complete the bounded replay, then determine checkpoint resumability. Retain uncertainty about the final queue trigger. Analyze the five recovered b5 controller histories.
2. Complete the missing Chang certification, entropy and rule readout. Implement the selected option (c) in a dated revision, then repeat the required freeze, CPU checks and replacement pilot. K1 remains triggered until the remedy passes its gates.
3. Choose the GPU product and packing from measured throughput and current capacity. Complete the selected product's 110-epoch memory/RSS canary, cost certification and decision-time checks. Resolve Delta's resource conditions and remaining preflight and fingerprint gates.
4. Launch or resume only with explicit action-time authority, a validated immutable run handoff, a passing manifest check and cleared scientific gates. Saved pilot progress alone does not authorize production.
5. After full schedules finish, validate selected checkpoints and report matched-seed comparisons with intervals. Evaluate fixed selections on the held-out archive; retain separate N8/N64 budgets and native versus augmented cost conventions. Diagnose Engram's metric-reproduction failure before using its final results.
6. Review the bounded code-compatibility change in draft PR #1, recover missing historical associations and complete applicable metric and calibrated-width checks. Synthesize selected, validated candidates separately from training. The earlier [R4 synthesis attempt](R4_HARDWARE_SYNTHESIS.md) stopped at a Vitis frontend failure; it supplies no current-candidate hardware result.

The completed [60-model fixed-precision baseline](../../README.md#pre-conference-fixed-precision-study) remains the reference. FPGA resources, latency, initiation interval and timing closure for current candidates still require hardware measurements.

[Current-work index](README.md) · [Historical experiment index](EXPERIMENT_INDEX_20260923.md)
