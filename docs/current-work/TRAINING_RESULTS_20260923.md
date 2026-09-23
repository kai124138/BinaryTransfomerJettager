# Final 1,000-epoch training results — 23 September 2026

**Snapshot: 23 September 2026, 15:53 PDT (22:53 UTC).** All 12 architecture runs and all 15 attention runs reached their configured 1,000 epochs. Five architecture runs recorded a checkpoint within their own final cost limits. None of the 15 attention runs reached the shared 350,000-EBOP limit.

[Machine-readable results](training-results-20260923.json) · [Architecture protocol](TRAINING_BATCH_PLAN_WITH_FROZEN_BACKBONE_FOLLOWUP.md) · [Attention protocol](BATCH20260918_ATTENTION_STUDY.md) · [Training curves](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917)

These are trainer-recorded measurements on the 124,000-jet internal-validation split. They have not been recomputed from saved predictions on the held-out 260,000-jet archive. The architecture study has one initialization seed; its rankings do not establish training-seed robustness. EBOPs are a differentiable computational-cost estimate, not measured FPGA resources or latency.

## Completion and feasibility

| Campaign | Runs | Reached 1,000 epochs | Runs with a feasible checkpoint |
|---|---:|---:|---:|
| Architecture, A00–A11 | 12 | 12 | 5 |
| Attention, B00–B04 × seeds 4–6 | 15 | 15 | 0 |
| **Total** | **27** | **27** | **5** |

The architecture Job completed at 2026-09-23T05:30:23Z. The attention Job completed at 2026-09-22T07:44:42Z. Kubernetes reports every index in both Jobs as succeeded.

## Architecture results

The selected columns report the highest validation categorical accuracy among checkpoints at or below that run's final target. A dash means the run never reached its target. Latest-epoch metrics are included to show the endpoint of training and must not be substituted for the selected checkpoint.

| Arm | Selected epoch | Selected accuracy | Selected AUC | Selected EBOPs | Latest accuracy | Latest AUC | Latest EBOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| A00 | 1,000 | 59.80% | 0.8585 | 323,771 | 59.80% | 0.8585 | 323,771 |
| A01 | 618 | 58.25% | 0.8473 | 344,270 | 58.11% | 0.8507 | 354,670 |
| A02 | 913 | **61.08%** | 0.8612 | 349,298 | 60.25% | 0.8619 | 326,839 |
| A03 | 1,000 | 59.44% | 0.8563 | 346,222 | 59.44% | 0.8563 | 346,222 |
| A04 | — | — | — | — | 41.50% | 0.7087 | 721,198 |
| A05 | — | — | — | — | 46.96% | 0.7660 | 2,486,562 |
| A06 | — | — | — | — | 38.00% | 0.6721 | 498,373 |
| A07 | — | — | — | — | 38.32% | 0.6988 | 362,195 |
| A08 | — | — | — | — | 35.18% | 0.6647 | 550,390 |
| A09 | — | — | — | — | 33.81% | 0.6545 | 721,178 |
| A10 | — | — | — | — | 28.43% | 0.6311 | 721,198 |
| A11 | 789 | **62.62%** | **0.8749** | 479,462 | 62.27% | 0.8730 | 482,849 |

**A02 is the leading candidate under 350k.** It combines channel-wise learned activation widths with FFN32 at N=8. A11 records the largest feasible validation accuracy, but it uses a separate 500k target. A00, A01 and A03 are the other feasible 350k runs.

The N16/N32 variants did not produce a feasible checkpoint, including the shallower, narrower and reduced-head designs. A07 came closest at 362,195 EBOPs but remained above 350k. The latest measurements of several infeasible runs are much worse than their early unconstrained checkpoints; the current record does not attribute that degradation to a single cause.

## Attention results

All rows are means across matched seeds 4, 5 and 6 at the final epoch; spreads are sample standard deviations. None is a budget-compliant result.

| Arm | Change | Validation accuracy | Validation AUC | Mean EBOPs |
|---|---|---:|---:|---:|
| B00 | Reference | 34.94% ± 7.05 pp | 0.6702 ± 0.0687 | 721,212 |
| B01 | One attention head | 35.99% ± 2.05 pp | 0.6693 ± 0.0080 | **465,004** |
| B02 | No positional table | 30.17% ± 8.76 pp | 0.6235 ± 0.1074 | 721,193 |
| B03 | 8-bit attention probabilities | **39.02% ± 4.60 pp** | **0.6999 ± 0.0273** | 688,438 |
| B04 | Gradual budget schedule | 36.33% ± 2.71 pp | 0.6713 ± 0.0186 | 721,193 |

B01 reduced cost most, but its approximately 465k endpoint still exceeds the target by about 33%. B03 has the strongest final mean accuracy/AUC. B02 has the largest seed spread and its seed-6 run ended at 20.08% accuracy and 0.5000 AUC. Continuing the campaign to 1,000 epochs did not solve the 350k feasibility problem.

## Relationship to the other September studies

The original seven N8 EBOP ablations also completed 1,000 epochs. Their updated selected-checkpoint held-out evaluation is in [the main result table](../../README.md#post-conference-ebop-constrained-training) and [machine-readable record](../../results/post_conference/ablation_metrics.json). All seven displayed checkpoints fit within 350k; channel-wise quantization has the highest held-out categorical accuracy, while reduced FFN has the highest held-out AUC.

The original Engram E00–E03 continuation and the fresh N8/N64 screen use different protocols and, for memory models, an augmented cost convention. See the [Engram record](ENGRAM_STUDY.md) and [N8/N64 exploratory screen](CONSTITUENT_SCREEN_20260923.md). Their results are not pooled with the native-only A/B table.

## Decision supported by this snapshot

A02 should be evaluated on the held-out archive and repeated across additional initialization seeds before it is treated as a final model. A11 is the corresponding candidate if the allowable computational ceiling is 500k. No B-series arm is eligible under 350k, so the attention campaign supplies a negative result under its tested settings rather than a finalist.

[Return to current work](README.md).
