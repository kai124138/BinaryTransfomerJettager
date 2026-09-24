# Full-run confirmation campaign — 24 September 2026

[Machine-readable status](confirmation-status-20260924.json) · [Complete historical experiment index](EXPERIMENT_INDEX_20260923.md) · [Prior N8/N64 screen](CONSTITUENT_SCREEN_20260923.md)

The active confirmation campaign removes the earlier short-screen promotion workflow. All 12 declared runs target **1,000 epochs**. Intermediate validation metrics, eBOP measurements and checkpoints are recorded, but they do not stop a run or decide whether another run starts.

## Deterministic checkpoint evaluation

The selected A02 and A11 seed-1 checkpoints were each loaded twice. Validation and held-out logits were byte-identical across reloads, with maximum absolute difference 0.0. Held-out evaluation uses 260,000 jets and does not select checkpoints.

| Arm | Selected epoch | eBOPs | Held-out accuracy | Held-out macro-OvR AUC |
|---|---:|---:|---:|---:|
| A02 | 913 | 349,298 | 60.7850% | 0.860432 |
| A11 | 789 | 479,462 | 62.3335% | 0.873813 |

A11 uses a separate 500k target, so its accuracy must not be presented as a same-budget win over A02.

## Runs now in the full queue

| Constituents | Arms | Seeds | Runs | Epochs per run | eBOP target |
|---:|---|---|---:|---:|---:|
| 8 | A00, A02, A03 | 2 and 3 | 6 | 1,000 | 350,000 |
| 64 | A07, E02, E05 | 2 and 3 | 6 | 1,000 | 5,000,000 |
| **Total** | | | **12** | | |

The N64 target is separate because the 50-epoch screen placed its strongest one-block and memory models near 4.6–4.8 million operations. The N64 run set includes the one-block A07 control, ungated-memory E02 and four-bit-memory E05.

## Execution policy and utilization

One NVIDIA A10 runs three batch-256 training processes at a time. Steady observed utilization is 99–100%. The scheduler always advances the least-complete runs, which keeps the GPU occupied while allowing every declared run to continue.

Processes restart after 20 epochs to release TensorFlow host memory. This is an operational boundary only. The runner restores the exact model, optimizer, PID and epoch state and continues toward epoch 1,000. The queue has no intermediate decision rungs, and `stop_on_target` is false.

At the dated snapshot, the six N8 runs had reached epochs 120–318. The first three N64 processes had each committed two epochs; the remaining three were queued. These counts report execution progress, not scientific outcomes. Exact per-run values are in the [status JSON](confirmation-status-20260924.json).

## Record boundaries

The prior 50-epoch constituent screen remains published because it records work that actually happened. Its results and limitations are historical evidence, not instructions for the active campaign. Final comparisons will be reported after the full schedules finish; N8 and N64 will retain their separate cost budgets.
