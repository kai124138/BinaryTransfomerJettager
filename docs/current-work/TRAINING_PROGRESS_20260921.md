# Training progress — 21 September 2026

**Snapshot: 21 September 2026, 17:39 PDT (22 September, 00:39 UTC).** Fifteen of the 27 public continuation runs have finished their 1,000-epoch schedules; 12 remain active. Five architecture runs have recorded a checkpoint within their own final budgets. None of the 15 attention runs has recorded a checkpoint within its 350,000-EBOP budget.

| Campaign | Runs | Finished runs | Active runs | Runs with a feasible checkpoint |
|---|---:|---:|---:|---:|
| Architecture, A00–A11 | 12 | 2 | 10 | 5 |
| Attention, B00–B04 × seeds 4–6 | 15 | 13 | 2 | 0 |

Indexed-job status was read at **2026-09-22T00:38:58.628694+00:00**; the checkpoint read completed at **2026-09-22T00:39:42.715826+00:00**. The original seven EBOP ablations completed previously and are separate from these 27 runs. [Machine-readable evidence](training-status-20260921.json) · [September 20 launch record](FULL_LENGTH_CONTINUATION_20260920.md).

## Architecture: recorded checkpoints within budget

All five rows below use eight constituents and initialization seed 1. They are the best feasible points recorded in durable trainer state, selected by validation categorical accuracy. Completed epochs describe training progress; selected epoch identifies the saved measurement and is displayed one-based.

| Run | Completed epochs | Selected epoch | Validation accuracy | Macro-OvR AUC | Native EBOPs | Budget |
|---|---:|---:|---:|---:|---:|---:|
| A00 / seed 1 | 761 | 685 | 58.77% | 0.8502 | 349,322 | 350,000 |
| A01 / seed 1 | 780 | 618 | 58.25% | 0.8473 | 344,270 | 350,000 |
| A02 / seed 1 | 663 | 638 | 60.55% | 0.8580 | 342,832 | 350,000 |
| A03 / seed 1 | 1,000 | 1,000 | 59.44% | 0.8563 | 346,222 | 350,000 |
| A11 / seed 1 | 534 | 502 | 62.38% | 0.8736 | 453,439 | 500,000 |

A02 is a candidate for final evaluation at the 350k budget, not an established improvement over the earlier held-out results. A03 has finished its schedule. A11 uses a larger 500k budget and must remain separate from the 350k comparison. Four of these five runs are still training.

## Architecture: all latest recorded measurements

Latest history rows describe individual epochs, not necessarily the selected checkpoints. All metrics use the internal-validation split of 124,000 jets. Reading history and checkpoint state is not atomic while training continues.

| Run | Committed epochs | History epoch | Latest validation accuracy | Latest AUC | Latest EBOPs | Budget | Ever feasible? |
|---|---:|---:|---:|---:|---:|---:|---|
| A00 / seed 1 | 761 | 762 | 57.17% | 0.8442 | 348,810 | 350,000 | Yes |
| A01 / seed 1 | 780 | 780 | 56.67% | 0.8416 | 377,038 | 350,000 | Yes |
| A02 / seed 1 | 663 | 663 | 57.13% | 0.8471 | 344,245 | 350,000 | Yes |
| A03 / seed 1 | 1,000 | 1,000 | 59.44% | 0.8563 | 346,222 | 350,000 | Yes |
| A04 / seed 1 | 1,000 | 1,000 | 41.50% | 0.7087 | 721,198 | 350,000 | No |
| A05 / seed 1 | 681 | 681 | 37.28% | 0.7099 | 2,486,582 | 350,000 | No |
| A06 / seed 1 | 612 | 612 | 32.02% | 0.6561 | 500,953 | 350,000 | No |
| A07 / seed 1 | 994 | 994 | 42.96% | 0.7407 | 363,891 | 350,000 | No |
| A08 / seed 1 | 545 | 546 | 29.49% | 0.6550 | 550,959 | 350,000 | No |
| A09 / seed 1 | 887 | 888 | 33.99% | 0.6635 | 723,832 | 500,000 | No |
| A10 / seed 1 | 622 | 623 | 35.80% | 0.6952 | 721,961 | 250,000 | No |
| A11 / seed 1 | 534 | 535 | 61.13% | 0.8694 | 454,884 | 500,000 | Yes |

A03 and A04 have finished. A04 has no feasible checkpoint: its final history row records 41.50% accuracy, AUC 0.7087 and 721,198 EBOPs. Its saved best-AUC point, at epoch 39, records 64.77% accuracy at 2,771,603 EBOPs. These two checkpoints motivate a trajectory analysis; they do not establish that compression caused the accuracy change.

## Attention: no feasible checkpoint in any of 15 runs

Thirteen runs have finished 1,000 epochs. B02/seed 6 and B03/seed 6 have committed epochs 989 and 867, respectively. All 15 durable states still record `best_feasible=null`. Their lowest observed costs span **464,979–721,173 EBOPs**, above the common 350,000 target. All latest history rows record the controller coefficient at its configured maximum, beta=0.001.

| Run | Committed epochs | History epoch | Latest validation accuracy | Latest AUC | Latest EBOPs | Lowest recorded EBOPs |
|---|---:|---:|---:|---:|---:|---:|
| B00 / seed 4 | 1,000 | 1,000 | 41.77% | 0.7247 | 721,208 | 721,163 |
| B00 / seed 5 | 1,000 | 1,000 | 27.70% | 0.5931 | 721,230 | 721,173 |
| B00 / seed 6 | 1,000 | 1,000 | 35.36% | 0.6929 | 721,198 | 721,158 |
| B01 / seed 4 | 1,000 | 1,000 | 35.19% | 0.6777 | 465,004 | 464,989 |
| B01 / seed 5 | 1,000 | 1,000 | 38.31% | 0.6684 | 465,004 | 464,984 |
| B01 / seed 6 | 1,000 | 1,000 | 34.46% | 0.6618 | 465,004 | 464,979 |
| B02 / seed 4 | 1,000 | 1,000 | 34.57% | 0.6751 | 721,198 | 721,158 |
| B02 / seed 5 | 1,000 | 1,000 | 35.86% | 0.6954 | 721,188 | 721,168 |
| B02 / seed 6 | 989 | 989 | 25.53% | 0.5950 | 721,213 | 721,158 |
| B03 / seed 4 | 1,000 | 1,000 | 42.28% | 0.7110 | 688,435 | 688,400 |
| B03 / seed 5 | 1,000 | 1,000 | 41.02% | 0.7199 | 688,445 | 688,385 |
| B03 / seed 6 | 867 | 867 | 27.08% | 0.5823 | 688,472 | 688,395 |
| B04 / seed 4 | 1,000 | 1,000 | 33.75% | 0.6640 | 721,198 | 721,163 |
| B04 / seed 5 | 1,000 | 1,000 | 36.06% | 0.6575 | 721,188 | 721,173 |
| B04 / seed 6 | 1,000 | 1,000 | 39.16% | 0.6924 | 721,193 | 721,173 |

The completed attention runs have not achieved the intended accuracy/resource tradeoff. Additional training through 1,000 epochs did not produce a feasible checkpoint in those 13 runs. This is a negative result under the current settings, not proof that the architecture cannot meet the target under any training procedure. When no feasible checkpoint exists, the trainer exports its minimum-cost fallback; that fallback is not a successful budget-constrained result.

## Interpretation and next checks

The within-budget N=8 checkpoints warrant evaluation, but the current measurements do not establish strong absolute performance, a statistically significant gain, or FPGA readiness. Architecture runs have one initialization seed. Latest epochs, best feasible checkpoints and historical held-out evaluations must not be mixed in a ranking.

The next diagnostic steps are to inspect accuracy and per-layer cost across training, determine which operations keep cost above budget, and evaluate selected checkpoints against matched baselines on the same split. A structural cost floor and damage from compression pressure are hypotheses to test, not established causes. No new training sweep or protocol change was made for this status update.

## Verification scope

This update reads Kubernetes job status and committed trainer records. It does not recompute predictions, re-evaluate the held-out archive, validate exported arithmetic, or measure FPGA resources. Existing September 15 held-out tables and figures retain their original dates and scope. Code/config hashes, recorded selection points, source paths relative to each run, and exact unrounded metrics are included in the JSON evidence.

[Return to current work](README.md).
