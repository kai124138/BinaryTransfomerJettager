# Training progress — 20 September 2026

Cluster verified at **2026-09-20T20:46:42Z** (13:46 PDT). The three public training campaigns have **zero active workers**. Screening runs are awaiting a continuation decision; completion of a screening interval does not imply completion of the configured 1,000-epoch schedule.

| Campaign | Runs | Progress | Execution state |
|---|---:|---|---|
| Architecture screen, A00–A11 | 12 | 100 / 1,000 epochs each | All 12 screening indexes completed September 18 |
| Attention / precision / schedule, B00–B04 × seeds 4–6 | 15 | 400 / 1,000 epochs each | All 15 screening indexes completed September 19 |
| Original EBOP ablations | 7 | 1,000 / 1,000 epochs each | All seven completed; final distillation run finished September 20 |

[Machine-readable job evidence](training-status-20260920.json) · [Architecture metrics](live-status.json) · [Attention metrics](batch20260918-status.json) · [Original ablation completion and final-epoch metrics](../../results/post_conference/ablation-training-status-20260920.json).

## Attention screen: final-epoch observations

All 15 jobs completed the 400-epoch screen. Final-epoch metrics were recoverable from **9 of 15** retained container logs; the other six completed containers no longer expose their logs. Missing values below are unavailable, not zero. These are internal-validation measurements on 124,000 jets, rounded by the trainer to six decimals. They have not been recomputed from saved predictions.

| Run | Epochs | Latest val accuracy | Latest val AUC | Latest EBOPs / 350k |
|---|---:|---:|---:|---:|
| [B00 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/417bc88cd02e) | 400 / 1,000 | Unavailable | — | — |
| [B01 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/46c390b6c0e4) | 400 / 1,000 | 34.5379% | 0.656619 | 465,521 / 350,000 |
| [B02 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/6ddab006f995) | 400 / 1,000 | 33.6855% | 0.648487 | 721,208 / 350,000 |
| [B03 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/64aa21dac400) | 400 / 1,000 | 32.3379% | 0.639498 | 690,488 / 350,000 |
| [B04 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/37f5c59cacf9) | 400 / 1,000 | Unavailable | — | — |
| [B00 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/b89beb498eb8) | 400 / 1,000 | Unavailable | — | — |
| [B01 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/cd187434fdfc) | 400 / 1,000 | Unavailable | — | — |
| [B02 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/f0d2d1ea5296) | 400 / 1,000 | Unavailable | — | — |
| [B03 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/eab07105ae52) | 400 / 1,000 | Unavailable | — | — |
| [B04 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/199429658783) | 400 / 1,000 | 31.9185% | 0.611768 | 721,220 / 350,000 |
| [B00 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/7239c1688717) | 400 / 1,000 | 31.5903% | 0.651301 | 721,722 / 350,000 |
| [B01 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/8f1fa97adfbd) | 400 / 1,000 | 32.6685% | 0.660689 | 465,506 / 350,000 |
| [B02 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/b83fc8bbb99e) | 400 / 1,000 | 41.3782% | 0.719726 | 723,763 / 350,000 |
| [B03 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/60550ff8dac9) | 400 / 1,000 | 32.4806% | 0.660870 | 688,425 / 350,000 |
| [B04 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/24a674f320ef) | 400 / 1,000 | 41.3153% | 0.708155 | 721,233 / 350,000 |

**All nine observable last checkpoints exceed the 350,000-EBOP target**, at 465,506–723,763 EBOPs. Their validation accuracies span 31.5903%–41.3782%, and each reports the cost-controller coefficient at its configured maximum, beta=0.001. This indicates that the observed final checkpoints have not achieved the intended accuracy/resource tradeoff. It does not establish whether an earlier feasible checkpoint exists, and the six unavailable rows cannot be ranked. No winner or promotion is declared. Inspect complete checkpoint histories and the cost/controller behavior before selecting continuations.

The [allowlisted final-epoch source lines](training-log-metrics-20260920.json) retain timestamps and line hashes. The [protocol](BATCH20260918_ATTENTION_STUDY.md) describes the matched arms and final-budget selection rule. Full three-seed comparisons require the six missing summaries and selected-checkpoint records.

## Architecture screen

The original 12-run screen remains complete at 100 epochs per run, with no continuation job present at this cluster check. The existing September 18 W&B metrics remain the latest verified numerical snapshot. Every latest checkpoint in that snapshot exceeds its own target. A11 records 61.2129% validation accuracy at 688,677 EBOPs against its 500k target; among the 350k-target rows, A00 records 58.3468% at 599,799 EBOPs. These observations are not feasible-checkpoint selections or claims of statistical superiority.

## Original seven ablations

The 1,000-epoch job completed all seven indexes at **2026-09-20T06:08:41Z**. The previously stalled distillation arm resumed and reached epoch 1,000: its final training-log AUC is **0.844425**, at **344,430 EBOPs**. This is the final epoch, not necessarily the selected checkpoint; no categorical accuracy was printed on that line.

The main README's September 15 held-out table is a separately dated evaluation snapshot. Its interim checkpoint evaluations remain historical until final selected checkpoints are re-evaluated. Training completion alone does not update held-out accuracy or checkpoint digests.

## Verification scope

This update uses read-only Kubernetes job status and retained logs. An anonymous W&B query did not expose the projects, and authenticated W&B summaries were not refreshed for this snapshot. Existing W&B source timestamps are preserved. No training was started, resumed, or promoted, and no model, dataset, project visibility, or hardware measurement changed. The earlier fixed-precision study remains the archived 60-run result set in the main README.

[Return to current work](README.md).
