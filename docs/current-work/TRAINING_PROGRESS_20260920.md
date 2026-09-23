# Training progress — 20 September 2026

**September 23 update:** [final continuation results](TRAINING_RESULTS_20260923.md). The observations and original protocol below retain their stated dates.

**Later execution update:** all 27 public architecture/attention runs are submitted for full-length continuation. [Current continuation status](FULL_LENGTH_CONTINUATION_20260920.md). The screening-completion observations below retain their original times.

Cluster verified at **2026-09-20T20:46:42Z** (13:46 PDT). The three public training campaigns have **zero active workers**. Screening runs are awaiting a continuation decision; completion of a screening interval does not imply completion of the configured 1,000-epoch schedule.

| Campaign | Runs | Progress | Execution state |
|---|---:|---|---|
| Architecture screen, A00–A11 | 12 | 100 / 1,000 epochs each | All 12 screening indexes completed September 18 |
| Attention / precision / schedule, B00–B04 × seeds 4–6 | 15 | 400 / 1,000 epochs each | All 15 screening indexes completed September 19 |
| Original EBOP ablations | 7 | 1,000 / 1,000 epochs each | All seven completed; final distillation run finished September 20 |

[Machine-readable job evidence](training-status-20260920.json) · [Architecture metrics](live-status.json) · [Attention metrics](batch20260918-status.json) · [Original ablation completion and final-epoch metrics](../../results/post_conference/ablation-training-status-20260920.json).

## Attention screen: final-epoch observations

All 15 jobs completed the 400-epoch screen. **All 15 final-epoch summaries are now available**, fetched from authenticated W&B at **2026-09-20T21:56:56.882734+00:00**. This supersedes the earlier nine-of-fifteen container-log coverage. These are internal-validation measurements on 124,000 jets; they have not been recomputed from saved predictions.

| Run | Epochs | Latest val accuracy | Latest val AUC | Latest EBOPs / 350k |
|---|---:|---:|---:|---:|
| [B00 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/417bc88cd02e) | 400 / 1,000 | 39.9323% | 0.703529 | 721,737 / 350,000 |
| [B01 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/46c390b6c0e4) | 400 / 1,000 | 34.5379% | 0.656619 | 465,521 / 350,000 |
| [B02 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/6ddab006f995) | 400 / 1,000 | 33.6855% | 0.648487 | 721,208 / 350,000 |
| [B03 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/64aa21dac400) | 400 / 1,000 | 32.3379% | 0.639498 | 690,488 / 350,000 |
| [B04 / seed 4](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/37f5c59cacf9) | 400 / 1,000 | 31.9363% | 0.670348 | 723,748 / 350,000 |
| [B00 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/b89beb498eb8) | 400 / 1,000 | 27.6992% | 0.557795 | 721,223 / 350,000 |
| [B01 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/cd187434fdfc) | 400 / 1,000 | 33.8427% | 0.660193 | 465,516 / 350,000 |
| [B02 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/f0d2d1ea5296) | 400 / 1,000 | 42.0290% | 0.722428 | 721,747 / 350,000 |
| [B03 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/eab07105ae52) | 400 / 1,000 | 41.2226% | 0.716131 | 688,967 / 350,000 |
| [B04 / seed 5](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/199429658783) | 400 / 1,000 | 31.9185% | 0.611768 | 721,220 / 350,000 |
| [B00 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/7239c1688717) | 400 / 1,000 | 31.5903% | 0.651301 | 721,722 / 350,000 |
| [B01 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/8f1fa97adfbd) | 400 / 1,000 | 32.6685% | 0.660689 | 465,506 / 350,000 |
| [B02 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/b83fc8bbb99e) | 400 / 1,000 | 41.3782% | 0.719726 | 723,763 / 350,000 |
| [B03 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/60550ff8dac9) | 400 / 1,000 | 32.4806% | 0.660870 | 688,425 / 350,000 |
| [B04 / seed 6](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917/runs/24a674f320ef) | 400 / 1,000 | 41.3153% | 0.708155 | 721,233 / 350,000 |

**All 15 latest checkpoints exceed the 350,000-EBOP target**, at 465,506–723,763 EBOPs. Their validation accuracies span 27.6992%–42.0290%. The nine retained final-epoch logs each report the cost-controller coefficient at its configured maximum, beta=0.001; that coefficient was not recovered for the remaining six summaries. These final checkpoints have not achieved the intended accuracy/resource tradeoff. A subsequent read of all 27 architecture/attention durable checkpoint states found `best_feasible=null` for every run at its configured final target; see [checkpoint-state evidence](checkpoint-screen-status-20260920.json). No winner is declared.

The [complete W&B source snapshot](wandb-snapshot-20260920.json) preserves exact metrics and source timestamps. The earlier [nine final-epoch log lines](training-log-metrics-20260920.json) remain as corroborating evidence. The [protocol](BATCH20260918_ATTENTION_STUDY.md) describes the matched arms and final-budget selection rule.

The user authorized continuation of every current architecture and attention run to the configured 1,000 epochs on September 20, superseding selective promotion for these campaigns. See the [verified submission and continuation record](FULL_LENGTH_CONTINUATION_20260920.md).

## Architecture screen

The original 12-run screen remains complete at 100 epochs per run, with no continuation job present at this cluster check. A fresh W&B read on September 20 confirmed the existing September 18 numerical values and source timestamps. Every latest checkpoint in that snapshot exceeds its own target. A11 records 61.2129% validation accuracy at 688,677 EBOPs against its 500k target; among the 350k-target rows, A00 records 58.3468% at 599,799 EBOPs. These observations are not feasible-checkpoint selections or claims of statistical superiority.

## Original seven ablations

The 1,000-epoch job completed all seven indexes at **2026-09-20T06:08:41Z**. The previously stalled distillation arm resumed and reached epoch 1,000: its final training-log AUC is **0.844425**, at **344,430 EBOPs**. This is the final epoch, not necessarily the selected checkpoint; no categorical accuracy was printed on that line.

The main README's September 15 held-out table is a separately dated evaluation snapshot. Its interim checkpoint evaluations remain historical until final selected checkpoints are re-evaluated. Training completion alone does not update held-out accuracy or checkpoint digests.

## Verification scope

This update uses read-only Kubernetes job status and retained logs. Authenticated W&B summaries were refreshed after the initial log-only report. All missing attention metrics are recovered, and original per-run source timestamps are preserved. No training was started, resumed, or promoted, and no model, dataset, project visibility, or hardware measurement changed. The earlier fixed-precision study remains the archived 60-run result set in the main README.

[Return to current work](README.md).
