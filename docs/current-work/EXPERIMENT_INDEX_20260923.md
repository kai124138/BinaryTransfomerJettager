# Complete experiment index — 23 September 2026

This page is the entry point for the full published experiment record. It includes the N64 studies and Engram work as well as the architecture and attention campaigns. The [machine-readable W&B registry](../../results/wandb-project-inventory-20260923.json) captures all **86 run records** visible in the three September projects.

## Coverage at a glance

| Study | Scope | Execution result | Main record |
|---|---|---|---|
| Pre-conference precision baseline | N=8/16/32/64 × five precisions × three seeds = 60 runs | 60 complete; held-out AUC reproduced for every seed | [Per-seed results](../../results/pre_conference/auc_per_seed.csv) |
| Initial N8 EBOP work | 13 W&B records: four-arm pilot, resource-priority retry, standalone long run and seven ablations | 12 finished, one failed before an epoch | [Records below](#n8-ebop-project-13-wb-records) |
| Architecture A00–A11 | 12 seed-1 runs, N=8/16/32 | 12 reached 1,000 epochs; five feasible | [Final A/B report](TRAINING_RESULTS_20260923.md) |
| Attention B00–B04 | Five N16 methods × seeds 4/5/6 = 15 runs | 15 reached 1,000 epochs; none feasible | [Final A/B report](TRAINING_RESULTS_20260923.md) |
| Original Engram E00–E03 | Four N16 seed-1 runs | Four training loops complete; E00 outer validation passed, E01–E03 failed metric reproduction; none feasible | [Original Engram report](ENGRAM_STUDY.md) |
| Fresh matched constituent screen | 19 normalized setups × N=8/64 = 38 intended cases | 28 complete, seven partial, two canary-only, one statically rejected; none feasible | [Exact screen results](../../results/constituent_study/results-20260923.json) |
| Frozen-output refinements | Rounded output correction and 4-/8-bit frozen-head fits | Completed software evaluation; two modest held-out improvements | [Exact results](../../results/post_conference/frozen_output_results.json) |
| Hardware work | Softmax/device characterization and R4 export/synthesis investigation | Dated estimates and failure evidence; no new routed Engram/N64 result | [Hardware record](R4_HARDWARE_SYNTHESIS.md) |
| Full-run confirmations | N8 A00/A02/A03 and N64 A07/E02/E05 × seeds 2/3 = 12 runs | Active; every run targets 1,000 epochs with no intermediate promotion gate | [September 24 status](CONFIRMATION_RUNS_20260924.md) |

The 86-record W&B inventory below is a frozen September 23 snapshot. The confirmation runs launched on September 24 are recorded separately and are not silently added to that historical count.

## How the W&B dashboard counts map to experiments

| W&B project | Dashboard records | What those records contain |
|---|---:|---|
| [`BNJetTag-EBOPs-N8`](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-EBOPs-N8) | 13 | Four initial budget runs, one resource-priority retry, one standalone 1,000-epoch run and seven ablations |
| [`BNJetTag-Batch20260917`](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Batch20260917) | 27 | Twelve architecture runs and fifteen attention runs |
| [`BNJetTag-Engram-Experimental`](https://wandb.ai/kayamaguchi-uc-san-diego/BNJetTag-Engram-Experimental) | 46 | Four original Engram runs, five diagnostic TF32 attempts and 37 fresh N8/N64 training runs |
| **Total** | **86** | **81 scientific training records plus five diagnostic attempts** |

The dashboard count is an execution-record count, not a count of independent final models. The fresh constituent screen has **38 intended cases** although only 37 appear as W&B runs: E07/N64 was rejected before launch because its fixed memory arithmetic alone exceeds the 350k target. The five TF32 records diagnose an inference/reload problem and are not additional scientific comparisons.

## Every fresh N64 case

All 18 trained N64 cases completed 50 epochs. E07/N64 was statically rejected. None produced a feasible checkpoint, so these are final-epoch diagnostics rather than selected budget-compliant results.

| Variant | Family | Status | Accuracy | Macro-OvR AUC | Selection cost |
|---|---|---|---:|---:|---:|
| A00 | Architecture | 50/50 | 51.19% | 0.8261 | 9,474,139 |
| A01 | Architecture | 50/50 | 48.70% | 0.8038 | 10,800,700 |
| A02 | Architecture | 50/50 | 53.06% | 0.8218 | 9,181,152 |
| A03 | Architecture | 50/50 | 51.57% | 0.8180 | 9,161,276 |
| A06 | Architecture | 50/50 | 45.30% | 0.7764 | 7,207,292 |
| A07 | Architecture | 50/50 | 57.95% | 0.8619 | **4,634,372** |
| A08 | Architecture | 50/50 | 56.41% | 0.8470 | 6,491,283 |
| A09 | Architecture | 50/50 | 48.79% | 0.8095 | 9,183,816 |
| A10 | Architecture | 50/50 | 51.63% | 0.8145 | 9,250,253 |
| B01 | Attention | 50/50 | 56.03% | 0.8563 | 5,195,955 |
| B02 | Attention | 50/50 | 50.60% | 0.8193 | 9,219,798 |
| B03 | Attention | 50/50 | 54.59% | 0.8354 | 8,646,668 |
| B04 | Attention | 50/50 | 51.84% | 0.8172 | 9,211,753 |
| E02 | Engram, ungated | 50/50 | **66.60%** | **0.8997** | 4,720,386 |
| E03 | Engram, gated | 50/50 | 65.46% | 0.8979 | 4,966,751 |
| E04 | Engram, two-block gated | 50/50 | 63.91% | 0.8880 | 9,517,420 |
| E05 | Engram, four-bit memory | 50/50 | 66.52% | 0.8988 | 4,816,654 |
| E06 | Engram, 64-row table | 50/50 | 62.56% | 0.8825 | 4,927,819 |
| E07 | Engram, rank bigrams | Static rejection | — | — | Fixed memory cost 838,272 > 350k target |

The N64 Engram endpoints are the strongest observed values in this short screen, but every trained N64 case costs at least 4.63 million operations. The study has one seed, a 50-epoch high-learning-rate schedule and no feasible checkpoint. It supports further investigation, not a budget-compliant superiority claim.

## Every fresh N8 case

The screen normalizes equivalent source configurations into 19 distinct variants; aliases such as the identical reference controls are recorded in the [38-case index](../../code/constituent-study-20260922/index.json). Partial rows show their last committed epoch.

| Variant | Family | Status | Accuracy | Macro-OvR AUC | Selection cost |
|---|---|---|---:|---:|---:|
| A00 | Architecture | 50/50 | 62.17% | 0.8747 | 696,951 |
| A01 | Architecture | 40/50, crashed | 61.12% | 0.8667 | 699,566 |
| A02 | Architecture | 39/50, crashed | 59.78% | 0.8611 | 664,198 |
| A03 | Architecture | 50/50 | 62.16% | 0.8727 | 632,846 |
| A06 | Architecture | 50/50 | 60.54% | 0.8584 | **386,219** |
| A07 | Architecture | 50/50 | 61.12% | 0.8660 | 411,302 |
| A08 | Architecture | 38/50, crashed | 63.03% | 0.8748 | 629,863 |
| A09 | Architecture | 38/50, crashed | 63.20% | 0.8765 | 731,330 |
| A10 | Architecture | 37/50, crashed | 62.87% | 0.8752 | 619,706 |
| B01 | Attention | 33/50, crashed | 62.39% | 0.8730 | 619,286 |
| B02 | Attention | 50/50 | 62.61% | 0.8731 | 637,091 |
| B03 | Attention | 35/50, crashed | 60.96% | 0.8683 | 665,279 |
| B04 | Attention | 50/50 | **63.41%** | **0.8788** | 600,329 |
| E02 | Engram, ungated | 2/50 canary | 58.03% | 0.8468 | 629,746 |
| E03 | Engram, gated | 2/50 canary | 56.84% | 0.8420 | 658,860 |
| E04 | Engram, two-block gated | 50/50 | 61.69% | 0.8733 | 648,406 |
| E05 | Engram, four-bit memory | 50/50 | 62.37% | 0.8723 | 418,159 |
| E06 | Engram, 64-row table | 50/50 | 60.56% | 0.8659 | 430,891 |
| E07 | Engram, rank bigrams | 50/50 | 61.33% | 0.8660 | 491,803 |

No N8 row reached its configured budget. A06 came closest at 386,219 operations against 350k. Canary-only and partial rows must not be ranked as completed 50-epoch results.

## Original N16 Engram continuation

| Run | Model | Training | Latest accuracy | Latest AUC | Augmented cost | Outer result |
|---|---|---:|---:|---:|---:|---|
| E00 | Two-block reference | 1,000/1,000 | 33.27% | 0.6384 | 721,193 | Succeeded; no feasible checkpoint |
| E01 | One-block control | 1,000/1,000 | 33.44% | 0.6450 | 362,158 | Metric reproduction failed |
| E02 | Ungated memory | 1,000/1,000 | 54.55% | 0.8322 | 380,009 | Metric reproduction failed |
| E03 | Gated memory | 1,000/1,000 | 52.55% | 0.8203 | 440,525 | Metric reproduction failed |

These four runs use the original N16/1,000-epoch protocol. They are separate from the fresh N8/N64 screen, which changes constituent count, schedule, learning rate, run names and source identity.

## N8 EBOP project: 13 W&B records

| Substudy | Records | Outcome | Repository evidence |
|---|---:|---|---|
| Initial relative-budget pilot | 4 | Control, 75% and 25% completed; 50% failed before an epoch. The 75% arm met its target. | [Pilot results](../../results/post_conference/budget_pilot.json) |
| Resource-priority retry | 1 | Completed in eight epochs at 847,982 EBOPs; quoted validation AUC 0.7973 | [Retry result](../../results/post_conference/resource_priority.json) |
| Standalone 350k long run | 1 | Completed 1,000 epochs; selected epoch 1,000 at 344,430 EBOPs and validation AUC 0.8536 | [Standalone result](../../results/post_conference/standalone_long_budget.json) |
| Seven matched ablations | 7 | All completed 1,000 epochs; all displayed selected checkpoints fit 350k | [Ablation results](../../results/post_conference/ablation_metrics.json) |

## N64 coverage outside the fresh screen

The pre-conference baseline also contains **15 completed N64 runs**: FP32, W8A8, W1A8, W1A6 and W1A4, each with three seeds. Their held-out mean AUCs are 0.9486, 0.9448, 0.9121, 0.9136 and 0.9073 respectively. Those results use a different 101-epoch fixed-precision protocol and should not be pooled with the 50-epoch constrained screen.

## Source and provenance map

| Evidence | Location |
|---|---|
| All 86 September W&B records, IDs, states, groups and selected summary fields | [W&B registry snapshot](../../results/wandb-project-inventory-20260923.json) |
| Architecture/attention exact 27-run final snapshot | [A/B JSON](training-results-20260923.json) |
| Original Engram source and eight configs | [`code/engram`](../../code/engram/README.md) |
| Original Engram final four-run snapshot | [Engram JSON](../../results/engram/status-20260923.json) |
| Exact fresh N8/N64 runtime and configurations | [`code/constituent-study-20260922`](../../code/constituent-study-20260922/README.md) |
| Fresh screen source hashes and preflight | [Source manifest](../../results/constituent_study/source_manifest.json) · [Preflight](../../results/constituent_study/preflight.json) |
| Fresh screen all 37 trained rows plus static rejection | [Constituent result JSON](../../results/constituent_study/results-20260923.json) |
| Pre-conference 60-run held-out results | [Per-seed CSV](../../results/pre_conference/auc_per_seed.csv) |

Checkpoint binaries, training arrays and full W&B histories are not copied into GitHub. The repository contains the compact numerical records, configurations, exact source bundles, hashes, run IDs and limitations needed to identify what was executed.

[Return to current work](README.md).
