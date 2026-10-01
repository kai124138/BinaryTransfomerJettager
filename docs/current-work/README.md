# Current work: results and pending gates

Updated 1 October 2026 from saved scientific records, reviewed cluster observations, the verified 253-file storage recovery and bounded code-compatibility checks. Option (c) was selected for Chang; production gates remain pending.

[Training rundown and next steps](RESULTS_AND_NEXT_STEPS_20261001.md) · [Machine-readable status](results-status-20261001.json) · [Project results](../../README.md)

| Campaign | Saved outcome | Next dependency |
|---|---|---|
| pT weighting | All 24 runs completed; weighting reduced integrated held-out AUC across eight paired seeds | Decide whether a separate pT-dependence objective warrants further study |
| Architecture and attention | All 27 training loops completed; five architecture runs feasible, no attention run feasible | Complete matched seed confirmations and verification |
| Twelve-run confirmation | Job failed; all twelve current checkpoint generations recovered at 280–318 of 1,000 epochs | Resolve reload failures and verify resumability; exact final queue trigger remains uncertain |
| Engram and constituent screen | Failed metric reproduction in the original memory arms; partial exploratory screen | Resolve evaluation and feasibility before claiming a gain |
| Chang | b3 readout complete; K1 fired; all five b5 epoch-500 snapshots recovered; option (c), traced-cost PID, selected | Missing b5 certification/entropy and controller history; amendment, new freeze and replacement pilot |
| Delta | Canary evidence only; A07 packing exceeded memory at three arms | Chang gate, remaining Delta checks and revised packing |
| GPU benchmark | Available runs complete; provisional throughput comparison | Current schedulable capacity and selected-product canary |
| [Code compatibility](https://github.com/kai124138/BinaryTransfomerJettager/blob/e56fd7361f0299c1721aa053506d1eabc092cdba/docs/current-work/CODE_COMPATIBILITY_20261001.md) | Bounded engineering PASS, 217 comparisons; draft PR #1, not merged | Historical metric and calibrated-width checks; attribution of 74 checkpoint paths |
| Current constrained-model hardware | No successful synthesis result established in this update | Conversion, fidelity checks and synthesis for a verified candidate |

The [dated report](RESULTS_AND_NEXT_STEPS_20261001.md) gives evaluation splits, seed counts, intervals, evidence dates and the dependency-ordered work list. Training completion, checkpoint feasibility, checkpoint reproduction and held-out verification are separate statuses.

## Earlier records

- [24 September confirmation plan and evaluation](CONFIRMATION_RUNS_20260924.md) and [status snapshot](confirmation-status-20260924.json). Their execution status is historical.
- [23 September completed architecture/attention results](TRAINING_RESULTS_20260923.md) and [exact recorded metrics](training-results-20260923.json).
- [Original Engram report](ENGRAM_STUDY.md), [matched constituent screen](CONSTITUENT_SCREEN_20260923.md), and [experiment index](EXPERIMENT_INDEX_20260923.md).

Internal validation selects checkpoints; the held-out archive evaluates fixed selections. Cost targets, constituent counts and input sets remain explicit in comparisons. eBOPs are a computational proxy and do not establish FPGA fit or latency.
