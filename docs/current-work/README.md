# Current work: results and pending gates

Updated 1 October 2026 from saved scientific records, the reviewed cluster capture at `05:13:12.975881Z` and bounded code-compatibility checks. No scientific gate was cleared.

[Training rundown and next steps](RESULTS_AND_NEXT_STEPS_20261001.md) · [Machine-readable status](results-status-20261001.json) · [Project results](../../README.md)

| Campaign | Saved outcome | Next dependency |
|---|---|---|
| pT weighting | All 24 runs completed; weighting reduced integrated held-out AUC across eight paired seeds | Decide whether a separate pT-dependence objective warrants further study |
| Architecture and attention | All 27 training loops completed; five architecture runs feasible, no attention run feasible | Complete matched seed confirmations and verification |
| Twelve-run confirmation | Job failure observed, with transition time 26 September 07:24:11Z; no final scientific result saved | Recover arm logs and checkpoints; diagnose before any authorized restart |
| Engram and constituent screen | Failed metric reproduction in the original memory arms; partial exploratory screen | Resolve evaluation and feasibility before claiming a gain |
| Chang | b3 readout complete; K1 fired; b5 Job completed 29 September 20:52:06Z, with arm outcomes/readout still missing | Full b5 readout and the pending control-method decision |
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
