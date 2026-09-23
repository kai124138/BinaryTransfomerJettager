# Engram-inspired jet memory: implementation and current results

**Snapshot: 23 September 2026, 15:53 PDT (22:53 UTC).** All four E00–E03 training loops reached 1,000 epochs. E00 passed the outer runner; E01–E03 failed the selected-checkpoint metric-reproduction check. **None recorded a checkpoint within the 350,000 augmented-cost target.**

[Exact final metrics](../../results/engram/status-20260923.json) · [Earlier failure evidence](../../results/engram/status-20260921.json) · [Frozen source and reproduction](../../code/engram/README.md) · [Runtime source manifest](../../results/engram/source_manifest.json)

## Latest recorded training measurements

These are trainer-recorded internal-validation values on 124,000 jets, not independently recomputed held-out test results. The memory variants include a custom arithmetic estimate, so their totals must not be treated as interchangeable with native-only EBOP counts from other studies.

| Run | Model | Completed epochs | Latest accuracy | Latest macro-OvR AUC | Augmented selection cost | Status |
|---|---|---:|---:|---:|---:|---|
| E00 | Two-block reference | 1,000 | 33.27% | 0.6384 | 721,193 | Outer runner succeeded; no feasible checkpoint |
| E01 | One-block control | 1,000 | 33.44% | 0.6450 | 362,158 | Final validation failed |
| E02 | One block + ungated memory | 1,000 | 54.55% | 0.8322 | 380,009 | Final validation failed |
| E03 | One block + gated memory | 1,000 | 52.55% | 0.8203 | 440,525 | Final validation failed |

E02 and E03 retain useful classification signals, but both exceed budget and lack a successfully validated final result. The gate comparison changes parameter count, scaling, rounding and cost allocation together; it does not isolate the effect of dynamic gating. No statistical superiority claim is made from this one-seed study.

## Method and experiment scope

The model is an Engram-inspired adaptation for complete jets, not a reproduction of a language model or an autoregressive key–value cache. Each input contains 16 constituents with standardized `pt`, `etarel`, and `phirel`. Bin boundaries are fitted to a training-only calibration sample. Eight bins per feature give an exact 512-row address for each constituent; this tuple mode introduces no hash collisions. Padding is masked.

E02 retrieves a trainable 32-dimensional value vector and adds it after the first attention residual. E03 also retrieves a key and quantizes a context gate based on the current hidden state:

```text
q = fixed_point_quantize(hidden)
g = quantize_unsigned(clip(0.5 + dot(q, key) / (4 * 32), 0, 1))
output = hidden + quantize(g * value)
```

The implementation applies explicit quantization to keys, values, queries, reductions and the residual; see [JetEngram](../../code/engram/bnhgq2/engram.py). Values initialize to zero so adding memory preserves the initial backbone outputs. Table updates use a 5× learning-rate multiplier and no table weight decay. Backbone and memory train jointly. The memory is multibit even though the backbone projection weights are binary.

The initial E00–E03 pilot paused at 100 epochs on a 1,000-epoch schedule. On September 20 all four were continued under the same configurations and checkpoint identities; all training loops are now complete. E04–E07 remain unrun under this original N16 continuation. Adapted versions were separately tested under the 50-epoch [N8/N64 exploratory screen](CONSTITUENT_SCREEN_20260923.md), whose protocol and run identities are distinct.

## Cost and storage contract

| Variant | Fixed custom arithmetic estimate | Logical table storage | Replicated storage scenario |
|---|---:|---:|---:|
| E02 | 17,920 bit operations | 16 KiB | 128 KiB |
| E03 | 78,496 bit operations | 32 KiB | 256 KiB |

Selection cost is native HGQ2 backbone EBOPs plus the fixed custom estimate. The latter contributes no activation-width gradient, so the existing backbone regularizer must make room for it. The shared caps are 64 KiB logical table storage and 512 KiB in the stated replication scenario. That scenario assumes eight copies of a two-read-port memory to serve 16 arbitrary addresses per jet at whole-jet II=1. It is not a measured FPGA resource requirement. Quantizer/control/wiring costs and physical memory rounding are not fully priced. No HLS lowering, timing closure or hardware II guarantee exists for the module.

## Finalization results and stale artifacts

E00 passed the outer `run_engram.py` finalization. E01–E03 fail its check comparing reloaded selected-checkpoint AUC/accuracy with recorded metrics at `atol=1e-7`, `rtol=0`. The [machine-readable trace excerpts](../../results/engram/status-20260921.json) retain the full-precision observed mismatches. Their cause is unresolved; this publication preserves the failing assertion and does not relax its tolerance.

The inner trainer writes `COMPLETE.json` before that check. Its presence therefore establishes completion of the training loop, not successful final validation. E02/E03 also retain `engram_result.json` and predictions/tables from the old **100-epoch screen**. Those files were not overwritten after the failed 1,000-epoch validation and must not be reported as final artifacts. They are not distributed here as current results.

## Gate diagnostics

At E03’s final training epoch, 96.38% of gate values in the fixed training probe equal 0.5; the gate mean is 0.5045 and standard deviation 0.0233. These describe a training probe, not all validation jets. Some variation is observed, but this does not establish a useful context-dependent gating benefit. The full probe fields remain in the status JSON.

## What is published and what remains open

The source under `code/engram` matches all 22 Python files in the production runtime manifest. It is frozen separately because the production experiment used a different trainer revision from the historical public `code/hgq2` package. Seed-1 configurations E00–E07, synthetic checks, dependency pins and the shared-cache preparation helper accompany it. Checkpoint binaries, event arrays and stale screening predictions are not included. Publishing these records does not change W&B project permissions.

Resolve selected-checkpoint reproduction, evaluate all four arms at matched training prefixes, and inspect per-layer cost before claiming a constrained accuracy benefit. Additional seeds and verified held-out evaluation are required for broader conclusions. Hardware implementation is a separate task.

[Return to current work](README.md).
