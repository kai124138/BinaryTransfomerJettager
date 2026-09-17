# R4 gradual-EBOP model: hardware synthesis

Status snapshot: **2026-09-17T17:58:17+00:00**. Vitis HLS is running; Vivado OOC follows automatically only if HLS succeeds. This page is updated separately from the remote job.

## Intent

Measure the hardware cost and timing of `ebops-n8-20260912-ablation-r4-gradual-w1a8-s1`, using its selected feasible checkpoint. This is the original trained model, separate from the planned frozen-backbone recovery studies. No retraining is required.

The model has eight constituents, three input features, d_model=32, two transformer layers, four heads, FFN=64, binary weights with learned layer scales, and learned activation widths. The A8 run name describes initialization; the export preserves the trained widths.

| Stage | Settings | Status |
|---|---|---|
| Model-to-export fidelity | Original selected checkpoint and fixed validation sample | Passed |
| Local C simulation | 4,096 jets | Bit-exact against export |
| Linux C simulation on Mulder | Fresh compilation, same 4,096 jets | Bit-exact against export |
| Vitis HLS 2023.2 | VU13P, 2.5 ns, RF=1, Latency, io_parallel | Running |
| Vivado OOC 2023.2 | xczu7ev, 2.5 ns, four threads | Queued after successful HLS |
| Place-and-route / bitstream | Outside this study | Not launched |

## What the results will establish

Read whole-model latency and initiation interval from the HLS report. **RF=1 alone does not establish whole-jet II=1.** Report Vivado LUT, FF, DSP and BRAM usage separately at post-synthesis and post-optimization, with pre-route timing and explicit device labels. Input standardization and final output classification are outside this synthesized logits-producing core.

Mulder's VU13P Vivado device-license probe failed; the xczu7ev probe passed. Consequently, Vivado will use the project's established **xczu7ev proxy flow**. Its resource and timing results cannot certify VU13P device fit or timing closure. HLS itself continues to target VU13P. No hardware measurements are available in this snapshot.

## Export corrections checked before launch

The legacy export path rebuilt projection activations at eight bits. This export copies the checkpoint's complete learned fixed-point grids and sizes internal carry grids to avoid additional truncation. Learned layer scales and biases use the existing 16-fractional-bit realization; the rebuilt export passes the model-fidelity gate but is not asserted identical to float32 training arithmetic.

Layer tracing also exposed a zero-grid conversion defect: ReLU followed by a saturating signed quantizer with levels {−1,0} must produce zero, but quantizer fusion emitted an unsigned type that could produce 0.5. A guarded signed-cast repair restores the original behavior. A 20-case boundary regression passes, and the repaired export matches C simulation on all 4,096 checked jets on both platforms. These are sample-based fidelity checks, not full held-out accuracy measurements.

## Resource discipline

One serialized synthesis chain; no GPU allocation. Guards stop its own subprocesses at 64 GiB process-group RSS, six hours for HLS, eight hours for Vivado, or 50 GiB work-directory growth. Temporary files use the isolated work directory. Raw reports, constraints, verification records and logs are retained before reporting results.

Next update: HLS resources, top-level latency/II, and the subsequent Vivado synthesis state. [Return to the work hub](README.md).
