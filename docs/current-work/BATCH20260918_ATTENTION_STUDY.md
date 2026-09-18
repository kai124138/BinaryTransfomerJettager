# Fifteen-run follow-up: attention and compression timing

**Campaign `batch20260918` · submitted 2026-09-18T12:14:11Z**

The study tests whether changes to attention design, probability precision, or compression timing improve five-class jet tagging at a final native cost of 350,000 EBOPs. It trains the complete binary-weight transformer, using five arms and three matched initialization seeds per arm. [Current results and startup status](README.md) · [Machine-readable status](batch20260918-status.json) · [Config index](../../code/hgq2/configs/batch20260918/index.json).

## Fixed settings and comparisons

The reference is A04 from the original architecture screen: N=16 constituents, D=32 embedding, F=32 feed-forward width, L=2 blocks, H=4 heads, learned positional encoding, global average pooling, no normalization, and channel-wise learned activation widths initialized at 8 bits. Projection weights use `binary_absmean`; activation widths remain learned rather than fixed at 8 bits. Features are `pt`, `etarel`, `phirel`, with class order g, q, W, Z, t. The reference is selected for a controlled N=16 comparison, not because it won the earlier screen.

| Arm | Exact override from the reference | Hypothesis |
|---|---|---|
| B00 | None | Measure reference variation across fresh seeds |
| B01 | `arch.n_heads = 1` | Test single-head attention at the same embedding width; head dimension changes from 8 to 32 |
| B02 | `arch.pos_enc = "none"` | Test the contribution of the explicit learned constituent-position table |
| B03 | `quant.softmax_out_bits = 8` (was 10); `softmax_out_i = 1` unchanged | Test probability precision independently of the remaining attention internals |
| B04 | `experiment.target_schedule = [[0,525000],[100,420000],[200,350000]]` | Test relaxed initial compression followed by the same final resource limit |

B02 does not establish permutation invariance: removing one positional table does not prove that all other parameters, preprocessing, or quantization are position-independent. B01 also changes head dimension and attention layout; it is not a reduction of every attention operation by a factor of four.

Every arm uses initialization seeds **4, 5, 6**; split seed **1** and order seed **20260912** remain unchanged. Training/validation membership remains 496,000 / 124,000 events, with the existing N=16 preparation and train-only standardization. The 260,000-event test set is excluded from run selection and has been inspected in earlier studies; it is not a fresh untouched benchmark.

Adam uses learning rate **2e-5**, beta1=0.9, beta2=0.98, weight decay 0.01, clip-by-value 1, and batch 256 (validation batch 1024). The fixed horizon is **1,000 epochs**, one warm-up epoch and 999 decay epochs with polynomial power 1. PID settings remain P=1, I=0.05, D=0, warm-up 10, initial beta 1e-7, bounds [1e-10,1e-3]. Only B04 changes the budget schedule.

## Selection and compute allocation

All 15 runs pause at cumulative epoch **400** while preserving model, optimizer, controller and resume state. Do not shorten the configured schedule or restart the learning rate at promotion. B04 changes targets at zero-based epochs 100 and 200 and trains its last 200 screening epochs at 350k.

Select the highest validation categorical accuracy among checkpoints whose native EBOP count is **≤350,000**, even during B04's relaxed phase. AUC, lower cost, and earlier epoch break ties. Missing feasible checkpoints remain explicitly unavailable. Report each seed and aggregate variation; any superiority claim needs an uncertainty analysis. No gain is assumed from the hypotheses.

The planned continuation retains **B00 and two selected variant arms**, each with all three seeds, to epoch 1,000. Screening costs 15×400 = **6,000 epoch passes**; that continuation adds 9×600 = **5,400**, totaling **11,400**. These are epoch passes, not GPU-hours. Record measured training time before estimating remaining compute. This new campaign is separate from the original protocol's conditional B-series and should not be double-counted with overlapping future refinements.

## Launch and implementation evidence

The cluster confirms submission of `kai-batch0918-screen-e400` with parallelism **15**. The timestamped worker counts are in the [status record](batch20260918-status.json); submission and Running/Ready status do not prove a completed training epoch. The campaign uses the existing W&B project `BNJetTag-Batch20260917` and the new group **`batch20260918`**.

All 15 configurations passed local synthetic build, finite-gradient-step, and save/reload checks. Arm-effect checks included the missing positional table in B02 and B04's budget transitions. Resume/config/code guard checks passed on selected B02/B04 cases; uninterrupted-versus-resumed equivalence was tested for B02. These are software checks, not real-data training or FPGA results.

The immutable submitted bundle implements optional positional encoding and campaign indexing. The configs below record that submitted training campaign; the repository's older training/export source may not yet reproduce the new bundle without those changes. **B02 currently needs exporter updates before hardware evaluation** because existing conversion/extraction code assumes a positional layer. Other new checkpoints also require numerical export validation and hardware measurement. Native EBOP compliance alone cannot certify whole-jet II=1, resource fit, or timing closure.

## Exact configurations

| Arm | Configurations |
|---|---|
| B00 | [seed 4](../../code/hgq2/configs/batch20260918/batch20260918-b00-s4.json) · [seed 5](../../code/hgq2/configs/batch20260918/batch20260918-b00-s5.json) · [seed 6](../../code/hgq2/configs/batch20260918/batch20260918-b00-s6.json) |
| B01 | [seed 4](../../code/hgq2/configs/batch20260918/batch20260918-b01-s4.json) · [seed 5](../../code/hgq2/configs/batch20260918/batch20260918-b01-s5.json) · [seed 6](../../code/hgq2/configs/batch20260918/batch20260918-b01-s6.json) |
| B02 | [seed 4](../../code/hgq2/configs/batch20260918/batch20260918-b02-s4.json) · [seed 5](../../code/hgq2/configs/batch20260918/batch20260918-b02-s5.json) · [seed 6](../../code/hgq2/configs/batch20260918/batch20260918-b02-s6.json) |
| B03 | [seed 4](../../code/hgq2/configs/batch20260918/batch20260918-b03-s4.json) · [seed 5](../../code/hgq2/configs/batch20260918/batch20260918-b03-s5.json) · [seed 6](../../code/hgq2/configs/batch20260918/batch20260918-b03-s6.json) |
| B04 | [seed 4](../../code/hgq2/configs/batch20260918/batch20260918-b04-s4.json) · [seed 5](../../code/hgq2/configs/batch20260918/batch20260918-b04-s5.json) · [seed 6](../../code/hgq2/configs/batch20260918/batch20260918-b04-s6.json) |

[Return to current work](README.md).
