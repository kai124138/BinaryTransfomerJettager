# Family Q: activation widths and EBOPs control

Method cards for the activation quantizers and the controller that drives EBOPs to the target.
With binary weights, activation widths are the only thing the EBOPs penalty can change, so this
family and the static floors in [rules.md](rules.md#static-floors) are closely tied. Each card
ends with the Delta entry it became, the wave-1 measurement it turned into, or the reason it was
parked. Code references are to [`code/hgq2/bnhgq2/`](../../code/hgq2/bnhgq2/) in this repository
and to the HGQ2 package (hgq2 0.1.9). Sources were read on 26 and 27 September 2026.

## What the code already does

[`qat.py`](../../code/hgq2/bnhgq2/qat.py) has three activation-width policies behind
`quant.act_calib`: `frozen` (a static grid calibrated on the first batch), `trainable` (a fixed
total width with a trainable integer/fractional split) and `free` (fully learnable integer and
fractional bits under a monotone L1 penalty). Granularity is `tensor` (one grid per tensor) or
`channel`. [`ebops_target.py`](../../code/hgq2/bnhgq2/ebops_target.py) turns a target into a fixed
threshold after calibration, recomputes EBOPs every epoch with HGQ2's own trace, logs whether the
budget is met, and keeps the best feasible checkpoint. The closed-loop controller is HGQ2's
`BetaPID`; an open-loop, piecewise `beta_schedule` also exists
([`train.py`](../../code/hgq2/bnhgq2/train.py)), and the two are mutually exclusive. The
constrained runner ([`ablation.py`](../../code/hgq2/bnhgq2/ablation.py)) also reads a per-epoch
`experiment.target_schedule` and an `experiment.recovery_after_epochs` freeze.

HGQ2's `QSoftmax` prices its own internals in EBOPs: the max subtraction, the row accumulation and
the exp × inverse product are summed into the layer's cost, and the attention-weighted sum is an
ordinary traced product. A quantizer's reported width is relu(i + f), so a channel can reach
0 bits once i + f goes negative. The reference quantizer set uses this: per-channel learned widths
that wrap on overflow and can reach 0 bits, a learned softmax output, and trainable exp and
inverse tables with a 4-bit minimum.

### Q01: per-channel activation widths

**Mechanism.** One trainable (i, f) per channel instead of one per tensor, with the L1 penalty
normalized so that the pressure per channel matches the per-tensor case. **Why here.** With ±1
weights the activation grid is the only place precision lives, and channels that feed a binary
matmul carry very different ranges. **Source.** HGQ2 treats per-element and per-channel widths as
its default mode (Sun, "Tutorial on HGQ and Alkaid", FastML 2026). **Hazards.** EBOPs traces the
widths correctly; downstream, per-channel types can turn a shared adder tree into channel-specific
logic, more LUTs at equal EBOPs. **Earlier attempt.** Channel granularity did better than tensor
granularity in the constrained N = 8 ablation ([project README](../../README.md#post-conference-ebop-constrained-training)).
**Delta:** per-channel widths are part of the reference configuration; the tensor-wide variant is
parked.

### Q02: per-value (fully heterogeneous) activation widths

**Mechanism.** One (i, f) per value of the tensor: per position and per channel. **Why here.**
Constituents are pT-sorted, so activations differ by particle rank, and a per-position width could
follow that structure. **Source.** Laatu et al., "Sub-microsecond Transformers for Jet Tagging on
FPGAs", arXiv:2510.24784, section 2.2: "Value-wise heterogeneous quantization is also applied in
the datapath, which breaks the otherwise permutation-invariant model to further reduce the firmware
footprint." HGQ-LUT (Sun et al., arXiv:2604.22293) uses per-element widths with zero-bit pruning on
a different layer type. Neither isolates the effect. **Change.** A new granularity value,
`element`, about 10 to 20 lines; shapes are static at fixed N. **Hazards.** The number of width
parameters grows by T (64) or T² (4,096) per site, so the penalty per parameter may need
retuning; on a T × T score tensor, per-element widths can inflate HLS resources regardless of
EBOPs, a case where EBOPs and LUT diverge. The model is no longer permutation invariant.
**Prediction.** Up at 350k, where bits are placed by particle rank; flat at 5M. **Delta entries:**
M011, factor A of FF2, and the package M065.

### Q03: activation width initialization

**Mechanism.** The reference initializes the fractional bits f and lets the integer bits track the
data range; only f feels the EBOPs gradient. This card starts from a wider fixed f0. Three things
could differ between two initializations: the total starting width, whether the integer split is
calibrated on data first, and the overflow mode (wrap or saturate). A usable arm changes one at a
time. **Why here.** A wider start means the optimizer travels further to reach a given target;
under wrap-around overflow, a start that is too narrow aliases badly until i grows. **Source.** No
published number isolates the initialization; Laatu et al. report only whole-recipe endpoints.
**Change.** A fixed f0 key if the reference quantizer code exposes it, otherwise a patch of about
20 lines. **Prediction.** Slower EBOPs descent; selected accuracy within the interval. **Delta
entries:** M012 (f0 set before launch) and factor B of FF2. A lower f0 is parked until M012 is
read.

### Q04: controller variants (PID gains, open-loop β)

**Mechanism.** Two controllers exist: `BetaPID`, closed loop against a fixed target, and a
piecewise open-loop β schedule whose checkpoints are admitted to a Pareto front only when they meet
a fixed threshold. This card retunes the PID gains, or runs the open-loop schedule at the same
threshold. **Why here.** With binary weights the controller has one lever (activation widths), so
it may oscillate or converge slowly, and gains or schedule shape may matter more than with HGQ
weights. **Source.** PID control of a regularization weight and piecewise schedules are standard
patterns; the HGQ2 tutorial presents the target-via-β pattern. No paper isolates the choice for
binary weights. **Hazards.** The two paths fail differently: the PID path reports "no feasible
checkpoint" when the target is never met, while the open-loop path leaves an empty front, which is
easy to misread. An open-loop run must check for that explicitly. **Prediction.** Gentler gains
put more seeds under the target by epoch 500 without moving the selected accuracy much; an open
loop leaves fewer seeds feasible than the PID at matched epochs. **Delta entries:** M013 (p 2,
i 0.2, 350k only) and M014 (open-loop β, schedule set before launch, confirm-only because it spans
the full 7,000 epochs).

### Q05: a moving EBOPs target

**Mechanism.** Ramp the target itself down over training instead of scheduling β against a fixed
target; the constrained runner already reads a per-epoch `target_schedule`. **Hazards.** Checkpoint
selection must key off the final target only; a checkpoint feasible at an intermediate, looser
target must never count. **Status:** parked. Gradual budget schedules were tried in September: at N = 8 they gave no
clear gain over a fixed target ([project README](../../README.md#post-conference-ebop-constrained-training)),
and at N = 16 no run reached 350k ([attention study results](../current-work/TRAINING_RESULTS_20260923.md)).
It returns only if M014 shows that early pressure does harm.

### Q06: cost-first selection

**Mechanism.** Select the lowest-EBOPs feasible checkpoint instead of the most accurate one.
**Why here.** It would show how far the selection rule moves the pick, and whether a Pareto front
is a better report than one number. **Status:** converted into a measurement (Z05 in
[rules.md](rules.md#wave-1-measurements-before-any-gpu-time)): re-selection from the logs of
existing runs, with no training. The rule of record stays: highest validation accuracy among
feasible checkpoints, and "no feasible checkpoint" when there is none.

### Q07: an EBOPs ladder

**Mechanism.** Run the same configuration at several targets and read accuracy against cost as
one curve. **Why here.** A single iso-EBOPs point cannot distinguish "binary is worse at this
budget" from "binary has a different knee". Laatu et al. report only the 350,000 target and no
accuracy-against-EBOPs curve. **Hazards.** Lower targets may leave fewer seeds feasible, so every
point reports its feasible-seed count, never a mean that silently drops infeasible seeds.
**Delta entry:** M010, a rung at 1.4M between d32h4 at 350k and at 5M. A 700k rung is parked
until the first rung shows curvature.

### Q08: softmax table and output widths

**Mechanism.** Under the earlier quantizer set, the softmax output was fixed at 10 bits and the exp
and inverse tables were static. Under the reference set they are learned, with the tables held at
a 4-bit minimum. This card lowers that minimum to 2 bits. **Why here.** At 0-bit activations the
softmax tables are the whole static floor of an attention block
([rules.md](rules.md#static-floors)), so the table minimum decides how much of a 350k budget
attention consumes before any channel is alive. **Source.** HGQ2's `QSoftmax` cost accounting;
no published number. **Hazards.** EBOPs traces this knob fully. The softmax tables are lookup
tables, not division circuits, so the resource risk of widening them is table depth (BRAM or LUT),
not DSPs. The same accumulator gap as elsewhere applies to the row sum. **Prediction.** Lowers the
d32h4 0-bit floor from 343,040 to 114,176 (computed from layer shapes); accuracy at 5M flat.
**Delta entry:** M003, and the floor crosses M051, M052, M054 and M060. A fixed 4-bit softmax
output is parked for the case where the reference reverts to the earlier quantizer set.

### Q09: table-based nonlinearities

**Mechanism.** HGQ2's `QAffinedUnaryFunctionLUT('tanh')` is a trained affine-plus-table
approximation of tanh, synthesized as a lookup. Placed before attention and before the FFN of a
model with no normalization layers, it adds a cheap bound on the activation range. HGQ-LUT
generalizes the idea to whole neurons built as trained logic LUTs with a LUT-count surrogate in
place of EBOPs. **Why here.** The reference has no normalization at all, so the question is whether
a cheap table bound recovers some of the range control a norm layer would give. HGQ-LUT is also a
threat to the framing: its paper reaches 0 DSP without binary weights, so if its LUT count at
comparable accuracy undercuts ours, binary weights are not the only route to 0 DSP. **Source.** Sun
et al., HGQ-LUT, arXiv:2604.22293, https://arxiv.org/abs/2604.22293 (reports 0 DSP with fewer LUTs
than HGQ at matched accuracy on a 16-feature jet task; the tables were read from the HTML version
and still need a check against the PDF, open question OQ-10). **Hazards.** A table layer may not be
billed correctly by the EBOPs trace, so any model containing one reports the table cost as its own
quantity; no export path exists yet. **Prediction.** A small accuracy gain from range control on a
norm-free graph. **Delta entries:** M016 and factor C of FF2. Tanh as the FFN activation (A14) is
merged here; ReLU is free in hls4ml.

### Q10: softmax-free or cheaper attention

**Mechanism.** Replace the softmax with ReLU or sigmoid attention, a base-2 exponent, or a fixed
normalization. **Why here.** The softmax's inverse is the one division-like operation in an
otherwise add-and-multiply network, and under the reference quantizers its tables are the whole
static floor. ReLU(scores)/N removes both; at N = 64 the division is a shift. **Sources.**
Wortsman, Lee, Gilmer and Kornblith, "Replacing softmax with ReLU in Vision Transformers",
arXiv:2309.08586, https://arxiv.org/abs/2309.08586; Ramapuram et al., "Theory, Analysis, and Best
Practices for Sigmoid Self-Attention", arXiv:2409.04431, https://arxiv.org/abs/2409.04431;
Stevens, Venkatesan, Dai, Khailany and Raghunathan, "Softermax: Hardware/Software Co-Design of an
Efficient Softmax for Transformers", DAC 2021, arXiv:2103.09301, https://arxiv.org/abs/2103.09301.
The first two report accuracy comparable to softmax at ViT and ImageNet scale; no number was read
at table level (open question OQ-12), and none targets hls4ml or binary weights. **Change.** A new
attention branch with re-derived quantizer ranges, since a ReLU attention row no longer sums to 1,
about 80 to 150 lines. **Prediction.** Feasible at 350k; accuracy at 5M below the softmax base,
because competitive normalization does real work when constituents compete for attention mass.
**Delta entries:** M005 and M059 (with a Q/K width floor). Sigmoid attention is parked: same
mechanism, and it needs a table.

### Q11: input width matched to L1 PUPPI candidates

**Mechanism.** Pin the input quantizer to the precision at which L1 PUPPI candidates arrive from
the Correlator Layer 2, instead of letting it float under the EBOPs penalty. **Status:** parked;
the per-field bit width has not been found in a primary source. CMS, "Reconstructing jets in the
Phase-2 upgrade of the CMS Level-1 Trigger with a seeded cone algorithm", arXiv:2310.08062,
https://arxiv.org/abs/2310.08062, gives link bandwidths and the 128-particle truncation but no
per-field width; Odagiu et al., arXiv:2402.01876, give the three-feature convention but no width;
WOMBAT (arXiv:2505.05532, https://arxiv.org/abs/2505.05532) does not state a deployment precision
either. The likely sources are the CMS Phase-2 L1 trigger TDR (CERN-LHCC-2020-004) and the
Correlator Layer-2 firmware data formats (open question OQ-09). Kreis et al., arXiv:1808.02094,
was located but not read.

### Q12: channel pruning through 0-bit widths

**Mechanism.** Under the reference quantizers a channel whose i + f falls below zero reports 0
bits, so structured pruning is reachable without any change. **Why here.** For binary weights,
pruning a channel removes a whole column of ±1 weights and their adders, the one size lever this
design has besides width. **Hazards.** EBOPs shows the saving; whether hls4ml elides a 0-bit
datalane or synthesizes a 0-width wire is not known (open question OQ-13), so EBOPs may show a
saving that synthesis does not realize. **Status:** converted into a measurement (Z02): the
fraction of 0-bit channels per site at the selected checkpoint, from logged widths.

### Q13: accumulator-aware quantization (A2Q, A2Q+)

**Mechanism.** A2Q bounds the L1 norm of a layer's weights to a target accumulator width during
training, which provably avoids overflow. **Sources.** Colbert et al., "A2Q:
Accumulator-Aware Quantization with Guaranteed Overflow Avoidance", arXiv:2308.13504,
https://arxiv.org/abs/2308.13504 (extending arXiv:2301.13376); Colbert et al., "A2Q+: Improving
Accumulator-Aware Weight Quantization", arXiv:2401.10432, https://arxiv.org/abs/2401.10432. Only
abstracts were read, so no number is quoted (open question OQ-11). **Why it does not port.** After
absmean binarization every effective weight is ±β, so there is no weight magnitude to constrain.
For ±1 weights the accumulator width needed to sum n products of a b-bit activation without
overflow is exactly b_act + ⌈log₂ n⌉, with no training change. **Status:** A2Q training is
rejected; the closed-form accumulator width became a second cost column (Z03), reported beside
native EBOPs and never in place of it.

### Q14: fixed-width recovery

**Mechanism.** Two phases: learn widths under EBOPs pressure, then freeze every activation
quantizer's (i, f) and keep training the weights with no further width search. **Why here.** The
width search and the binarization are two noisy, non-stationary signals at once; separating "find
the widths" from "fit the weights to them" is a standard quantization-aware pattern applied at the
end rather than the start. **Source.** No single paper; the calibrate-then-fit practice of early
quantization-aware training (for example BinaryConnect, arXiv:1511.00363). **Hazards.** It
lengthens training. **Earlier attempt.** The fixed-width recovery arm of the constrained N = 8
ablation froze its widths at epoch 800 but selected its checkpoint at epoch 205, so the treatment
was never measured ([record](../../results/post_conference/ablation_metrics.json),
[config](../../code/hgq2/configs/post_conference_budget350k-fixed_width_recovery-w1a8.json)). **Prediction.**
Accuracy up after the freeze at unchanged EBOPs. **Delta entry:** M015, screened at H = 1,000 with
the freeze at epoch 500, and read only if the selected epoch comes after the freeze.

## Considered and left out

Per-value widths on the weight side would make the weights multi-bit, which belongs to the
labelled baselines, not this family. Package cards (for example Q01 with Q04 and Q07) are left to
the catalogue, where combinations are pre-registered with their decomposition ladders.
