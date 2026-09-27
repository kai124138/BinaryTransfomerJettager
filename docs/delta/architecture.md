# Family A: architecture

Method cards for the architecture of the binary transformer. Each card is a change to d32h4 (d_model
32, 4 heads, one block, FFN 32, learned positional encoding, no normalization layers, global
average pooling, ReLU FFN, binary weights, N = 64) unless it says otherwise, and ends with the
Delta entry it became or the reason it was parked. Numbers from Laatu et al. are from their
Table 1: one model per row, no seed interval, trained to a 350,000-EBOP target with HGQ weights,
selection rule not stated. They are their numbers on their metric, never ours. Sources were read
on 26 and 27 September 2026.

## What the code allows

The training graph is built in [`qat.py`](../../code/hgq2/bnhgq2/qat.py) and the export graph in
[`build.py`](../../code/hgq2/bnhgq2/build.py). The config accepts `arch.{n_part, n_feat, d_model,
n_heads, n_layers, ffn_dim, n_classes, pool, ffn_act, norm, norm_placement, pos_enc, softmax_free,
input_std}`. The export builder accepts only `norm` ∈ {subln, none}, so any other normalization is
new code on both sides. The positional table is dropped on export and folded into the input
projection's bias, so `pos_enc none` is config-only on both sides. Pooling has one code path,
global average pooling. The only multiplier-bearing operations in the export graph are the two
activation × activation contractions inside attention (the scores and the attention-weighted sum);
every weight-side multiply stays ±1 or a two-signed-digit constant, and a comment in `build.py`
records that a general-valued scale left in the weights inferred DSPs. Any card that adds another
activation × activation product (attention pooling, a gated FFN, pairwise features) is scored
against that risk.

### A01: width and heads at fixed N = 64

**Mechanism.** Vary d_model over {16, 24, 32, 48} and the head count over divisors of d_model,
with one block and everything else fixed. **Why here.** Binary weights carry less information per
weight, and BitNet's scaling argument recovers capacity with width. At a fixed budget, width and
head count trade against each other, and at 350k the head count sets the softmax floor.
**Source.** Wang et al., BitNet, arXiv:2310.11453. No published number for this ladder.
**Hazards.** Wider d_model grows the two activation × activation contractions linearly; narrower
d_model may starve the head. **Prediction.** Accuracy is not monotone in width at a fixed target.
**Delta entries:** M001 (two heads), M002 (one head) and M042 (d_model 16); d24h2 itself is the
350k base. d_model 48 is parked until the first rung shows curvature.

### A02: depth (two blocks), and weight sharing across blocks

**Mechanism.** Two blocks instead of one, either independent or with the second block's weights
tied to the first (cross-layer sharing). **Source.** Lan et al., "ALBERT: A Lite BERT",
arXiv:1909.11942, https://arxiv.org/abs/1909.11942 (from a summary, not a table: sharing attention
parameters costs little, sharing FFN parameters costs more; BERT scale). **Hazards.** Two blocks
roughly double the attention and FFN EBOPs; the 0-bit floor of d32h4 with two blocks is 686,080
(computed from layer shapes), above 350k. Tying weights reduces stored parameters but not EBOPs,
because activations still pass through twice. **Prediction.** Up at 5M if depth matters.
**Delta entry:** M044, at 5M only. Weight tying is rejected: no EBOPs saving.

### A03: FFN width

**Mechanism.** FFN width in {16, 32, 64} at d_model 32 (the reference ratio is 1; standard
transformers use 4). **Why here.** The FFN is two binary layers, and a wider one may recover
per-weight capacity at a cost linear in the width. **Source.** No binary-specific paper; the
ratio of 4 comes from the original transformer and is only a comparison point. **Earlier attempt.**
In the constrained N = 8 ablation, FFN 32 did better than FFN 64 at 350k on one seed
([project README](../../README.md#post-conference-ebop-constrained-training)). **Prediction.** Worse
at 350k, flat or up at 5M. **Delta entry:** M043 (FFN 64). FFN 16 is parked.

### A04: positional encoding, learned or none

**Mechanism.** Drop the learned additive positional table, in training and export. **Why here.**
Constituents form an unordered set, so a per-position term makes the output depend on an ordering
that is not a property of the jet; removing it restores exact permutation invariance. **Sources.**
Komiske, Metodiev and Thaler, "Energy Flow Networks: Deep Sets for Particle Jets",
arXiv:1810.05165, https://arxiv.org/abs/1810.05165; Qu and Gouskos, "ParticleNet",
arXiv:1902.08570, https://arxiv.org/abs/1902.08570. **Hazards.** None for hardware. If removing
the table hurts, the model is using position as a proxy for the pT ordering already in the data,
which is a different claim from "attention needs position". Without it, gated tokens become exact
duplicates. **Earlier attempt.** Removing the positional table at N = 16 was inconclusive, with a
large seed spread ([attention study](../current-work/TRAINING_RESULTS_20260923.md)). **Delta entry:**
M046 at 5M, and the no-PE cells of the gate × PE × mask cube (M075, M077, M078). At 350k the base
d24h2 already has no positional table.

### A05: normalization (none, RMSNorm, SubLN)

**Mechanism.** Four points: no normalization (the reference); the frozen two-signed-digit affine
that already restores each β; RMSNorm, trainable and without mean-centering; and SubLN, the code's
other supported value. **Why here.** A binary layer cannot fold a multiplicative normalization
scale into its ±1 kernel, so the scale needs its own affine, which is what the frozen affine
already is; the real comparison is a trainable normalization against the frozen one. **Sources.**
Zhang and Sennrich, "Root Mean Square Layer Normalization", arXiv:1910.07467,
https://arxiv.org/abs/1910.07467 (full-precision numbers only); SubLN from Wang et al.,
arXiv:2310.11453, as implemented in [`subln.py`](../../code/hgq2/bnhgq2/subln.py). **Hazards.**
RMSNorm's sum of squares is an activation × activation term per channel and needs an inverse
square-root table; SubLN confounds any comparison across weight widths. **Status:** RMSNorm and a
trainable affine are parked; SubLN is rejected here.

### A06: pooling (average, max, scaled sum)

**Mechanism.** Replace global average pooling with max pooling or a power-of-two-scaled sum.
**Why here.** At N = 64 averaging is already an exact shift, so there is no accumulator problem to
fix; max pooling is a different inductive bias. **Hazards.** Max needs an hls4ml max reduction
over the token axis, not yet verified. All three are permutation invariant. **Prediction.** Scaled
sum and average are the same operation; max is worse. **Status:** parked.

### A07: attention pooling (PMA) or a class token

**Mechanism.** Pooling by multi-head attention with a learned seed vector, or a prepended class
token read out at the head. **Source.** Lee et al., "Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks", arXiv:1810.00825,
https://arxiv.org/abs/1810.00825. **Hazards.** A third activation × activation pair, the exact
operation that threatens the 0-DSP property if its scale is not foldable; a class token also
changes T from 64 to 65 in every shape. **Status:** rejected unless run as a labelled baseline.

### A08: head depth and width

**Mechanism.** The reference head is pooling, one hidden dense layer (d → d), ReLU, and the output
layer. This card uses three hidden layers of width 32. **Why here.** The head is the cheapest
place to add capacity per EBOP if the backbone is the accuracy bottleneck. **Source.** No
published ablation separates head depth from the rest of an architecture; Laatu et al. do not
state their head. **Change.** A list-valued `arch.head_dims`, about 25 lines on both sides.
**Hazards.** EBOPs linear in the added width; no new multiplier type; no export path yet.
**Prediction.** Flat. **Delta entries:** M045 and factor D of FF2.

### A09: a binary Deep Sets body (the threat card)

**Mechanism.** Replace attention with a Deep Sets body: a per-constituent MLP, a pooled context
added back to every constituent, a second per-constituent MLP, pooling and the head, all binary.
No Q, K, V or softmax. **Why here.** Laatu et al. report that their multi-head attention model at
N = 64 "is consistently collapsing over several trained models despite the bitwidth constrained to
at least one bit, turning it into a Deep Set" (section 3). If the same collapse happens with binary
weights, a binary Deep Set at equal EBOPs may match or beat binary attention, and the harder
attention-side work buys nothing at N = 64. **Sources.** Laatu et al., arXiv:2510.24784, section 3;
Komiske, Metodiev and Thaler, arXiv:1810.05165. **Published effect.** Laatu et al., Table 1, N = 64:
Deep Sets (HGQ) 79.4 % accuracy, their multi-head attention 77.9 %, their Linformer 79.8 %. HGQ
mixed precision, one model each, not comparable with any binary number without a matched run.
**Change.** A new model builder, about 80 to 120 lines; the layer widths are set before launch from
the published Deep Sets baselines. **Hazards.** No activation × activation products at all, so it
is more conservatively DSP-free than the transformer; exactly permutation invariant. No export
path yet. **Prediction.** Accuracy at 350k at least that of the 350k base. **Delta entry:** M006.

### A10: Linformer and MLP-Mixer

**Mechanism.** Linformer projects the keys and values along the sequence axis to k < N before
attention. MLP-Mixer replaces attention with a dense layer over the constituent axis.
**Why here.** Both avoid the N × N score matrix, and under the reference quantizers Linformer cuts
the softmax floor with the number of score entries: at k = 8 the d32h4 0-bit floor falls from
343,040 to 41,984 (computed from layer shapes). **Sources.** Wang et al., "Linformer: Self-Attention
with Linear Complexity", arXiv:2006.04768, https://arxiv.org/abs/2006.04768 ("performs on par with
standard Transformer models" at BERT scale); Laatu et al., Table 1, N = 64: Linformer 79.8 %, and an
MLP-Mixer row of 79.7 % quoted from their reference 18, which is not stated to be trained at 350k.
**Hazards.** Linformer's projections are two more binary matrices per attention call, and k must be
chosen with the budget in mind (Laatu et al. use k = 2). MLP-Mixer's token-mixing weights are
per position pair, so it is not permutation invariant. **Prediction.** Linformer at k = 8 is
feasible at 350k with real headroom and within the interval of the base at 5M. **Delta entries:**
M004 (k = 8) and the crosses M053, M054, M056 and M060. MLP-Mixer is parked; Linformer at k = 16 is
a parked second rung.

### A11: JEDI-linear-style interactions

**Mechanism.** JEDI-linear replaces the O(N²) pairwise interaction network with an edge function
affine in its two inputs, f_R(I_i, I_j) = W₁I_i + W₂I_j + C, and uses distributivity to rewrite the
sum over j as W₂ times a global average plus W₁I_i + C: one pooled quantity shared by every
constituent. That is closer to Deep Sets than to attention. **Source.** Que, Sun, Paramesvaran,
Clement, Karakoulaki, Brown, Laatu, Cox, Tapper, Luk and Spiropulu, "JEDI-linear: Fast and
Efficient GNNs for Jet Tagging on FPGAs", arXiv:2508.15468, https://arxiv.org/abs/2508.15468. The
paper cites a full pairwise interaction network at 30 particles using 71 % of a VU13P's DSPs
(LL-GNN, arXiv:2209.14065), and reports, for 64 particles with 16 features, 82.4 % accuracy, 79 ns
latency, 192k LUT and 0 DSP at II = 1 on a VU13P, and 81.8 % accuracy with 3 features (resources
not stated for that row). HGQ weights and da4ml synthesis; not a binary number. **Hazards.** It
trades genuinely pairwise information for a jet-level summary. **Status:** parked behind M006,
which answers the collapse question first.

### A12: Bi-Real shortcuts and residual scaling

**Mechanism.** An identity shortcut around the quantization step, not only around the sub-layer,
optionally with a scale. **Source.** Liu et al., "Bi-Real Net: Enhancing the Performance of 1-bit
CNNs", arXiv:1808.00278, https://arxiv.org/abs/1808.00278 (W1A1 ImageNet CNNs; mechanism only).
**Hazards.** A learned residual scale must be a power of two or a two-signed-digit constant, or it
reopens the DSP question. **Prediction.** No effect at one block, where the existing residual adds
already provide a path. **Status:** parked; revisit if M044 (two blocks) survives.

### A13: gated FFN (GLU variants)

**Mechanism.** Two parallel projections, one gated and multiplied element-wise with the other,
then projected down. **Source.** Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202,
https://arxiv.org/abs/2002.05202 (T5 log-perplexity, full precision; no transfer). **Hazards.** The
gate is an activation × activation multiply per FFN, the second-most direct threat to the 0-DSP
property after A07. **Status:** rejected.

### A14: FFN activation, ReLU or a tanh table

**Mechanism.** Replace the FFN's ReLU with HGQ2's `QAffinedUnaryFunctionLUT('tanh')`. **Hazards.**
A table whose size is keyed to its input and output widths, against a ReLU that costs a comparison.
**Status:** merged into M016, which places the tanh table before attention and the FFN as a range
bound.

### A15: low-rank factorization of binary layers

**Mechanism.** Factor a d × d binary matrix into d × r and r × d binary matrices. **Hazards.** The
rank-r intermediate stream is a new quantized activation site and pipeline stage that re-adds most
of the saving; the arithmetic d·r + r·d against d·d has to be checked per site. **Status:** parked;
no source, and predicted not to save EBOPs.

### A16: Engram-style conditional memory

**Mechanism.** A gated, hash-addressed lookup into a learned table, inserted after the first
attention residual, following DeepSeek's Engram design (arXiv:2601.07372,
https://arxiv.org/html/2601.07372v1). **Status:** rejected here. The memory tables are
multi-bit, they have no HLS lowering, and their cost lies outside native EBOPs. The idea has its own
record in this repository ([Engram study](../current-work/ENGRAM_STUDY.md)).

## Considered and left out

ParT-style pairwise interaction features, used as an attention bias, are a large addition: an
N × N × features tensor consumed by every head. BitParT (Rai et al., arXiv:2508.07431) keeps that
pathway at full precision for physics reasons. It needs its own cost pass before it can be carded
(open question OQ-17). LUT-native table layers (`QDenseT`) are a different design family, not a
binary one. A full ParT reproduction is too large for one entry. Ternary attention weights belong to
the labelled baselines.
