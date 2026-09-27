# What Delta tests: mechanisms of accuracy loss in a binary-weight tagger

This note organizes the method cards in the five family notes by mechanism, so that each Delta
entry tests a named cause of lost accuracy instead of adding one more trick to a pile. It makes
predictions and nothing else: no training has been done, and every predicted sign below is a
hypothesis for Delta to test.

Statements carry one of three labels. **[source]** is a primary source that was read, with its
section, table or figure. **[derived]** is arithmetic or a code reading on a named file of this
repository. **[conjecture]** is reasoning that Delta is meant to test.

**Transfer warning, for every source below.** None of the cited works studies a transformer of
about 10,000 parameters with three input features, no normalization layers and an EBOPs budget.
Most are ImageNet CNNs (the binary-network literature) or BERT-scale and LLM-scale models, and
several binarize activations as well as weights. Their numbers indicate the direction of an
effect, never its size here.

## 0. The binarizer and the cost rule the theory relies on

The binarizer is `bitnet_binary_ste` in
[`code/hgq2/bnhgq2/qat.py`](../../code/hgq2/bnhgq2/qat.py): α = mean(W), W_c = W − α,
β = mean|W_c|, q = +1 if W_c ≥ 0 else −1, and the effective weight is q·β. There is one β per
tensor, and centering balances each tensor's signs by construction. The backward pass is an
identity STE on W_c / stop_gradient(β), with no clip on the latents. Because β is computed from
the latent magnitudes, a layer's output scale and its latents' inertia are one quantity [derived].
The graph has no normalization layers, so each β is restored by an explicit affine after its own
matmul, and the export builder rounds those constants to two signed digits so that they stay
DSP-free ([`build.py`](../../code/hgq2/bnhgq2/build.py)) [derived]. Biases are float, and a
learned positional table folds into the input projection's bias on export. The input projection
is itself a binary layer with fan-in 3.

In the reference configuration ([README](README.md#the-reference-configuration)) the activation
widths are learned per channel and can reach 0 bits; the softmax output is learned and the exp and
inverse tables are trainable. Earlier project runs used a saturating quantizer whose channels
never fell below 1 bit and a softmax output fixed at 10 bits. Width, in this note, means the
learned relu(i + f) of a channel, with the sign bit reported separately.

EBOPs, as HGQ2 counts them, are Σ b_i·b_j over products plus Σ max(b_k, b_l) over explicit
additions; a 0-bit operand is pruned [source: Sun et al., HGQ, arXiv:2405.00645v3, section 3.3,
Eq. 11, https://arxiv.org/abs/2405.00645]. For our layers that gives N·fan_in·fan_out·b_act for
a weight × activation layer with ±1 weights, N·N·d·b_q·b_k for the attention scores, and
N·N·d·b_A·b_V for the attention-weighted sum, plus a softmax term of its own [derived]. The
shape formulas Delta uses are in [rules.md](rules.md#static-floors).

## 1. Where accuracy goes when weights become ±1

### M1: capacity per parameter, and the fan-in-3 first layer

In high dimension, binarization keeps a vector's direction: the angle between a random vector and
its sign vector tends to arccos √(2/π) ≈ 37°, and weight·activation dot products stay highly
correlated. The correlation is much weaker in the first layer [source: Anderson and Berg,
arXiv:1705.07199 (2017), "Angle Preservation" and Fig. 3, https://arxiv.org/abs/1705.07199].

Our first layer has fan-in 3 (pT, η_rel, φ_rel). A ±1 row over three inputs has 2³ = 8 sign
patterns, 4 up to a global sign. With one β per tensor, the 32 output channels of the input
projection can point in at most 8 input directions, all with equal gain, and differ otherwise
only through the float bias that carries the positional table [derived]. There is no
nonlinearity between the input projection and Wq, Wk, Wv, so the 8-direction embedding composes
with the next ±1 layer into small-integer coefficients. Rank at most 3 would hold for a float
input projection too; the binary-specific loss is the discreteness of directions and gains, not
the rank [derived and conjecture]. Hidden layers have fan-in 32 and 2³² patterns per row, so there
the angle argument applies and the loss is the distortion per row, not a shortage of directions
[conjecture]. Binary networks usually buy capacity back with width, and the EBOPs budget removes
width (M7) [conjecture].

*Observables.* The number of distinct sign rows of the input projection's kernel on a checkpoint.
A labelled non-binary input projection as a baseline, with its gap to the base at the same seeds.
*Families.* P (more or derived input features), B (a power-of-two per-channel gain on the input
projection), A (a wider d_model). A float or multi-bit first layer is a baseline only, because a
layer with more than two weight values breaks the thesis.

### M2: gradient mismatch in the straight-through estimator

The STE is biased and low-variance [source: Bengio, Léonard and Courville, arXiv:1308.3432 (2013),
https://arxiv.org/abs/1308.3432]. A well-chosen STE gives a coarse gradient whose expectation
correlates positively with the population gradient, and a poor one gives instability near some
minima [source: Yin et al., ICLR 2019, arXiv:1903.05662, https://arxiv.org/abs/1903.05662]; that
analysis is for activation quantization, so it transfers to our weight STE only by analogy.
BiBERT names "optimization direction mismatch" in the backward pass as one of two causes of the
fully binarized BERT's drop [source: Qin et al., ICLR 2022, arXiv:2203.06390, sections 1 and 3.3,
https://arxiv.org/abs/2203.06390]. IR-Net treats the backward loss with an annealed sign
approximation, EDE [source: Qin et al., arXiv:1909.10788 (2019), https://arxiv.org/abs/1909.10788].
Ours is an identity STE with no clip, so it has no dead-weight zone and passes no information
about the distance to the threshold [derived].

*Observables.* The latent-versus-binary evaluation gap, with the latent float kernels in place of
q·β; Helwegen et al. report that latent-weight evaluation is not better than binary [source:
arXiv:1906.02107, section 3]. The cosine between the STE gradient and the latent-model gradient on
one batch. *Families.* B (clipped or annealed STEs), R (the LR magnitude).

### M3: latent-weight flips, inertia and oscillation

Latent magnitude is inertia: the larger |w̃|, the stronger the gradient signal needed to flip the
sign. The optimizer mostly changes inertia rather than binary weights, clipping caps inertia, and
lowering the LR late in training raises it [source: Helwegen et al., NeurIPS 2019,
arXiv:1906.02107, section 3, https://arxiv.org/abs/1906.02107]. Their Theorem 1 (the binary
trajectory is invariant to the LR if the initialization is rescaled) needs a pseudo-gradient that
does not depend on |w̃|. Ours violates it: β = mean|W_c| scales the forward pass, so inertia and
output gain move together [derived]. Latents that grow over a long run make a layer both louder
and stiffer [conjecture].

Weight decay on latents trades stability for dependence on the initialization, and the flip-flop
(FF) ratio rises with it. Liu et al., Table 1 (ImageNet top-1, Adam, their ResNet-18-based binary
network): weight decay 1e-5 gives FF 2.33e-3 and 61.73 %; 5e-6 gives 1.62e-3 and 61.89 %; 0 gives
2.86e-4 and 61.49 %; two steps (5e-6, then 0) give 4.50e-4 and 63.23 %. Adam revives dead weights
better than SGD [source: Liu et al., ICML 2021, arXiv:2106.11309, sections 3.2 and 3.3, Table 1,
https://arxiv.org/abs/2106.11309].

In quantization-aware training, latents oscillate around a decision threshold; a lower LR shrinks
the amplitude but not the frequency, and the harm comes through corrupted BatchNorm statistics
and training noise, with dampening and iterative freezing as remedies [source: Nagel et al., ICML
2022, arXiv:2203.11086, sections 2 to 4, https://arxiv.org/abs/2203.11086]. We have no BatchNorm,
so their main harm path is absent. What remains is noise, and checkpoint-to-checkpoint jitter that
per-epoch selection can exploit or suffer from [conjecture]. Our threshold is the tensor mean α,
which moves as the weights update: a second, collective source of flips that the literature above
does not treat [derived].

*Observables* (none logged today): the per-layer FF ratio per epoch from sign snapshots (7,424
binary weights in d32h4, negligible cost); the C2I ratio (final sign against initial sign); the latent
|w̃|/β histogram; the fraction of latents within one Adam step of the threshold; the per-layer β
trajectory. *Families.* R (optimizer, weight decay, LR, restarts, Bop), B (latent clip, β
decoupled from latent magnitude).

### M4: loss of scale information

A real-valued scale beside binary weights recovers much of the accuracy lost to binarization;
XNOR-Net uses one α = mean|W| per filter [source: Rastegari et al., ECCV 2016, arXiv:1603.05279,
https://arxiv.org/abs/1603.05279]. Data-driven per-channel rescaling and learnable shifts (RSign,
RPReLU) help further [source: Martinez et al., ICLR 2020, arXiv:2003.11535, section 4.3,
https://arxiv.org/abs/2003.11535; Liu et al., ReActNet, ECCV 2020, arXiv:2003.03488,
https://arxiv.org/abs/2003.03488]. We have one β per tensor and no LayerNorm γ to supply a
per-channel gain; BitNet's SubLN placement gives one, and our norm-free choice removed it
[derived]. Per-channel output magnitude is therefore set by the input statistics and the ±1
pattern alone [conjecture on the consequence]. Learned activation widths set precision and range,
not gain, so they cannot stand in for a missing per-channel scale [derived: a quantizer preserves
values within its range].

*Observables.* The per-channel pre-activation sd of each binary layer on validation jets, the
dead-ReLU fraction, and the least-squares per-channel α on a probe batch against the per-tensor β;
a large spread means scale information is being lost. *Families.* B (per-channel α restricted to
powers of two, see section 3), A (a per-channel shift), Q.

### M5: attention-specific failure

**Temperature coupling [derived and conjecture].** The logit scale is β_q·β_k·(Q·K of ±1
projections)/√d_h, so a head's temperature can only move through two per-tensor β's that are also
inertia (M3). Attention entropy is bounded below by a quantity that falls exponentially with the
spectral norm of the logits, and low entropy goes with instability [source: Zhai et al., ICML
2023, arXiv:2303.06296, https://arxiv.org/abs/2303.06296]. Large β's risk peaked attention and
small ones give flat attention.

**Information degradation [source, partial transfer].** In BiBERT, binarizing the softmax output
drives the entropy of the binarized attention to 0, and Bi-Attention restores it by maximizing
entropy [source: arXiv:2203.06390, section 3.2]. BiBERT binarizes weights, activations and
embeddings. Its argument applies here only as the budget drives the softmax output and the Q/K/V
widths toward 1 or 0 bits (M7).

**Rank collapse [source, weak transfer].** Pure attention loses rank doubly exponentially with
depth, and skip connections and MLPs prevent it [source: Dong, Cordonnier and Loukas,
arXiv:2103.03404 (2021), https://arxiv.org/abs/2103.03404]. With one block this is weak for the
base and matters only for L ≥ 2 [conjecture].

**Budget-induced collapse [source and derived].** Laatu et al. report that their multi-head
attention model at N = 64 and 350,000 EBOPs "is consistently collapsing over several trained
models despite the bitwidth constrained to at least one bit, turning it into a Deep Set", with
HGQ, not binary, weights [source: Laatu et al., arXiv:2510.24784, section 3,
https://arxiv.org/abs/2510.24784]. Collapse at this budget is therefore not a binary-weight
effect. Section 3 below shows why the counting rule prices attention out first, and
[rules.md](rules.md#static-floors) shows that a 4-head d_model 32 block at N = 64 cannot keep its
attention alive at 350k. Attributing a collapse to binary weights needs a matched non-binary arm
(M049).

**Measurement hazard [derived and conjecture].** The pT gate turns every gated constituent into
the same standardized vector, with no mask. Entropy as a fraction of log 64 is then inflated by
the padding fraction of each jet, not by collapse, so the diagnostic should be reported against
log(n_valid) per jet, or with gated keys masked in the diagnostic pass only. Without positional
encoding, gated tokens are exact duplicates. With global average pooling and no mask, the pooled
mean also encodes the number of gated constituents, a multiplicity feature as much as a bug
[conjecture].

*Observables.* The Q/K and V 0-bit fractions and the entropy from logged widths and a softmax
tap; entropy over log(n_valid); per-head logit sd; and an attention-ablation delta, the validation
accuracy with the attention output replaced by the mean over V, which tests Deep-Set collapse
independently of the widths. *Families.* A (Linformer, fewer heads, no positional encoding, a
Deep Sets control), Q (table widths, Q/K width floors, a per-group penalty weight), P (masking
gated keys).

### M6: the interaction with the norm-free design

BitNet places LayerNorm (SubLN) to stabilize 1-bit training [source: Wang et al.,
arXiv:2310.11453, https://arxiv.org/abs/2310.11453]. σReparam trains transformers without
LayerNorm by controlling spectral norms [source: arXiv:2303.06296]. Binary networks are sensitive
to shifts in the activation distribution [source: ReActNet, arXiv:2003.03488, abstract]. Without
normalization and with absmean scales, latent drift (M3) changes β, hence the range of the
residual stream, hence what the learned integer bits must cover: either the quantizers saturate
or wrap, or the integer bits grow and EBOPs rise [derived from the code path; the sign of the net
effect is a conjecture].

*Observables.* The correlation of each layer's β trajectory with its integer-bit trajectory i;
the saturation fraction at each quantizer. *Families.* A (a normalization layer, which costs LUT
and possibly DSP, see section 3), B (β as a separate power-of-two parameter), R.

### M7: width learning under an EBOPs penalty

With weights pinned at 1 bit, every weight × activation term costs MACs·b_act, so the β penalty
can lower cost only by narrowing or zeroing activation channels, and a 0-bit channel deletes a
whole row of MACs [derived]. HGQ weights can instead be narrowed or pruned per parameter, with 0-bit
weights "effectively pruned" [source: arXiv:2405.00645, section 3.3]. Binary therefore pays for
cost reduction with structured width loss, while HGQ pays with unstructured sparsity. This is the
sharpest binary-specific mechanism Delta tests [conjecture on its size].

Reference points computed from layer shapes for d32h4 at N = 64: at 8-bit widths the attention
scores alone cost 64·64·32·8·8 = 8,388,608 EBOPs, against 3,204,352 for all the dense layers and
the head together. At 1-bit widths the six d × d layers of the block (Q, K, V, output, and the
two FFN layers at FFN 32) cost 6 · 64·32·32 = 393,216, already above 350,000 [derived]. A feasible
350k binary d32h4 at N = 64 must zero a large share of its channels, and attention is the
cheapest place to find the EBOPs.

*Observables.* Per-layer, per-channel 0-bit fractions and the EBOPs split (activation ×
activation, weight × activation, softmax) against epoch; the effective residual width.
*Families.* Q (width floors, per-group penalty weights, the β controller, table widths), A
(Linformer and fewer heads to cut the N² terms, a smaller d_model with wider bits), P (fewer
constituents).

## 2. Why the reference recipe might matter more for binary weights than for HGQ weights

**Steps, not epochs [derived].** At 558,000 training jets, batch 2,790 gives 200 steps per epoch,
so 7,000 epochs is 1.40M steps. The project's earlier recipe (batch 256, 1,000 epochs) on the same
split takes about 2,180 steps per epoch, 2.18M in all. The reference takes fewer optimizer steps,
sees 7 times the samples and peaks at a 150 times higher LR (3e-3 against 2e-5). Any gain it shows
over the earlier recipe is about LR magnitude, batch noise and restarts, not more updates.

**Adam defaults [source and conjecture].** Adam's second moment revives dead weights and copes
with a rugged binary landscape better than SGD [source: arXiv:2106.11309, section 3.2]. Under
Adam the per-step latent move is about LR-sized whatever the gradient, so the ratio LR/|w̃| sets
the flip propensity [conjecture, grounded in Helwegen et al., section 3]. At 3e-3, flips stay
possible throughout training; at 2e-5 far fewer signs may leave their initial values [conjecture;
test: the C2I ratio of the two recipes].

**Weight decay [source and derived].** Weight decay raises the FF ratio and lowers the dependence
on the initialization [source: arXiv:2106.11309, Table 1]. With absmean scales it also shrinks β
and so every layer's output gain [derived]. Removing it tests M3 and M6 together; the confirming
observable, independent of accuracy, is a higher FF ratio and a smaller β trajectory with decay.

**Warm restarts [source and conjecture].** SGDR improves anytime performance [source: Loshchilov
and Hutter, ICLR 2017, arXiv:1608.03983, https://arxiv.org/abs/1608.03983]. For a binary network
each restart is a step change in LR against accumulated inertia, so it should give a burst of
flips, and the low-LR end of each cycle should freeze signs again [conjecture, via Helwegen et
al., section 3, and Nagel et al., section 2: the amplitude scales with the LR]. A restart also
perturbs the β controller and the widths [conjecture]. *Observable:* the FF ratio against epoch,
with a spike at each 500-epoch boundary; the position of the best feasible checkpoint within its
cycle (predicted late); whether feasibility is lost after each restart.

**Large batches and flat minima [source and conjecture].** Large batches tend toward sharp minima
[source: Keskar et al., ICLR 2017, arXiv:1609.04836, https://arxiv.org/abs/1609.04836]. The binary
loss surface is steeper than the ternary or float one [source: Bai et al., BinaryBERT,
arXiv:2012.15701 v2, section 2.2, Figs. 2 and 3, https://arxiv.org/abs/2012.15701]. For a binary
network, less gradient noise also means fewer spurious flips. The two effects pull in opposite
directions, so the sign of the batch effect is open [conjecture].

**A long schedule for the widths [conjecture].** With HGQ weights the budget can be met early by
pruning weights. Binary has to reorganize which channels it uses, so it plausibly needs a longer
schedule.

## 3. The cost side

**What EBOPs counts** [source: arXiv:2405.00645v3, section 3.3, Eq. 11]: products Σ b_i·b_j and
explicit additions Σ max(b_k, b_l). Control logic and FIFOs are excluded. The paper relates EBOPs
empirically to LUT and DSP usage on hls4ml for UltraScale+.

**What EBOPs leaves out for ±1 weights [derived].** A ±1 MAC costs b_act in EBOPs, the width of
one addition or subtraction. The adder tree's width grows by about log₂(fan_in) bits (5 bits at
fan-in 32), and that growth is not charged. Equal EBOPs is not equal LUT.

**The scale-restore multiplies.** A per-tensor scale-restore multiply stays DSP-free only when its
constant is a power of two or has at most two nonzero signed digits; the export builder rounds β
that way, and its comments record that a general-valued scale left in the weights inferred DSPs
([`build.py`](../../code/hgq2/bnhgq2/build.py)).

**HGQ-LUT, the rival route to 0 DSP.** It turns each neuron into trained logic LUTs with
per-element zero-bit pruning and a LUT-count surrogate beside EBOPs. It deletes operations where
we make each one cheaper. It has no transformer or attention support, and its cost model has no
term that rewards ±1 weights over a pruned 2-bit weight [source: Sun et al., arXiv:2604.22293,
https://arxiv.org/abs/2604.22293]. It is not a lever inside a binary attention block; a LUT-based
head is the only crossing point, and it needs a different toolchain [conjecture].

**A neighbour's measurement.** In one hls4ml flow on a VU13P, BitNet-1.58 ternary MLPs beat
binary ones on AUC, LUT and latency for a 16-feature, two-class jet task [source: Sloot, "Do
BitNet Gains Survive Synthesis?", FastML 2026, Table 1,
https://indico.cern.ch/event/1654479/papers/7189055/files/16211-11_Do_BitNet_Gains_Survive_Syn.pdf].
Binary is not automatically the cheaper point.

| class | examples | EBOPs | FPGA reality |
| --- | --- | --- | --- |
| training-only | recipe, distillation, STE variants, Bop, latent clip, restarts, progressive binarization that ends at ±1 | unchanged | free [derived] |
| EBOPs-visible | activation widths, N, d_model, heads, the activation × activation terms | charged | roughly tracks LUT [source: HGQ] |
| EBOPs-invisible, LUT-real | accumulator growth, bias adds, requantization integer bits, extra affines, normalization layers, the positional add | not charged | real LUT, possible DSP |
| DSP hazard | per-channel scales with arbitrary constants | not charged | DSP unless each scale is a power of two or has at most two signed digits |
| thesis-breaking | ternary, a multi-bit first layer, per-weight scales | – | baseline only |

How HGQ2 bills a ternary layer's zeros (as pruned 0-bit weights, or as 1-bit ones) is not stated
in the HGQ text (open question OQ-05 in [rules.md](rules.md#open-questions)).

## 4. Mechanism, method and predicted sign

Each cell names the lever, the predicted sign on validation top-1 accuracy at a fixed EBOPs target
(+ helps, − hurts, 0 neutral, ± depends on the budget), and the diagnostic that confirms the
mechanism without looking at accuracy. Every sign is a conjecture; that is what Delta tests.

| mechanism | B binarization | R recipe | Q widths and EBOPs | A architecture | P inputs |
| --- | --- | --- | --- | --- | --- |
| M1 capacity, fan-in 3 | power-of-two per-channel gain on the input projection: +; more distinct sign rows | 0 | width floor on the input-projection output: +; fewer 0-bit channels | wider d_model at a fixed target: ±; effective width | derived features (log pT, ΔR), fan-in above 3: +; more distinct rows, smaller latent-vs-binary gap |
| M2 STE bias | clipped or annealed STE: + if the instability sits in M2; higher STE/latent gradient cosine | lower final LR: +; smaller latent-vs-binary gap | 0 | 0 | 0 |
| M3 flips, inertia | latent clip, dampening or freezing: +; lower FF ratio late in training | no weight decay against decay: + without; Bop: ±; restarts: + with FF spikes at the boundaries | 0 | 0 | 0 |
| M4 scale loss | per-channel power-of-two α: +; per-channel sd spread explained | 0 | per-channel widths, already on: 0 | per-channel shift: +; fewer dead ReLUs | 0 |
| M5 attention | learned Q/K temperature: +; entropy over log n_valid away from 0 and 1 | distillation with an attention-map term: +; larger attention-ablation delta | Q/K/V width floors or exempting attention from β: ± (+ at 5M, possibly − at 350k because cuts move elsewhere); 0-bit fractions | Linformer, fewer heads, a Deep Sets control: + at 350k; EBOPs split | masking gated keys: +; entropy normalization |
| M6 norm-free drift | β decoupled from latent magnitude: +; lower β-versus-i correlation | no weight decay: + (β not shrunk) | 0 | a normalization layer: + accuracy, − LUT (not EBOPs) | 0 |
| M7 width against budget | 0 | long schedule: +; EBOPs reaches target | per-group penalty weights, width floors, the β controller: ±; 0-bit maps | cut the N² terms: + at 350k, 0 at 5M | fewer constituents: + at 350k if attention survives; EBOPs split |

## 5. Interactions: which combinations to cross

**Synergy or redundancy, crossed in small factorials at matched seeds:**

1. **Distillation × progressive binarization.** Two-step training (activations first, then
   weights) and a sequence of teacher-student pairs close the gap more than either alone [source:
   Martinez et al., arXiv:2003.11535, sections 3 and 4; Liu et al., arXiv:2106.11309, Table 1,
   two-step rows]. BiT distills through successively lower-precision teachers [source: Liu et
   al., NeurIPS 2022, arXiv:2205.13016, https://arxiv.org/abs/2205.13016]. BinaryBERT starts a
   binary model from a half-sized ternary one by ternary weight splitting [source:
   arXiv:2012.15701, Fig. 4]. Here the intermediate stage must be float or HGQ, with the final
   layers at ±1. Predicted: synergy, larger at the low budget [conjecture].
2. **Per-channel scale × learned widths.** A power-of-two per-channel gain is free in hardware (a
   shift absorbed into the next quantizer's binary point) and changes relative channel
   contributions, which widths cannot do [derived]; the widths then re-adapt. Predicted:
   synergy, with the order of training mattering [conjecture].
3. **Restarts × latent treatments (clip, weight decay, Bop).** All three lower effective inertia,
   so they should be partly redundant. Bop has no LR, so cosine restarts do not act on it unless
   γ or τ is scheduled [source: arXiv:1906.02107, section 4; conjecture on the redundancy].
4. **EBOPs target × attention levers.** At 5M attention is affordable and at 350k it is priced
   out (M7), so every attention lever runs at both targets or its sign cannot be read [derived
   and conjecture].
5. **Gate × positional encoding × masking.** Duplicated gated tokens interact with removing the
   positional table and with the entropy diagnostic; masking changes both the accuracy and the
   diagnostic [conjecture].
6. **Input features × first-layer scale (M1).** Both enlarge the set of directions the input
   projection can express, so they are probably redundant at the margin.

**Expected additive, so main effects suffice:** training-only B and R levers acting on M2 and M3
against P levers acting on M1; logit distillation against architecture size; data order and split
seeds against everything [conjecture]. FF1 tests this instead of assuming it, and a pair that
shows a paired interaction outside its interval in any wave is promoted to a factorial.

## 6. Open theoretical questions

1. Is 350k at N = 64 reachable by d32h4 with a nonzero attention branch? The shape arithmetic in
   [rules.md](rules.md#static-floors) says not with every attention channel alive; the trace and
   the reference widths at epoch 500 settle the rest.
2. Is the collapse Laatu et al. report pure budget geometry? A matched HGQ-weight arm on this
   pipeline at 350k and 5M, with the attention diagnostics and the ablation delta, settles it.
3. Is the fan-in-3 input projection the binding capacity loss? The distinct-row count on trained
   checkpoints and a labelled non-binary input projection at matched seeds settle it.
4. Does the earlier low-LR recipe train signs at all? If its C2I ratio is near 1, the recipe gap
   is M3, not capacity.
5. How does HGQ2 bill a ternary zero, and what is the LUT per EBOP of a ±1 layer at fan-in 32? A
   trace of one ternary layer and one single-layer synthesis through place and route settle it,
   and decide whether iso-EBOPs comparisons flatter binary.
6. Does a restart schedule beat one cosine of equal total steps for binary weights? One matched
   arm settles it; the observable is the FF spikes against the best-checkpoint position.
7. Does the ordering at the trigger operating point match top-1 accuracy and AUC? Every
   comparison should add per-class signal efficiency at fixed mistag. In BitParT, AUC moved by
   0.0006 while background rejection fell by about 15 to 17 % [source: Rai et al.,
   arXiv:2508.07431, Top Tagging benchmark, results table, https://arxiv.org/abs/2508.07431].
8. Literature still to read: weights-only binary small transformers with multi-bit activations;
   Bop or flip-aware optimizers on transformers; sharpness-aware training for binary networks; and
   whether BinaryBERT's 2-bit to 1-bit weight cliff (their Fig. 1: about 3.8 points on MRPC and
   0.9 on MNLI-m, BERT with 8-bit activations) has a small-model analogue. If it does, it
   threatens the binary framing.
