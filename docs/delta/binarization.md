# Family B: binarization and latent-weight optimization

Method cards for the weight side of the binary layers. Each card is a change to the reference
binarizer unless it says otherwise, and ends with the Delta entry it became (see
[methods.md](methods.md)) or the reason it was parked. Sources were read on 26 and 27 September
2026; where only an abstract was read, the card says so and quotes no table number. No card
contains a result of this project's own training.

## The current scheme

The reference binarizer is `bitnet_binary_ste` in
[`code/hgq2/bnhgq2/qat.py`](../../code/hgq2/bnhgq2/qat.py), mirrored for export by
`absmean_binarize` in [`binarize.py`](../../code/hgq2/bnhgq2/binarize.py), both per tensor:

```
alpha = mean(W)                      # centered; XNOR-Net and BitNet use the uncentered mean(|W|)
wc = W - alpha
beta = mean(|wc|) + 1e-6
q = where(wc >= 0, +1, -1)           # strict bipolar sign, never 0
ws = wc / stop_gradient(beta)        # bounded backward (no 1/beta blow-up as beta -> 0)
wq = ws + stop_gradient(q - ws)      # STE, forward = q
effective weight = wq * beta         # in {-beta, +beta}
```

The module docstring's claim that this replicates BitNet is overstated: BitNet and XNOR-Net use the
uncentered mean(|W|), while this scheme centers first. So it already does the zero-mean half of
what IR-Net's Libra-PB proposes (B06); what a Libra-PB card adds is standardization and a
power-of-two scale. The backward pass normalizes by stop_gradient(β) rather than clipping the
latents at ±1. That choice was made after an earlier backward produced NaNs when a kernel's β
collapsed (the docstring records it), so B04, B05 and B08 are changes from this STE, not from
plain BinaryConnect.

**The β-fold hazard.** Without normalization layers, every β is restored by an explicit affine.
The export builder ([`build.py`](../../code/hgq2/bnhgq2/build.py)) rounds each restored β to a
constant with at most two signed digits, which keeps it DSP-free; its comments record that a
general-valued β left in the weights inferred DSPs. Any card that turns the per-tensor scalar into
a per-channel or non-power-of-two vector multiplies that hazard by the number of channels. Each
card below says whether it is worse, the same or better.

### B01: centered per-tensor absmean (the reference)

The reference scheme, described above. BitNet (Wang et al., "BitNet: Scaling 1-bit Transformers
for Large Language Models", arXiv:2310.11453) is the nearest published form; its numbers are at
LLM scale and do not transfer. Every other card in this family is a change from B01.

### B02: uncentered per-tensor absmean (BitNet and XNOR-Net convention)

**Mechanism.** Drop the centering step: β = mean(|W|), q = sign(W). This is what the docstring
claims to replicate. **Why here.** A one-line test of whether centering buys anything in small
matrices, where the per-tensor mean may already sit near zero. **Sources.** Rastegari et al.,
"XNOR-Net", arXiv:1603.05279 (2016); Wang et al., BitNet, arXiv:2310.11453. No published number
isolates this choice. **Change.** A boolean `center` flag in the binarizer and its export mirror,
about 10 lines. **Hazards.** None beyond B01: one subtraction fewer, the same {−β, +β} output and
fold class. **Earlier attempts.** The two formulas were compared on one stored checkpoint and
never as a training arm; because β is in the gradient path, the uncentered form would train to a
different checkpoint. **Combines with.** Any STE variant (B04 to B08). **Prediction.** Flat
accuracy; fewer collective flips, because the threshold no longer moves with the tensor mean.
**Delta entry:** M017.

### B03: per-channel or learned scale (XNOR-Net++)

**Mechanism.** Replace the per-tensor β with a per-output-channel scale learned by backpropagation
instead of computed from mean(|W|). **Why here.** It could recover accuracy lost to forcing one
scalar on a whole matrix, though our matrices are small (d_model 32, FFN 32). **Source.** Bulat
and Tzimiropoulos, "XNOR-Net++: Improved Binary Neural Networks", BMVC 2019, arXiv:1909.13863
(abstract only: learned scales are "significantly more accurate" than analytic ones on ImageNet
with binary weights and activations; no table number read). **Change.** A trainable per-channel β
decoupled from mean(|W_c|), about 15 to 20 lines in the binarizer, plus every fold class in the
export. **Hazards.** This is the card that breaks the cheap fold. The query and key scale
currently joins the 1/√d softmax factor as one scalar; per-channel β_q and β_k vectors cannot, so
they become per-dimension rescales before a per-channel quantizer. On the explicit-scale layers
(input projection, attention output, second FFN layer, final head layer) a per-channel β becomes a
per-column multiply after the accumulator, worse than B01. Weights stay ±1. **Prediction.** Neutral
to slightly up in accuracy; EBOPs unchanged, fold cost up, so a naive EBOPs-against-accuracy
reading would look free when the hardware cost is not. **Delta entries:** kept only as power-of-two
gains, M024 (input projection) and M025 (every binary layer); the arbitrary learned per-channel β
is rejected.

### B04: clipped-identity STE (hard-tanh)

**Mechanism.** The backward passes gradient only where |w| ≤ 1, the original BinaryConnect and
Binarized Neural Networks backward. **Why here.** It is the default in most public binary-network
code, and it avoids the 1/β problem by clipping the latents instead of normalizing by β.
**Sources.** Hubara et al., "Binarized Neural Networks", NeurIPS 2016, arXiv:1602.02830; the STE
itself from Bengio, Léonard and Courville, arXiv:1308.3432. Their numbers are for CNNs with binary
weights and activations and do not transfer. **Change.** About 5 lines in the binarizer backward.
**Hazards.** None beyond B01; backward only. **Combines with.** Any scale scheme; excludes B05, B06's
backward half and B08, which use the same slot. **Prediction.** Flat to up at a peak LR of 3e-3,
with fewer divergences; the earlier NaN appeared at 4-bit activations, so a direction at other
widths is unclear. **Delta entry:** M018.

### B05: Bi-Real ApproxSign (piecewise-quadratic backward)

**Mechanism.** A piecewise-quadratic backward that approximates d(sign)/dw more tightly near zero.
The paper's shortcut connection is an activation-path change and belongs to family A (A12).
**Source.** Liu et al., "Bi-Real Net", ECCV 2018, arXiv:1808.00278 (abstract only; no table number
read). **Hazards.** Backward only. **Prediction.** Neutral to up, smaller than in Bi-Real's W1A1
setting because our activations are multi-bit. **Status:** parked; it shares the backward slot
with M018 and M019 and is designed for W1A1. A head-to-head follows EDE if EDE survives.

### B06: IR-Net Libra-PB (balanced, standardized binarization with a power-of-two scale)

**Mechanism.** Balance and standardize the latents (divide by their sd, not only recenter) before
the sign, and constrain the scale to a power of two so the multiply becomes a shift. **Why here.**
It is the one card in this family whose paper claims a cheaper scale in hardware, not only better
accuracy. **Source.** Qin et al., "Forward and Backward Information Retention for Accurate Binary
Neural Networks" (IR-Net), CVPR 2020, arXiv:1909.10788 (abstract only: "consistently outperforms"
other quantization methods on CIFAR-10 and ImageNet; no table number read). **Change.** Standardize
after centering and round β to 2^round(log₂ β) with a straight-through rounding, about 10 lines,
mirrored exactly in the export path. **Hazards.** The power-of-two constraint is the good case for
the fold table: a shift folds into a downstream binary point without a multiplier, whether or not
hls4ml treats it specially (unverified). Weights stay {−1, +1}. **Prediction.** Accuracy
non-inferior; a small LUT or DSP gain at export that only synthesis can show. **Delta entry:** M023
(the power-of-two absmean scale), read for non-inferiority and sent to a synthesis check if it
passes.

### B07: IR-Net EDE (annealed sign approximation)

**Mechanism.** A temperature-scheduled backward that starts near identity (wide support, low bias)
and sharpens toward the sign derivative (narrow support, low variance). **Why here.** The reference
schedule restarts the LR every 500 epochs. A fixed narrow STE may give unstable gradients right
after a restart, and a fixed wide one never sharpens; tying the anneal to the cosine cycle is a
natural pairing that IR-Net did not test. **Source.** Qin et al., arXiv:1909.10788 (as B06;
abstract only). **Change.** A step-dependent temperature in the backward, driven by an epoch
callback, about 15 lines. **Hazards.** Backward only. **Prediction.** Up if restarted with each
cosine cycle; flat or noisy if annealed once over 7,000 epochs while restarts fight it. **Delta
entries:** M019 (restarted each cycle), M072 (annealed once, with no restarts), and factor A of
FF1.

### B08: ReSTE (power-function backward)

**Mechanism.** A backward f(w) = sign(w)|w|^p with p annealed over training, framed as a trade
between estimation error and gradient stability. **Source.** Wu et al., "Estimator Meets
Equilibrium Perspective: A Rectified Straight Through Estimator for Binary Neural Networks
Training" (ReSTE), ICCV 2023, arXiv:2308.06689 (abstract only; no table number read).
**Prediction.** Up relative to B04 at CNN scale; weak at ours. **Status:** parked; same backward
slot as M018 and M019.

### B09: stochastic binarization

**Mechanism.** Sample +1 with probability hard_sigmoid(w/β) and −1 otherwise, instead of a
deterministic sign; export stays deterministic. **Source.** Courbariaux, Bengio and David,
"BinaryConnect", NeurIPS 2015, arXiv:1511.00363, which reports stochastic binarization comparable
to or slightly better than deterministic on small benchmarks; nothing transfers. **Hazards.** A
training-to-export mismatch if the stochastic and deterministic forward passes diverge beyond what
the export fidelity check allows. **Prediction.** Uncertain sign and wider seed spread; its main
claim is about spread, so it needs many seeds to read. **Status:** parked (low theory priority).
BinaryConnect's latent range [−1, 1] is the source of the clip in M021.

### B10: Bop (a binary optimizer)

**Mechanism.** Keep an exponential moving average m of the gradient per binary weight and flip the
weight once |m| exceeds a threshold τ with the matching sign; Adam still trains everything else.
The paper's argument is that latent weights provide inertia rather than "how binary" a weight is,
and Bop makes the inertia explicit. **Why here.** It removes the assumption that a float latent
updated by Adam is the right state for a ±1 weight, the largest training-loop change in this
family. **Source.** Helwegen, Widdicombe, Geiger, Liu, Cheng and Nusselder, "Latent Weights Do Not
Exist: Rethinking Binarized Neural Network Optimization", NeurIPS 2019, arXiv:1906.02107.
**Change.** A custom optimizer with per-variable routing, about 60 to 100 lines. **Hazards.** None
on the export side (the weight is still ±1); on the training side a wrong threshold can over- or
under-flip. Combining Bop with a learned per-channel β means two optimizers over disjoint variable
sets. **Prediction.** Unknown sign: Bop was designed and validated for W1A1, so the gradient-noise
regime that motivates it is only partly present here. **Delta entries:** M020, and M071 (Bop with
no restarts). The hyperparameters and the flip rule are discussed next.

#### Bop hyperparameters

Helwegen et al. give γ and τ for three experiments. Section 5.1, Fig. 1, is a qualitative sweep on
BinaryNet and CIFAR-10 over 100 epochs: γ ∈ {10⁻², 10⁻³, 10⁻⁴} at τ = 10⁻⁶, and τ ∈ {0, 10⁻⁶, 10⁻⁵}
at γ = 10⁻³. High γ and low τ both raise the flip rate and give fast, noisy early learning that
plateaus low; (10⁻², 10⁻⁶) and (10⁻³, 0) are named as aggressive settings, and at τ = 0 validation
accuracy deteriorates over time. No working point is recommended from it. Section 5.2, the CIFAR-10
benchmark (BinaryNet-style VGG, 500 epochs, batch 50), uses γ = 10⁻⁴ decayed by 0.1 every 100
epochs and τ = 10⁻⁸ fixed, with Adam at initial LR 10⁻² for the BatchNorm variables; the text
reports 91.3 % top-1 for Bop against 90.9 % for the latent-weight Adam baseline. Section 5.3,
ImageNet (BinaryNet, XNOR-Net and Bi-Real Net, batch 1,024), uses τ = 10⁻⁸ and γ decayed linearly
from 10⁻⁴ to 10⁻⁶; Table 1 reports top-1/top-5 of 41.1/65.4, 45.9/70.0 and 56.6/79.4 for Bop
against 40.1/66.3, 44.2/69.2 and 56.4/79.5 for the latent-weight baselines. τ is never decayed.
Section 6 proposes γ decay as an LR analogue and a scheduled τ as future work, not as used values.

The Larq implementation, `larq.optimizers.Bop`, defaults to threshold 1e-8 and gamma 1e-4 with no
decay (larq/larq `master`, commit 3d7de8832a477285bbf3c36252e24fcb9299a959, `optimizers.py`
lines 314 to 316; a master commit, not a tagged release). Its docstring warns that the default
threshold "is not optimal for all situations". Its update flips to the literal negation of the
weight, with no latent and no scale, as in the paper's Algorithm 2.

None of this transfers directly. The paper's networks are W1A1 convolutional networks with only
BatchNorm scalars trained by Adam; ours is a W1A8 transformer with many more float variables. γ
sets an averaging horizon of about 1/γ steps, so decay schedules written in epochs at batch 50 or
1,024 mean something different at batch 2,790. And τ is an absolute threshold on the averaged
gradient, which the paper itself notes depends on gradient magnitude; our gradient is the bounded
STE gradient after value clipping, in different units. M020 and M071 therefore use γ = 1e-4,
undecayed, and τ = 1e-8 as a starting point only, and the |m| distribution of the reference binary
latents is measured before launch, which may set a τ scan.

The flip rule is fixed before launch. Option 1 keeps the latent and reflects it about the tensor
mean on a flip, w ← 2α − w, which preserves |w − α| and so the layer scale β, keeping activation
ranges comparable with an Adam-trained base; the flip condition is Algorithm 2's, but the state
and the scale bookkeeping are an extension the paper argues is unnecessary. Option 2 is Bop as
published: the weight is the sign itself and a flip sets w ← −w with latents at ±1, so β becomes
1 − α² and activation scales change relative to the base. Eq. (6) of the paper uses |m| ≥ τ while
Algorithm 2 uses |m| > τ; the two differ only at equality.

### B11: BinaryBERT ternary weight splitting (two-stage ternary to binary)

**Mechanism.** Train a half-width ternary network, then split each ternary weight into two binary
weights so that the doubled binary network reproduces the ternary output exactly, and fine-tune.
**Source.** Bai et al., "BinaryBERT: Pushing the Limit of BERT Quantization", ACL 2021,
arXiv:2012.15701 (abstract: "only a slight performance drop compared with the full-precision
model while being 24x smaller" at BERT-base scale; no table number read). **Hazards.** The
intermediate stage is ternary, a labelled deviation during training only; the architecture doubles
mid-run, which breaks the EBOPs controller and the seed pairing. **Status:** parked. The two-stage
idea is tested more cheaply by M027 (warm start from a float teacher).

### B12: BiT (elastic binarization and multi-step distillation)

**Mechanism.** A "two-set" binarization with learned levels, an elastic binary activation, and
distillation through successively lower-precision teachers. **Source.** Liu et al., "BiT: Robustly
Binarized Multi-distilled Transformer", arXiv:2205.13016 (abstract: a binarized transformer within
5.9 % of a full-precision BERT baseline on GLUE; no table number read). **Hazards.** The two-set
piece may mean more than two effective weight levels at inference, which would break the thesis;
the distillation piece touches only the loss. **Delta entries:** the staged-teacher idea became
M027 (warm start) and M063 (distillation from an int8 teacher); the two-set levels are rejected.

### B13: ReActNet RSign and RPReLU (learnable activation shifts)

**Mechanism.** A learnable per-channel threshold before the sign (RSign) and a learnable shift and
slope after the nonlinearity (RPReLU), correcting for activation distributions that drift away
from a fixed threshold. Here it is adapted to multi-bit activations as a per-channel shift before
each activation quantizer. **Source.** Liu, Shen, Savvides and Cheng, "ReActNet", ECCV 2020,
arXiv:2003.03488, which reports 69.4 % ImageNet top-1 with binary weights and activations, a W1A1
CNN number that does not transfer. **Hazards.** The weight side is untouched; the shift is an add,
cheap in fixed point, but it may change which widths the controller settles on. **Prediction.** Up,
and the budget may be met at lower widths. **Delta entry:** M026 (shift only, no slope), and
M066 with a power-of-two gain.

### B14: ternary baselines (BitNet b1.58, TWN, TTQ)

**Mechanism.** Weights in {−1, 0, +1} with one (TWN, b1.58) or two (TTQ) learned scales per layer.
**Why here.** Ternary is a comparison, never the thesis. It is also where an independent benchmark
points: Sloot (FastML 2026) found BitNet-1.58 ternary MLPs ahead of binary ones on AUC and LUT on a
16-feature, two-class jet task (Table 1: 0.9254 against 0.9178 AUC, 80.7k against 87.3k LUT,
C-synthesis estimates on a VU13P). **Sources.** Li, Zhang and Liu, "Ternary Weight Networks",
arXiv:1605.04711; Zhu, Han, Mao and Dally, "Trained Ternary Quantization", arXiv:1612.01064; Ma et
al., "The Era of 1-bit LLMs", arXiv:2402.17764; Sloot, "Do BitNet Gains Survive Synthesis? An
Implementation-Aware Benchmark of Low-Precision FPGA Inference for Jet Classification", FastML
2026, https://indico.cern.ch/event/1654479/papers/7189055/files/16211-11_Do_BitNet_Gains_Survive_Syn.pdf.
**Hazards.** Three weight values per layer; every card that could introduce a zero state must be
checked for zero weights at export. How HGQ2 bills a ternary zero is open (Z06). **Delta entry:**
M047, a labelled baseline and a feasibility probe at 350k.

## Considered and left out

**EMA or SWA of latent weights** has no binary-specific source with a number; the recipe family
carries it as M034. **Flip-rate regularization** appears as a diagnostic in several binary-network
papers, including asides in Helwegen et al., but not as a method with its own source; Bop's
threshold is an implicit flip-rate control. **Binary-friendly initialization** is one of Bi-Real
Net's listed contributions rather than a separate method. **BiBERT** (Qin et al., "BiBERT:
Accurate Fully Binarized BERT", ICLR 2022, arXiv:2203.06390) is mainly an attention-side and
distillation-side contribution, overlapping B12 and B13, and is used in the theory note instead.
**Generic soft-sign annealing** duplicates B07 and B08. **Per-channel weight decay and latent
clipping** are hyperparameters inside BinaryConnect and Bop rather than published methods; latent
clipping is tested as M021.
