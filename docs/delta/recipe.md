# Family R: training recipe, optimization and regularization

Method cards for the training recipe. Each card is a change to the reference recipe (Adam with
framework defaults, cosine restarts with peak LR 3e-3 every 500 epochs, batch 2,790, 7,000
epochs; see the [overview](README.md#the-reference-configuration)) and ends with the Delta entry it
became or the reason it was parked. Predictions are directions for validation accuracy and 350k
feasibility; none is a result. Sources were read on 26 and 27 September 2026, and cards say when
only an abstract was read.

### R01: the reference optimizer (Adam, framework defaults)

**Mechanism.** Keras Adam with no arguments: β₁ 0.9, β₂ 0.999, ε 1e-7, no weight decay, no
gradient clipping. **Why here.** Adam's per-parameter second moment rescales an STE gradient that
is already a biased proxy for the sign. Unclipped Adam on latent weights with an STE is the
setting that other binary pipelines stabilize with clipping or normalization, so instability
would show up here first. **Source.** Kingma and Ba, "Adam: A Method for Stochastic
Optimization", arXiv:1412.6980, https://arxiv.org/abs/1412.6980. No published number applies.
**Hazards.** Without clipping, one divergent step can corrupt the latents. **Delta:** this is the
base, not an entry. Its partner is R07.

### R02: the reference schedule (cosine restarts, undecayed peak)

**Mechanism.** Cosine decay from 3e-3 to a 1e-6 floor over each 500-epoch cycle, stepped per
epoch, with every cycle the same length (t_mul = 1) and every restart back at the full peak
(m_mul = 1): 14 cycles in 7,000 epochs. **Why here.** A ±1 loss surface is a union of flat cells.
A full-strength restart late in training can push weights across cell boundaries that a decaying
schedule would have frozen, which is either useful exploration or late instability. **Source.**
Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts",
arXiv:1608.03983, https://arxiv.org/abs/1608.03983 (abstract level: 3.14 % and 16.21 % test error
on CIFAR-10 and CIFAR-100; image classification, does not transfer). **Prediction.** Seed-to-seed
spread is larger at readouts just after a restart than mid-cycle. **Delta:** the base; its
variants are R03, R04 and M032.

### R03: restart-period and peak-decay variants

**Mechanism.** Two knobs on R02: t_mul > 1 lengthens each successive cycle, and m_mul < 1 decays
the peak at each restart. **Why here.** If undamped restarts destabilize late training, m_mul < 1
keeps the early exploration and damps the late risk. **Source.** Loshchilov and Hutter,
arXiv:1608.03983, which defines T_mult and the restart decay (no per-variant table read).
**Change.** Two arguments of the schedule function. **Prediction.** m_mul = 0.85 raises 350k
feasibility, with fewer late divergences and little accuracy cost. **Delta entry:** M031 (m_mul
0.85, screened at H = 1,500 because cycle 1 is identical to the base).

### R04: linear warm-up

**Mechanism.** Ramp the LR linearly from near zero to the peak over the first k epochs. The
reference starts every cycle, including the first, at the peak; the project's earlier recipe warms
up for 1 epoch. **Why here.** Early STE gradients are the noisiest, because latents start random.
**Source.** Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia and He, "Accurate,
Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv:1706.02677,
https://arxiv.org/abs/1706.02677, section 5.1 and Table 1. **Published effect.** Table 1
(ResNet-50 on ImageNet, top-1 validation error, mean ± sd over 5 trials): batch 256 without
warm-up 23.60 ± 0.12 %; batch 8k with gradual warm-up 23.74 ± 0.09 %; batch 8k with no warm-up
24.84 ± 0.37 %; batch 8k with constant warm-up 25.88 ± 0.56 %, worse than none. Full precision; what
carries over is that gradual warm-up, not any warm-up, closes the gap. **Prediction.** Fewer early
collapses; accuracy flat. **Delta entry:** M030 (10 epochs).

### R05: one-cycle schedule

**Mechanism.** One rise and fall of the LR over the whole run instead of repeated restarts.
**Source.** Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning
Rates", arXiv:1708.07120, https://arxiv.org/abs/1708.07120 (full-precision CIFAR and ImageNet; no
number transfers). **Hazards.** One large-LR excursion and no restart to recover a bad one; with
Adam, the momentum cycle would have to become a β₁ cycle. **Status:** parked. At the screen
horizon it reduces to a warm-up (M030) plus one reference cycle; M032 (one cosine over the whole
horizon) covers the no-restart extreme.

### R06: peak LR and the linear scaling rule

**Mechanism.** Scale the peak LR linearly with batch size, with a warm-up to avoid early
divergence. **Why here.** Earlier binary runs on the archived recipe (batch 256) collapsed within
the first epochs at a peak LR of 2e-4 or more, while the reference peaks at 3e-3 with about 11 times
fewer steps per epoch. Lower peaks locate the stability edge. **Source.** Goyal et al.,
arXiv:1706.02677, section 2.1, the linear scaling rule η = 0.1·kn/256; Table 1 as in R04.
**Prediction.** Lower peaks give fewer divergences, and accuracy flat or up if the base is
unstable, down if it is stable. **Delta entries:** M028 (peak 1e-3) and M029 (peak 3e-4). Batch
256 with a rescaled LR is parked: about 10 times the steps per epoch.

### R07: the project's earlier Adam variant (β₂ 0.98, weight decay 0.01, clipvalue 1)

**Mechanism.** Adam with β₁ 0.9, β₂ 0.98, decoupled weight decay 0.01 and per-component gradient
clipping at 1.0, as in [`train.py`](../../code/hgq2/bnhgq2/train.py). The lower β₂ shortens the
second-moment memory, decoupled decay acts on the latents directly, and value clipping bounds every
gradient component (chosen over global-norm clipping, which overflowed float32 in an earlier deep
configuration). **Source.** Loshchilov and Hutter, "Decoupled Weight Decay Regularization",
arXiv:1711.05101, https://arxiv.org/abs/1711.05101 (abstract level); the β₂ and clip values are
project choices, not from a paper. **Prediction.** At least as many feasible seeds as R01, with a
smaller seed spread. **Delta entries:** the three fields are decomposed as a 2 × 2 at 5M: M033
(weight decay alone), M074 (clip and β₂ without decay) and M103 (all three), with the base as the
fourth corner. The same optimizer is the replacement base if the reference proves unstable.

### R08: EMA and SWA of the weights

**Mechanism.** EMA keeps a shadow copy of every weight, w_ema ← α·w_ema + (1 − α)·w, used at
evaluation. SWA averages checkpoints collected late in training. Neither exists in the code
today. **Why here.** Both assume continuous weights: averaging latents gives a non-binary value
that has to be re-thresholded before export, and whether to export the average's sign or average
already-signed checkpoints is a real design choice. **Sources.** For EMA, Polyak and Juditsky
(1992), used through Tarvainen and Valpola, "Mean teachers are better role models",
arXiv:1703.01780, https://arxiv.org/abs/1703.01780. For SWA, Izmailov, Podoprikhin, Garipov, Vetrov
and Wilson, "Averaging Weights Leads to Wider Optima and Better Generalization", arXiv:1803.05407,
https://arxiv.org/abs/1803.05407, whose section 3.4 (Figs. 4 and 5, PreResNet-164 and VGG-16 on
CIFAR-100) shows a wider basin than SGD; no portable number. **Hazards.** Re-thresholding an
averaged latent can silently change which weights are +1 or −1, so the exported model must be the
one evaluated. **Prediction.** A small gain and less checkpoint jitter. **Delta entry:** M034 (EMA
of latents at decay 0.999, re-binarized for evaluation). SWA across cycle ends is parked.

### R09: knowledge distillation (logit, feature, attention map)

**Mechanism.** Train the binary student against a teacher's temperature-scaled outputs as well as
the labels, or match intermediate features or attention maps. **Why here.** BitParT binarizes only
the FFN and head and keeps attention at full precision, an implicit path from a
teacher-precision attention (Rai et al., arXiv:2508.07431). A fully binary student has no such
path, so an external teacher may matter more. **Sources.** Hinton, Vinyals and Dean, "Distilling
the Knowledge in a Neural Network", arXiv:1503.02531, https://arxiv.org/abs/1503.02531 (abstract
level); Zagoruyko and Komodakis, "Paying More Attention to Attention", arXiv:1612.03928,
https://arxiv.org/abs/1612.03928, whose "attention" is a CNN activation map, so the
attention-map variant here borrows the name, not a verified transformer method; for jet tagging,
arXiv:2311.14160, https://arxiv.org/abs/2311.14160, not yet read in full (open question OQ-15).
**Change.** A frozen teacher forward pass and a combined loss (1 − λ)·CE + λ·KD, about 40 to 60
lines, and more for the feature and attention variants. **Hazards.** None at inference; about twice
the cost per step. **Earlier attempt.** A distillation arm ran in the constrained N = 8 ablation on
the earlier recipe ([project README](../../README.md#post-conference-ebop-constrained-training));
Delta retries it on the reference recipe with teachers trained on the student's own split.
**Prediction.** Up, larger at the lower budget and on the classes where the binary gap is largest.
**Delta entries:** M035 (logit, T 2, coefficient 0.5, as in
[the earlier configuration](../../code/hgq2/configs/post_conference_budget350k-knowledge_distillation-w1a8.json)),
M036 (attention map), M063 (an int8 teacher), and the packages M061, M062 and M064 with the warm
start M027. The teachers are trained on the student's own split.

### R10: label smoothing

**Mechanism.** Targets of 1 − ε on the true class and ε/(K − 1) elsewhere. **Sources.** Szegedy,
Vanhoucke, Ioffe, Shlens and Wojna, "Rethinking the Inception Architecture", arXiv:1512.00567,
https://arxiv.org/abs/1512.00567, section 7; Müller, Kornblith and Hinton, "When Does Label
Smoothing Help?", arXiv:1906.02629, https://arxiv.org/abs/1906.02629, whose abstract states that a
teacher trained with label smoothing makes distillation "much less effective". **Status:** parked.
It has no binary mechanism, and a smoothed teacher would weaken R09.

### R11: sharpness-aware minimization

**Mechanism.** Perturb the weights toward the locally worst direction, take the gradient there, and
descend from the original weights. **Why here, and the caveat.** A ±1 weight space has no
continuous neighborhood to be flat in, so SAM would act on the latents, and flatness in latent
space is not shown to imply anything about the thresholded weights. **Sources.** Foret, Kleiner,
Mobahi and Neyshabur, "Sharpness-Aware Minimization for Efficiently Improving Generalization",
arXiv:2010.01412, https://arxiv.org/abs/2010.01412 (Fig. 1: relative error reductions of roughly
0 to 40 % across benchmarks, full precision); Liu, Cai and Zhuang, "Sharpness-aware Quantization
for Deep Neural Networks", arXiv:2111.12273, https://arxiv.org/abs/2111.12273 (abstract: 1.2 %
over AdamW on 4-bit ViT-B/16 and 0.9 % over the previous best on 4-bit ResNet-50, ImageNet top-1;
4-bit, not 1-bit). **Status:** parked: twice the step cost, and the transfer to signs is
unestablished.

### R12: dropout and stochastic depth

**Mechanism.** Randomly zero activations, or skip whole blocks. **Sources.** Srivastava, Hinton,
Krizhevsky, Sutskever and Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from
Overfitting", JMLR 15 (2014); Huang, Sun, Liu, Sedra and Weinberger, "Deep Networks with Stochastic
Depth", arXiv:1603.09382, https://arxiv.org/abs/1603.09382. **Status:** parked. The model has one
block and is limited by capacity rather than overfitting, and zeroed activations interact with the
learned widths in an untested way.

### R13: physics-motivated augmentation

**Mechanism.** Rotations about the jet axis in the η-φ plane, reflections of η and φ, pT-scaled
smearing of positions, and constituent dropout. **Source.** Dillon, Kasieczka, Olischläger, Plehn,
Sorrenson and Vogel, "Symmetries, Safety, and Self-Supervision" (JetCLR), arXiv:2108.04253,
https://arxiv.org/abs/2108.04253, which defines these augmentations for contrastive pre-training
on top tagging; no supervised number transfers. **Change.** A per-batch transform on training data
only, after the standardization statistics are fixed. **Earlier attempts.** None in this project.
**Prediction.** Reflections, which are exact symmetries of the relative coordinates, narrow the
seed spread and hold or raise the mean; smearing and dropout are approximate and less certain.
**Delta entry:** M037 (η and φ reflection), also factor E of FF1. Rotation, smearing and
constituent dropout are parked until the exact reflection is read.

### R14: pT sample reweighting

**Mechanism.** Per-class weights that flatten the training pT spectrum. **Status:** rejected here.
It has no binary mechanism, the reference recipe uses no sample weights, and an earlier N = 8
study of this reweighting found that it lowered accuracy.

## Considered and left out

A curriculum over the EBOPs target belongs to family Q; the boundary is the target schedule, not
the optimizer. Ensembling is excluded because N binary models cost N times the logic inside one
fixed trigger envelope, before any accuracy question arises. Focal and class-balanced losses are a
form of reweighting and are left out with R14. SGD and Nesterov momentum are out of scope: neither
the reference nor the project's own recipe uses SGD.
