# Family P: inputs, data and the L1 trigger context

Method cards for what the model sees. Each card is a change to the reference input pipeline
(N = 64 leading-pT constituents; features pT, η_rel, φ_rel; constituents with pT < 2 GeV zeroed
before standardization; a 90/10 split of the 620,000-jet training archive; no sample or class
weights) and ends with the Delta entry it became or the reason it was parked. The dataset is the
[HLS4ML LHC Jet dataset, 150-particle release](https://zenodo.org/records/3602260) (M. Pierini,
J. M. Duarte, N. Tran and M. Freytsis, Zenodo record 3602260, derived from Coleman et al.,
arXiv:1709.08705). Each jet stores up to 150 zero-padded constituents with 16 features; the median
jet in the held-out archive has 46 non-zero constituents (measured on the HDF5 files). Sources
were read on 26 and 27 September 2026.

### P01: constituent count N, as a deliberate cross-N comparison

**Mechanism.** N sets the sequence length. At a fixed budget, it trades precision per token against
context. **Why here.** HGQ weights can buy accuracy back with bits per token; a binary core cannot,
so if accuracy keeps rising with N at matched EBOPs, the binary core's headroom is in tokens.
**Source.** Laatu et al., "Sub-microsecond Transformers for Jet Tagging on FPGAs",
arXiv:2510.24784, https://arxiv.org/abs/2510.24784, sections 2.1 and 3, Table 1 and Fig. 1.
**Published effect.** Table 1, top-1 accuracy on the 260,000-jet test set, one model per point, all
at a 350,000-EBOP target: multi-head attention 66.3, 72.3, 77.0 and 77.9 % at N = 8, 16, 32 and 64
(the N = 64 model had collapsed into a Deep Set, section 3), Linformer 66.3, 72.8, 78.4 and 79.8 %.
LUT usage of the attention model falls from 279k to 180k to 47k from N = 16 to 64, because the
budget is spent on bits, not tokens; the Fig. 1 caption attributes the drop at 64 particles to
"the fixed resource budget we enforced during training". No binary number exists. **Hazards.**
Attention MACs grow with N², and a cross-N comparison is confounded with the gate (P02) and the
standardization statistics (P06), which shift with the padding fraction. **Earlier attempts.**
Matched N = 8 and N = 64 runs exist only on a short 50-epoch schedule
([screen](../current-work/CONSTITUENT_SCREEN_20260923.md)); N = 32 at 350k under the earlier
quantizer set was never feasible. **Prediction.** Feasible at 350k at N = 32 with nearly every
channel alive (h 0.993, computed from layer shapes); accuracy against N = 64 undetermined.
**Delta entries:** M009 (N = 32) and M058 (N = 32 with two heads, the only architecture whose
1-bit-alive floor fits under 350k). Every result is labelled "crosses N". N = 16 is parked.

### P02: the pT gate

**Mechanism.** After taking the top N constituents by pT, every constituent below the threshold is
zeroed in all three features before the standardization statistics are computed, so gated slots
pull the mean and sd toward zero along with real low-pT constituents. **Why here.** A ±1-weight
network with no normalization has no learnable per-channel scale to absorb a shift in input
statistics, and moving the gate moves what "zero" means to every downstream binary layer.
**Source.** The gate is part of the reference configuration; Laatu et al. do not mention one. They
state only that particles are sorted by pT and carry "pT, η, and ϕ" (section 2.1). Whether their
Table 1 used absolute or jet-relative angles is not stated (open question OQ-14). **Hazards.** None
for hardware; the gate is preprocessing before the FPGA. **Prediction.** Sign open: more real
low-pT constituents against more noise in the statistics. **Delta entry:** M038 (gate off), and
the gate × PE × mask cells M075 to M078. Other thresholds (1 or 3 GeV) are not in the catalogue.

### P03: masking gated and padded keys

**Mechanism.** Today "below threshold" and "absent" are treated alike: multiplied by zero, then
standardized. An additive mask on those key positions before the softmax removes them from the
attention sum instead of asking the network to learn that one standardized vector means "ignore
me". **Why here.** With ±1 weights the network cannot cheaply express "if input equals the pad
value, output 0", and the pad vector moves whenever N, the gate or the statistics change. Masking
also stops padding from inflating the entropy diagnostic. **Source.** Standard transformer padding
practice; no jet-tagging-specific source. **Hazards.** hls4ml's table softmax was not built for a
runtime mask; a multiplexer per key position costs LUTs and may break the folding of identical
token instances (open question OQ-13). **Prediction.** Up, with the entropy diagnostic no longer
inflated. **Delta entries:** M039, M076 to M080.

### P04: constituent ordering

**Mechanism.** Order constituents by ΔR to the jet axis, or leave them unordered, instead of by pT.
**Why here.** With a learned positional table the network ties a position to a pT rank, and a
geometric order may be easier to represent. **Source.** Odagiu et al., "Ultrafast jet
classification at the HL-LHC", arXiv:2402.01876, https://arxiv.org/abs/2402.01876, use
permutation-invariant models, so there is no ordering ablation to borrow. **Status:** parked;
predicted to matter at N ≤ 16, not at N = 64, and moot without a positional table.

### P05: the input feature set

**Mechanism.** The dataset stores 16 constituent features (px, py, pz, e, erel, pt, ptrel, eta,
etarel, etarot, phi, phirel, phirot, deltaR, costheta, costhetarel). The reference uses three.
This card appends two quantities derived from those three: log pT and ΔR to the jet axis. **Why
here.** More input features mean more first-layer directions (M1 in [theory.md](theory.md)), the
cheapest place to add capacity if three features starve the binary core. **What an L1 candidate
carries.** A CMS L1 tau algorithm built on Correlator Layer-2 PUPPI candidates uses "the 10 highest
transverse-momentum particles and particle ID ... within a cone of ΔR < 0.4" (the
L1Phase2NNPuppiTau CMS TWiki, from a search digest, not read in full), so particle identity exists
on real hardware and is not in this dataset. No feature set here is L1-realistic in content, only
in count and kinematics; the three-feature convention comes from Odagiu et al., arXiv:2402.01876.
**Hazards.** The derived features are computed off-model, so HGQ2 never bills them, and on a chip
the square roots or tables they need are uncosted. The input layer's EBOPs grow linearly with the
feature count. **Prediction.** Up, from more expressible first-layer directions. **Delta entries:**
M040 (five inputs), M067 and M080 in packages, and factor B of FF1; every result is labelled
"crosses input set" and "not a hardware candidate".

### P06: standardization over real slots only

**Mechanism.** The standardization mean and sd are computed over the gated, padded training array,
including zeroed and absent slots. This card computes them over real slots only. **Why here.**
Absent and gated slots are a large share of the array, so they shrink the sd and pull the mean
toward zero, which sets the operating point of the first binary layer. **Source.** The reference
practice is described in [`data.py`](../../code/hgq2/bnhgq2/data.py) (`input_std_stats`). No
ablation exists. **Hazards.** None for hardware: the statistics become fixed constants at export
either way, but they must match the population the deployed trigger delivers. **Prediction.** Up,
with the first binary layer's operating point no longer set by padding. **Delta entries:** M041,
M079 (with masking) and M080.

### P07: input bit width

**Mechanism.** The fixed-point width of the input to the first binary layer, treated as its own
axis. **Source.** BitNet a4.8 (arXiv:2411.04965) pushes activations to 4 bits in an LLM, a scale
mismatch here. **Status:** parked. Under the reference quantizers the input width is learned with
every other width, and the per-field precision of real L1 candidates is not yet sourced (open
question OQ-09).

### P08: split fraction (90/10 against 80/20)

**Mechanism.** The reference uses 90/10 (62,000 validation jets); the project's earlier convention
is 80/20 (124,000). More training data against a noisier selection signal. **Hazards.** Changing the
split after seeing results would violate the selection rule, and it changes the validation set,
which is the screen readout. **Status:** parked; a confirm-only question if it is asked at all.

### P09: η and φ reflections

**Mechanism.** Flip the signs of η_rel and φ_rel per jet at training time. The relative
coordinates are covariant under reflections that the physics does not distinguish. **Why here.**
Augmentation costs nothing at inference and is one of the few candidate fixes for a wide seed
spread with zero hardware risk. **Source.** A symmetry argument; JetCLR (Dillon et al.,
arXiv:2108.04253) defines related augmentations for pre-training. **Hazards.** It must touch only
training batches, never validation or held-out data, and must not double-rotate the dataset's own
rotated columns. **Prediction.** A narrower seed spread; the mean flat or up. **Delta entries:**
M037 and factor E of FF1.

### P10: the class set

The task is fixed at five classes (g, q, W, Z, t) with macro one-vs-rest AUC. Merging W and Z, or
collapsing to quark/gluon against boosted objects, is a different task whose AUC cannot sit in a
five-class table. **Status:** rejected.

### P11: pT sample reweighting

**Status:** rejected here. It has no binary mechanism, the reference uses no sample weights, and an
earlier N = 8 study of this reweighting found that it lowered accuracy. The same card appears as
R14 in [recipe.md](recipe.md).

### P12: the figure of merit

**Mechanism.** AUC integrates over every threshold and accuracy uses the argmax, while an L1
trigger runs each tagger at one operating point set by its rate budget. The physics-relevant
number is signal efficiency at a fixed mistag rate, or background rejection at a fixed efficiency.
**Source.** CMS, "Reconstructing jets in the Phase-2 upgrade of the CMS Level-1 Trigger with a
seeded cone algorithm", arXiv:2310.08062, https://arxiv.org/abs/2310.08062: a total L1 decision
latency of about 12.5 μs, about 5 μs of it for track, calorimeter and muon reconstruction and about
1 μs for each Correlator layer; their seeded-cone jets close in 720 ns plus 138 ns of serial
transmission inside the Layer-2 budget. In BitParT (Rai et al., arXiv:2508.07431), AUC moved by
0.0006 while background rejection at 50 % efficiency fell by about 15 %. **Status:** converted into
a reporting rule (Z09): rejection at signal efficiency 0.5 and per-class efficiency at fixed
mistag are reported beside accuracy and AUC at every confirm. **Prediction.** For at least one pair
of Delta entries the AUC ranking and the working-point ranking will disagree.

## Considered and left out

Jet-image inputs are a different architecture family. Using all 150 constituents would need its
own budget reasoning and is almost certainly infeasible at L1 latency. Track quality and z0, which
real charged PUPPI candidates carry, are not stored in this dataset, the clearest reason no input
configuration here is fully L1-realistic. A different dataset (for example JetClass,
arXiv:2202.03772) would change the whole study. Domain adaptation does not apply to a
simulation-only benchmark.
