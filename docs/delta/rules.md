# Delta rules

These rules were fixed before any Delta configuration was trained. Each wave copies its entries
from [`catalogue/delta.json`](catalogue/delta.json) and its rules from this file. A wave may
tighten a rule in its own pre-registration; it may not loosen one after launch. Every number in
this file is either a design choice, shape arithmetic from
[`delta_spec.py`](catalogue/delta_spec.py), or a power constant from
[`screen_power.py`](catalogue/screen_power.py). None is a training result.

## Amendments from the wave-2 pre-registration (2026-09-28)

 The frozen wave-2 pre-registration amends the rules below for wave 2, dated before its launch. §5 is not rewritten; where it and this list differ for wave 2, this list and the wave-2 pre-registration govern. No rule is added here. The wave-2 pre-registration, whose sections the last column names, will be published with the wave-2 results.

| rule | before | wave 2 (after) | wave-2 section |
| --- | --- | --- | --- |
| Primary ranking statistic | lower 80 % bound of g | median of the paired per-seed gaps g_s, with [min, max]; mean g (family-pooled interval), the low-mode run count and the per-cell lower 80 % bound beside it. Mean g was chosen 2026-09-27 and replaced by median g 2026-09-28 | Selection rule, Ranking |
| Seed count | n ∈ {4, 6, 8} from √2 · the reference configuration's epoch-500 seed sd; ranking mode at n = 4 if none qualifies | n = 4 fixed in both families, ranking mode, lists labelled by s_int (next row); the seed-rule input is the replicas' own sd; significance mode not armed. If 1.0 pt ≤ s_int ≤ T, the top 16 (5M) / top 6 (350k) cells run seeds 5-8 | Seeds; Appendix A |
| Label rule | "ranked, not advanced" | "ranked" if the cell × seed interaction sd s_int ≤ T, else "descriptive"; T = 1.2 pt at 5M, 2.1 pt at 350k | Selection rule, Label |
| Family test | none in ranking mode | Dunnett-type many-to-one max-t of d = cell − placebo, each cell's own paired sd, one-sided α 0.10, critical value simulated for the family's m and n; a "no" is not evidence of absence | Selection rule, Family test |
| Compute pause | no seed-sd pause; base-stability pause | none: the n = 4 design runs whatever the replica sd reads; the base-stability pause is kept | Seeds; Controls, Base stability |
| Confirm cap | at most 12 cells, ordered by lower 80 % bound | unchanged at 12, ordered by median g | Selection rule, Cap |
| Control | reference-configuration snapshot at the same seed; replica on a drift trigger | the in-wave drift replica at the same seed is always the primary control; replica − reference configuration is descriptive only | Controls and pairing |
| Drift replicas | base arms at seeds 1-n | the d24h2 (350k) and d32h4 (5M) replicas at seeds 1-8 (seeds above n stop at epoch 500); the d32h4 350k replica at seeds 1-n | Arms, Drift replicas |
| Placebo | none | one placebo per target: the replica config with identity keys changed, seeds 1-4, 500 epochs; family-test reference only, not ranked | Arms, Placebo cells |
| Non-selecting companions | winner's-curse note only | last-epoch and split-half companions per run, each with g, interval, rank and Kendall τ; rank-move flag; neither changes the list | Selection rule, Winner's curse |
| Memory-growth launch gate | none | a host-memory growth canary on one d24h2 and one d32h4 cell, threshold in the wave-2 pre-registration; the wave waits if it fails | Prerequisites and launch gate |
| Launch condition | reference-configuration epoch-500 snapshots at seeds 1-n exist | the reference configuration's pilot epoch-500 validation readout | Prerequisites and launch gate |
| Ranked-list membership | every tested cell of the wave × target | 500-epoch paired cells only (at most 11 at 350k, 35 at 5M); M015, M031, M032 in a separate longer-horizon list, unpaired cells apart | Selection rule, Lists |
| Eligible epochs | not stated | epochs with a traced EBOPs value only (one in ten) | Selection rule |
| 350k base | d24h2 | unchanged (d24h2) | Arms, Base arms |

## Reading an entry

Each entry has an ID (M001 to M103), a family, the method cards it comes from (listed in the
family notes), the mechanisms it tests (M1 to M7 in [theory.md](theory.md)), the exact change
from its base configuration, a code tier, what earlier attempts looked like, its screen and
confirm targets, the screen horizon H, a pairing rule, a one-line directional prediction, the
mechanism diagnostic reported beside accuracy, and its static floors. The base is named per
target in the catalogue's `base` field: d24h2 at 350k for the 350k accuracy cells and FF2,
d32h4 at 350k for the feasibility reference and the 1.4M rung, and d32h4 at 5M for every 5M cell.

**Code tiers.** T0 is config-only on a key the constrained runner reads today (`arch.n_heads`,
`arch.n_part`, `arch.d_model`, `arch.ffn_dim`, `arch.n_layers`, `arch.pos_enc`,
`train.ebops.pid.*`, `experiment.recovery_after_epochs` and the optimizer fields). T0a is
config-only once the reference configuration's code is published: its keys (`train.lr` as the
peak, `train.lr_cycle_epochs`, `train.optimizer`, `arch.pt_gate_gev`, `quant.act_f0`) are
placeholder names until then. T1 is a named patch of about 80 lines or less. T2 is a new module:
an attention kind, a model body, an optimizer or a weight scheme. A combination takes the highest
tier of its parts. Optimizer changes are stated on the explicit optimizer path, so a
`train.weight_decay` can never be silently ignored.

**Pairing.** An entry whose tensors keep their shapes is *paired*: the same initialization at
seed s (matched by path and shape) and the same data order, analysed with a paired t test on
k − 1 degrees of freedom, beside the sign count and the per-pair correlation. An entry that
reshapes some tensors is *paired-if-hash*: paired only if a kernel-hash comparison, recorded
before any result, shows every unchanged-shape tensor identical to the base at seed s, otherwise
Welch. An entry that replaces the body, changes the inputs (N or the feature set) or starts from
a warm checkpoint is *unpaired* (Welch). Delta pods run on a different code version, and often a
different GPU class, from the reference runs, so same-seed pairs are never bit-identical; the
per-pair correlation is always reported and each wave carries a drift replica.

## Static floors

EBOPs, as HGQ2 counts them, charge products Σ b_i·b_j and explicit additions Σ max(b_k, b_l)
([Sun et al., arXiv:2405.00645](https://arxiv.org/abs/2405.00645), section 3.3, Eq. 11). With
binary weights (b_w = 1) the static floors of an architecture follow from its shapes. For N
tokens, d_model d, H heads, FFN width f, one block and three input features:

| term | EBOPs at activation width b |
| --- | --- |
| dense layers (input projection, Q, K, V, output, FFN, head) | N · fan_in · fan_out · b · b_w |
| Q·K and A·V | T · S · d · b · b, with T = S = N (S = k for Linformer) |
| softmax | 16·H·T·S + 4·(H·T·S − H·T) + H·T·S |

The softmax row is the exp × inverse product and the row sum with both tables at their 4-bit
minimum, plus one bit per score entry for the exponential's input, whose quantizer saturates in
hgq2 0.1.9 and so never reaches 0 bits. A small lookup-table term of at most a few tens of EBOPs
is not modelled. The **0-bit floor** puts every activation channel at 0 bits, which the reference
quantizers allow, so only the softmax remains. The **1-bit-alive floor** puts every channel at
1 bit. The headroom fraction is h = (target − floor0) / (floor1 − floor0), clipped at 1: it says
how far above "everything pruned" a feasible model at that target can sit.

Substitutions, each checked by an assertion in `delta_spec.py`:

- **d32h4** (H = 4, T = S = 64): softmax 262,144 + 64,512 + 16,384 = 343,040. Dense layers and
  head at 1 bit: 64·3·32 + 4·64·32·32 + 2·64·32·32 + 32·32 + 32·5 = 400,544. Q·K and A·V at
  1 bit: 2 · 64·64·32 = 262,144. So floor0 = 343,040 and floor1 = 1,005,728, and h at 350k is
  0.011.
- **d24h2** (H = 2): softmax 171,520, dense and head 251,064, Q·K and A·V 196,608, so floor0 =
  171,520, floor1 = 619,192 and h = 0.399.
- **Linformer, k = 8** on d32h4: 2,048 score entries give a softmax of 32,768 + 7,168 + 2,048 =
  41,984. Adding the dense layers (400,544), Q·K and A·V over 8 keys (32,768) and the two binary
  sequence projections (32,768) gives floor1 = 508,064.
- **N = 32** on d32h4: softmax 85,504, dense and head 200,864, Q·K and A·V 65,536, so floor1 =
  351,904, above 350,000 by 1,904. No single fits every channel at 1 bit under 350k; the package
  M058 (N = 32 with 2 heads) does, at 309,152.
- **Baselines** price the weight bits: int8 weights everywhere (M048) multiply the dense and head
  terms by 8, giving floor1 = 3,809,536; an int8 input projection only (M050) adds 7 · 6,144,
  giving 1,048,736. Ternary (M047) and learned weight widths (M049) have no computable floor1.
  At 0-bit activations weight bits cost nothing, so all four share the d32h4 floor0.
- **Limiting case:** with no attention (H = 0) every softmax term vanishes.

The full table by architecture is at the end of [methods.md](methods.md). Two consequences
shape the whole design.

d32h4 at 350k is a near-floor configuration. A feasible checkpoint keeps at most 6,960 EBOPs
outside the softmax, while the input projection alone costs 6,144 at 1 bit and the head 1,184.
Wq, Wk and Wv cost 65,536 each at 1 bit, so a feasible d32h4 at 350k has pruned its attention by
construction and may behave like a Deep Set. It is therefore a feasibility reference, never an
accuracy base, and every 350k accuracy screen runs on d24h2, whose h is 0.399.

These floors are shape arithmetic, not measurements. The wave-1 trace with HGQ2's own counter
replaces them. Until then no floor may be quoted as a property of a trained model.

## The 350k cells

**Static feasibility.** A 350k cell runs only if its 0-bit floor is below 350,000. Entries whose
purpose is floor reduction (M001 to M006, M009 and the floor crosses) are dropped at 350k if they
fail; the others keep their 5M cell. Already applied: M007 (floor0 474,112) and M044 (686,080)
run at 5M only.

**Roles.** The catalogue's `role_350k` field gives each 350k cell one of four roles.

| role | which entries | how it is read |
| --- | --- | --- |
| accuracy screen on d24h2 | changes that leave d32h4 near-floor (h < 0.10): widths, controller, inputs, head, FFN 64, d_model 16, the Q/K floor with 2 heads | the change is applied to d24h2 and paired with it; G0 to G3 apply |
| feasibility probe on d24h2 | M047, M048, M049: near-floor or not computable on d24h2 as well | G2 counts, EBOPs split and attention state only; no accuracy reading |
| floor family | M001 to M006, M009, M051 to M054, M056 to M060 | runs on its own d32h4-derived architecture; feasibility by G2 against d32h4 at 350k, accuracy against d24h2 by Welch, labelled cross-architecture |
| FF2 cell | M096 to M102 | built on d24h2 and paired with it |

The floor-family cells are never compared with the accuracy of d32h4 at 350k, which keeps under
7,000 EBOPs outside its softmax. Because d24h2 has no positional encoding, the no-PE cells M075,
M077 and M078 would duplicate other cells at 350k and are dropped there; at 350k the gate × PE ×
mask cube becomes the gate × mask square (d24h2, M038, M039, M076), and the full 2³ remains at 5M.

**If d24h2 fails at 350k** (fewer than ⌈3n/4⌉ of seeds 1 to n feasible at epoch H): the accuracy
screens and probes lose their control and keep only their 5M cell; the floor family is read for
feasibility alone; M013, which exists only at 350k, runs as a feasibility probe on d32h4 at 350k;
and FF2 moves to d32h4 at 5M with the in-wave replica as its all-low cell. If no 350k base
exists at all, every 350k accuracy cell and probe is dropped and the floor family still runs as
feasibility probes against the number of d32h4 seeds feasible at epoch H (zero if none).

**If the reference reverts to the earlier quantizer set** (saturating channels that never go
below 1 bit, a fixed 10-bit softmax output, fixed tables): every 350k cell becomes statically
infeasible, every screen runs at 5M against an in-wave replica of the earlier-quantizer d32h4,
the floor-reduction singles and crosses and M010 are dropped, and a fixed 4-bit softmax-output
entry parked for that branch is reinstated.

**Reporting.** Every 350k result states its architecture's floor, h and the attention state
(Q/K and V 0-bit fractions, entropy over log n_valid). A feasible near-floor checkpoint is
labelled "pruned by construction".

## Targets

An attention lever read at one target cannot be interpreted: at 5M attention is affordable, at
350k it is priced out ([theory.md](theory.md#5-interactions-which-combinations-to-cross), item 4).

| targets | entries |
| --- | --- |
| 350k and 5M | every floor entry, the width entries M011, M012, M015 and M016, the architecture entries M042, M043 and M045, the input entries that touch tokens or the first layer (M038 to M040), all four baselines, and packages built only from two-target singles |
| 350k only | M013 (PID gains) and the confirm-only M014 (open-loop β): their purpose is reaching the budget from a hard start; FF2, because its base exists only at 350k |
| 5M at screen, 5M and 350k at confirm | training-only levers (B, R, distillation, augmentation, real-slot standardization) and the no-PE single M046 |
| 5M only | M007 and M044, statically infeasible at 350k |
| 1.4M | M010, a ladder rung between d32h4 at 350k and at 5M with no same-target control, so G3 never applies |

Training-only levers screen at 5M because their mechanisms act on training dynamics, which need
live channels; screening them at 350k on d24h2 is the open alternative listed under
[choices still open](#choices-still-open).

## Screen and confirm

| | screen | confirm |
| --- | --- | --- |
| schedule | the reference recipe for H epochs: H = 500 (one cosine cycle, ending at the 1e-6 floor) unless the treatment starts later: M015 H = 1,000 (freeze at 500), M031 H = 1,500 (first decayed peak at epoch 501), M032 and its packages H = 2,000 (one cosine over 2,000) | the full 7,000 epochs (M032: one cosine over 7,000) |
| seeds | n per cell, seeds 1 to n, n ∈ {4, 6, 8} set before the wave by the rule below | 8 per cell |
| readout | validation only (62,000 jets): top-1 accuracy (primary), macro AUC, feasible count, divergence count, EBOPs and accumulator EBOPs, the mechanism diagnostic, all at the best-feasible-as-of-H snapshot. Never quoted outside the screen | selection on validation accuracy among feasible checkpoints, tie-break validation AUC, then lower EBOPs, then earlier epoch; an AUC-selected copy as a sensitivity check; held-out evaluation (260,000 jets) once, after the last epoch |
| control | the reference snapshot at epoch H, seeds 1 to n, or the in-wave replica when the drift trigger fires | the reference at the same target, seeds 1 to 8, last epoch |
| multiplicity | Benjamini-Hochberg, q = 0.10, within each wave × target; none in ranking mode | Holm, α = 0.05, across the confirm wave |
| reported beside | | per-class AUC, background rejection at signal efficiency 0.5, per-class signal efficiency at fixed mistag, attention state, EBOPs and accumulator EBOPs, validation minus held-out gap |

**Validity.** A checkpoint counts as feasible only if (a) its EBOPs is at or below the target,
(b) its EBOPs exceeds the architecture's 0-bit floor as traced by HGQ2's counter, and (c) its
validation accuracy exceeds p_maj + 5·√(p_maj(1 − p_maj)/62,000), with p_maj the majority-class
fraction of the gated validation labels. A checkpoint meeting (a) but not (b) or (c) is
"feasible, degenerate": counted separately and never carrying an accuracy. Test (b) needs each
architecture's traced floor, so the wave-1 trace gates those cells at both targets. The collapse
rule (below) and test (c) are separate; if (c) lies above 0.25, the collapse level is raised to it.

## Seed count

The advance rule has to resolve its own minimum detectable effect, g₀ = 0.3 points of validation
accuracy, under the multiplicity actually used. Gate (i) of the rule is a one-sided paired t test
at α_eff = q/m, the threshold of the first Benjamini-Hochberg discovery (conservative, since the
j-th discovery uses j·q/m); gate (ii) asks that the mean gap be at least g₀/2. k_joint(n, α) is
the gap, in units of the paired sd, at which (i) and (ii) together have 80 % power (seeded Monte
Carlo, 200,000 draws). The family size m counts the (entry, target) cells with a G3 test; the
baselines, the feasibility probes and M010 take none.

| family | m | α_eff | k_joint at n = 4 / 6 / 8 | paired-sd ceiling for 0.3 pt at n = 4 / 6 / 8 |
| --- | ---: | ---: | --- | --- |
| unadjusted (reference) | – | 0.05 | 1.68 / 1.20 / 0.99 | 0.18 / 0.25 / 0.30 pt |
| wave 2, 5M (singles) | 43 | 0.0023 | 4.83 / 2.50 / 1.83 | 0.062 / 0.120 / 0.164 pt |
| wave 2, 350k (singles) | 19 | 0.0053 | 3.66 / 2.09 / 1.58 | 0.082 / 0.144 / 0.190 pt |
| wave 3, 5M (packages) | 31 | 0.0032 | 4.33 / 2.33 / 1.73 | 0.069 / 0.129 / 0.174 pt |
| wave 3, 350k (packages) | 11 | 0.0091 | 3.04 / 1.84 / 1.43 | 0.099 / 0.163 / 0.210 pt |

The factorials have their own constants (per-run σ, BH within the block's effects): FF1 needs
σ ≤ 0.35 / 0.43 / 0.51 pt at n = 4 / 6 / 8, and FF2 needs σ ≤ 0.26 / 0.33 / 0.39 pt.

The rule:

1. sd_plan per target is √2 times the sd of epoch-500 validation accuracy of the base (d32h4 at
   5M for the 5M family, d24h2 at 350k for the 350k family) over seeds 1 to 8, feasible and
   non-degenerate seeds only. √2 is the bound for zero pair correlation. Once the drift replicas
   are read, sd_plan becomes max(sd_plan, sd_null). A measured paired sd can tighten the rule,
   never loosen it.
2. n is the smallest of 4, 6, 8 with k_joint(n, α_eff) · sd_plan ≤ 0.3 pt, per wave and target.
   Wave 3 never uses more seeds than wave 2 at the same target, because packages need their
   singles on the same seeds.
3. If no n qualifies, the wave runs at n = 4 as a declared ranking: cells are listed by the lower
   80 % bound of their gap, with no advance claims. The alternative, decided before launch, is
   n = 8 with the target raised to g_res = k_joint(8, α_eff) · sd_plan and gate (ii) at g_res/2,
   or a smaller BH family.

For 8 seeds to qualify at wave 2, 5M, the epoch-500 sd has to be at most 0.116 pt. At an
illustrative sd of 0.6 pt, g_res(8) would be 1.55 pt. Ranking mode is the likely outcome, and it
is pre-registered so that the selection it produces is a declared one. The confirm tier uses the
same arithmetic with k_t(8, 0.05/n_c) and no gate (ii); a confirm gap inside ±g_res,confirm is
"unresolved", not "flat". The cheap version (3 seeds) always runs as a ranking: k_joint(3) is
10.73 at the wave-2 5M α_eff.

The 0.3-point target is a design choice. At the roughly 80 % accuracy that Laatu et al. report
at N = 64 (Table 1), the binomial SE of one checkpoint's accuracy on 62,000 jets is
√(0.8 · 0.2 / 62,000) = 0.16 pt; 0.3 pt is about twice that, and gate (ii) at 0.15 pt about one
SE. Seed variance is handled by the rule above, not by the target.

## Controls

**Drift replica.** Every screen wave re-runs its bases at the Delta code version on Delta pods,
seeds 1 to n, each to the longest horizon that pairs with it; snapshots every 500 epochs give
every shorter horizon from the same run.

| wave | replica | horizon | serves |
| --- | --- | ---: | --- |
| 2 | d24h2 at 350k | 1,000 | the 350k accuracy cells, M015 |
| 2 | d32h4 at 350k | 500 | the floor-family feasibility count |
| 2 | d32h4 at 5M | 2,000 | the 5M cells, M031, M032 |
| 3 | d24h2 at 350k | 500 | the 350k packages, the FF2 all-low cell |
| 3 | d32h4 at 350k | 500 | the floor crosses' feasibility count |
| 3 | d32h4 at 5M | 2,000 | the 5M packages, M068, M069, M071 to M073, the FF1 all-low cell |

For each seed, g_rep = replica − reference snapshot. If the 95 % interval of g_rep excludes 0,
or |mean g_rep| > 0.3 pt, the wave's primary pairing switches to the replica at every horizon,
and both are reported. That separates code version, GPU class and date from the method.

**Replicas first.** The replicas take the first pods of a wave, and sd_null (the paired sd of
g_rep) is read at their epoch-500 snapshot before any other entry of the wave starts. If the
chosen n no longer qualifies, the wave switches to ranking mode before its entries launch.

**Base stability.** A base with more than a quarter of its seeds diverged, collapsed or
early-peak degraded by epoch H (G1 below; 2 or more of 4) is not a valid control, and the wave
pauses. The default replacement at 350k is d24h2 with the project's earlier optimizer (β₂ 0.98,
weight decay 0.01, clipvalue 1) on the same schedule; at 5M it is M103, the same optimizer on
d32h4, provided its own seeds are stable.

## Screen advance rule

For seeds s usable in both the entry and the base, g_s = acc_val(entry, s) − acc_val(base, s),
both at the best-feasible-as-of-H snapshot. The gates apply in order.

**G0, the treatment was measured.** Every entry declares its treatment onset: epoch 1 for most,
501 for M031, and for M015 the logged freeze epoch (the first feasible epoch at or after 500,
which may never come on a near-floor base). A selected checkpoint before onset means "treatment
not measured"; if that holds in more than a quarter of the seeds, the entry is reported as not
measured, neither a null nor an advance. An entry no screen horizon can reach has no screen cell:
M014, whose open-loop schedule spans the full 7,000 epochs, is confirm-only.

**G1, stability.** Divergence is a non-finite loss. Collapse is validation accuracy ≤ 0.25 for 10
consecutive epochs after epoch 20; the run is stopped and recorded. Diverged and collapsed runs
are never relaunched and never dropped. Early-peak degradation is computed from the per-epoch
logs without stopping the run, and needs all three of: the best validation accuracy falls at an
epoch ≤ 20; accuracy then stays at least 5 points below that best for 10 consecutive epochs
within the first 100; and the logged EBOPs over those epochs are still at least half the epoch-1
EBOPs. The third condition separates this from a budget squeeze, where accuracy falls because
the controller removed cost; a squeeze is recorded as "early peak, then squeeze" with its first
feasible epoch. More than a quarter of the seeds failing G1 makes the cell "unstable": reported,
not advanced.

**G2, feasibility.** k_e is the number of feasible, non-degenerate, non-diverged seeds at the
target, with the degenerate count reported beside it. An accuracy reading needs k_e ≥ 3. A
floor-family entry advances on feasibility alone if k_e ≥ 3 and k_e − k_base ≥ 2, where k_base
counts the feasible d32h4 seeds at 350k. The exact McNemar p is reported as a description.

**G3, accuracy.** With at least 3 paired seeds, report the mean g and its 95 % t interval, the
lower 80 % bound, the per-pair correlation, the one-sided paired t p-value (Welch for unpaired
entries), and the sign count with its exact sign-test p (descriptive). In significance mode a
cell advances if (i) its BH-adjusted p ≤ 0.10 within the wave × target family and (ii) the mean
g ≥ g₀/2. In ranking mode nothing advances on significance; cells are ranked by the lower 80 %
bound and labelled "ranked, not advanced". The baselines get G0 to G2 and the G3 report, but no
advance decision.

**Zero advances.** A significance-mode wave in which no cell advances reports "no entry resolved
at g₀ in this family"; that is its result. The ranked list is still passed on, labelled, with a
winner's-curse note, and cells from it may go to confirm as exploratory candidates, where the
8-seed Holm test is the only test.

**Non-inferiority (G3′).** Cost levers read for cost (M023, M024, M025) and floor entries at 5M
pass if the lower 95 % bound exceeds −0.3 pt. A non-inferior M023 goes to a synthesis check, not
to an accuracy confirm.

**Packages** must also beat their best component on the same seeds: mean g(package − best
component) > 0 with at least ⌈3n/4⌉ of n seeds positive (reported, not gated, in ranking mode).
The interaction I = g_package − Σ g_components is reported with its interval and flagged
"synergy" or "redundancy". Where a component has no cell at a target (M007 at 350k), the
decomposition there is one-sided and says so.

**Factorials** are read from their OLS models (below), with BH within each block's effects and
the same thresholds, or in ranking mode if the σ ceiling is not met. A two-factor interaction
that clears BH flags its pair as non-additive, and the "expected additive" claim in
[theory.md](theory.md) is revised for it.

**Mechanism.** An advanced entry whose diagnostic moved against its prediction is advanced but
flagged "mechanism not confirmed".

**Cap.** At most 12 confirm cells per confirm wave, taken from the list ranked by the lower 80 %
bound. Nothing moves into GPU time automatically.

## The fractional factorials

**FF1** is a 2^(5−1) design of resolution V on d32h4 at 5M, H = 500, n seeds per cell. Its
factors are A = EDE STE (M019), B = derived input features (M040, an input-set change by design),
C = logit distillation (M035), D = d_model 16 (M042) and E = η/φ reflection (M037). The generator
is E = −ABCD (I = −ABCDE), so every cell has an odd number of low factors: the fraction holds the
all-low cell, the ten two-high cells and the five four-high cells (M081 to M095 are the 15 new
cells). The all-low cell is the in-wave d32h4 5M replica, not the reference snapshot, because it
sits in the low group of every contrast and an offset there would bias all five main effects.
Each main effect is aliased with a four-factor interaction and each two-factor interaction with a
three-factor one, so all main effects and all ten two-factor interactions are estimable. The
model is accuracy ~ seed block + 5 main effects + 10 interactions, fitted by OLS on 16 cells × n
seeds (at n = 4: 64 runs, 19 parameters, 45 residual degrees of freedom). The one-high cells
exist anyway as singles and augment the fraction.

**FF2** is a 2^(4−1) design of resolution IV on d24h2 at 350k, H = 500. Its factors are four
width-and-head details the reference configuration does not have: A = per-value activation widths
(M011), B = a wider activation init (M012), C = a tanh table before attention and FFN (M016), and
D = a head of three hidden layers of width 32 (M045). The generator is D = ABC (I = ABCD): the
fraction holds the all-low cell (the in-wave d24h2 replica), the six two-high cells and the
all-high cell (M096 to M102). Main effects are clear of two-factor interactions; the interactions
are aliased in pairs (AB = CD, AC = BD, AD = BC). The model has seed blocks, 4 main effects and 3
aliased interaction pairs (at n = 4: 32 runs, 11 parameters, 21 residual degrees of freedom). If
d24h2 fails at 350k, FF2 moves to d32h4 at 5M. The open-loop β schedule (M014) stays out of FF2
because it changes feasibility rather than a width detail.

## Wave 1: measurements before any GPU time

Nothing in wave 1 trains a model. Work done in an interactive notebook informs the design only
and is never quoted.

| # | measurement | what it gates |
| --- | --- | --- |
| Z01 | static floors (0-bit and 1-bit-alive) of every Delta architecture, traced with HGQ2's own counter | replaces the shape arithmetic above; gates every 350k cell and test (b) of validity |
| Z02 | 0-bit channel maps: the fraction of channels with relu(i + f) = 0 per site, from logged widths | whether the budget is met by structured channel loss; the Q/K/V baseline for M007 |
| Z03 | accumulator EBOPs: b_acc = b_act + ⌈log₂ fan_in⌉ per binary layer, reported beside native EBOPs | a second cost column on every row |
| Z04 | distinct sign rows of `input_proj` on stored binary checkpoints (at most 8 at fan-in 3) | the M1 baseline for M024, M040, M050 |
| Z05 | re-selection of stored runs under other rules (lowest EBOPs; AUC against accuracy) | how far the rule moves the pick |
| Z06 | how HGQ2 0.1.9 bills a ternary zero | whether M047 is comparable with binary at equal target |
| Z08 | the diagnostic suite: attention entropy over log n_valid with gated keys masked, the attention-ablation delta (A·V replaced by the mean over V), sign snapshots (flip-flop and C2I ratios), the β trajectory, the latent-vs-binary gap | every entry's diagnostic column |
| Z09 | working-point metrics: rejection at signal efficiency 0.5 and per-class efficiency at fixed mistag | reported at every confirm |
| Z10 | data caches: gated 90/10 at N = 32, ungated 90/10 at N = 64, N = 64 with derived features, N = 64 with real-slot standardization | M009, M038, M040, M041 and their packages |
| Z11 | unit tests for delayed treatments: schedules at restart boundaries, the EDE period, the width freeze, the collapse stop | G0 |
| Z12 | the static-feasibility table applied to every entry × target | the 350k cells |
| Z13 | patch gate: every change is opt-in, and with it off the config digest and initial kernel hashes of the reference configs are unchanged | without it the reference snapshot is not a valid control |
| Z14 | screen resolving power: the epoch-500 validation-accuracy sd of the bases over seeds 1 to 8 | the seed count, before any wave-2 pod |
| Z15 | teacher leakage check: no teacher or warm-start checkpoint was trained on jets in the 90/10 validation split | the distillation entries |

## Prerequisite jobs

Two teachers are trained before the entries that need them: an FP teacher (d32h4, N = 64,
unquantized weights, the reference activation and softmax quantizers at their init widths, the
same split and recipe, no EBOPs pressure, 2,000 epochs, seed 101) for M027, M035, M036 and their
packages, and an int8 teacher, identical except for int8 weights, for M063. No stored checkpoint
qualifies: stored ones used an 80/20 split of the same 620,000 jets, so they have likely seen the
student's validation jets. Each new architecture (Linformer, ReLU/N, Deep Sets, N = 32, L = 2,
d_model 16, FFN 64, the 32/32/32 head, the tanh table, and distillation, which adds a teacher pass
per step) gets a 10-epoch timing canary in the first pod that introduces it.

## Budget

Run-epochs from `delta_spec.py`. At n = 4, wave 2 runs 284 entry runs (268 at H = 500, 8 at
1,000, 4 at 1,500, 4 at 2,000) and 12 replica runs, for 170,000 run-epochs; wave 3 runs 256 entry
runs (236 at H = 500, 20 at 2,000) and 12 replica runs, also 170,000; the teachers add 4,000.
The totals are 344,000 at n = 4, 514,000 at n = 6 and 684,000 at n = 8. A confirm wave of 12
cells is 672,000 run-epochs.

With s_e the seconds per epoch per process at K = 6 processes per pod and P pods, the screen's
wall clock, packing each horizon serially, is at most:

| n | P = 2 | P = 10 |
| ---: | ---: | ---: |
| 4 | 32,500 · s_e | 13,500 · s_e |
| 6 | 46,000 · s_e | 15,500 · s_e |
| 8 | 61,000 · s_e | 18,000 · s_e |

s_e has not been measured for these configurations; each architecture's canary sets it.

The cheap version covers 38 singles: those at code tier T0, T0a or T1 with a screen cell, leaving
out the baselines and the three teacher-dependent entries. It runs 3 seeds and one target each (5M for training-only
levers, 350k on d24h2 for the rest), with one 500-epoch replica per target: 69,000 run-epochs. It
answers which single levers move the base and which make 350k feasible, and loses the packages,
the factorials, the T2 architectures and the 5M cells of the attention levers.

Delta spends no FPGA synthesis. The power-of-two scale levers, any 0-DSP claim, the tanh-table
and key-mask export paths and the LUT cost of the accumulator need a separate synthesis study.

## Lessons from earlier runs that bind the design

**350k reachability binds before accuracy.** Only N = 8 has produced a feasible checkpoint at
350k in this project. Hence the static floors gate every 350k cell and the floor family is
screened first.

**A budget squeeze after an early unconstrained peak has not recovered.** In earlier runs,
squeezing a model to a budget far below the cost of its early unconstrained peak did not bring
back that peak's accuracy, with fixed and gradual schedules alike. So selection is only ever among feasible checkpoints, every run logs its first
feasible epoch, the screen reads the best feasible snapshot as of H, and a moving target entry
stays parked.

**Binary QAT is sensitive to the learning rate.** Earlier binary runs on the archived recipe
collapsed within the first epochs at a peak LR of 2e-4 or more, while the reference peaks at
3e-3. That is the largest single risk to the reference and to every entry paired with it. Hence
the stability gate in every screen, the base-stability pause, and early screens of an LR ladder
(M028, M029), a warm-up (M030), clipped STEs (M018, M019), the optimizer decomposition (M033,
M074, M103) and latent clipping (M021).

**Check that each number measures its treatment.** Earlier records include a width-freeze run
whose selected checkpoint preceded the freeze, memory models that failed metric reproduction on reload, AUC and accuracy selecting different
checkpoints, and a target set after the runs. Hence G0, horizons that cover each onset, one
selection rule, reload checks, and targets fixed here (350k, 1.4M, 5M).

**EBOPs is not silicon.** EBOPs does not count the β-restore affines, the growth of adder trees,
or memory tables, and the softmax has a fixed cost. So accumulator EBOPs sits beside native
EBOPs, no LUT, DSP or latency statement is drawn from any Delta number, and every entry carries
hardware-risk labels computed from its config change:

- "DSP audit required": the per-channel power-of-two gain (M024, M025 and packages M065, M066,
  M067, M080). No entry adds a new activation × activation product; those ideas are parked.
- "no HLS export path yet; export is refused": the uncentered binarizer (M017), the β modes
  (M022, M023), the channel gain, the head widths (M045), the tanh table (M016), the non-binary
  weight schemes (M047 to M050), and every package and factorial cell containing them.
- "no HLS export path yet": the Deep Sets body (M006), Linformer and ReLU/N attention (M004,
  M005, M053, M054, M056, M059, M060) and the runtime key mask (M039, M076 to M080). Bop is a
  training-only change and carries no label.
- "not a hardware candidate until the derived-feature path is costed": M040 and every package
  or FF1 cell containing it; the derived features are computed off-model and never billed.

**Single-seed gaps have gone flat at more seeds.** Hence at least 4 seeds per screen cell, 8 per
confirm cell, and a paired-gap sd from the replicas before any threshold is trusted.

## Parked and rejected

Parked ideas can return with a reason and a new ID. Rejected ones break the thesis or ask a
different question.

| idea | status | reason |
| --- | --- | --- |
| ensembling | rejected | N models cost N times the logic inside one fixed trigger envelope |
| per-channel arbitrary learned β | rejected as written | breaks the scalar fold; each channel becomes a multiplier and a DSP risk; kept only as power-of-two shifts (M024, M025) |
| BiT "two-set" weight levels | rejected | risks more than two effective weight values |
| input width matched to PUPPI candidates; input width sweep | parked | no source for the per-field PUPPI bit width yet; the input quantizer is learned anyway |
| attention pooling (PMA) or a class token | rejected unless a labelled baseline | a third activation × activation pair; a class token changes T to 65 in every shape |
| GLU FFN | rejected | a new activation × activation multiply per FFN |
| RMSNorm or a trainable norm | parked | the sum of squares is an activation × activation term plus an rsqrt table |
| SubLN | rejected here | it hurt binary under standardized inputs in earlier runs, and it confounds weight-width comparisons |
| Bi-Real shortcuts | parked | predicted no effect at L = 1; revisit if M044 (L = 2) survives |
| weight-tied L = 2 | rejected | no EBOPs saving (activations still pass twice) |
| tanh FFN activation | merged | into M016 (tanh as a pre-block bound); ReLU is free in hls4ml |
| low-rank binary factorization | parked | no source; predicted not to save EBOPs once the rank-r stream is paid |
| max or scaled-sum pooling | parked | scaled sum equals GAP at N = 64; max is predicted negative and its hls4ml support is unverified |
| MLP-Mixer | parked | not permutation invariant; the published N = 64 row is not stated as trained at 350k |
| JEDI-linear-style body | parked | after M006: Deep Sets answers the collapse question first |
| Engram-style memory | rejected here | multi-bit tables, no HLS lowering, memory cost outside native EBOPs; studied separately ([Engram record](../current-work/ENGRAM_STUDY.md)) |
| d_model 48, FFN 16, N = 16, Linformer k = 16, a 700k rung | parked | second rungs, run only if the first rung (M042, M043, M009, M004, M010) shows curvature |
| ApproxSign, ReSTE | parked | the same backward slot as M018 and M019; W1A1 motivation |
| stochastic sign | parked | its main claim concerns spread; low theory priority |
| ternary-weight splitting | parked | doubles the architecture mid-run, which breaks the controller and the pairing; M027 covers the two-stage idea |
| one-cycle schedule | parked | at the screen horizon it is M030 plus one reference cycle; M032 covers the no-restart extreme |
| batch 256 with scaled LR | parked | about 10 times the steps per epoch |
| SWA across cycle ends | parked | the same re-threshold issue as M034, horizon at least 1,500 |
| label smoothing | parked | no binary mechanism; a smoothed teacher weakens distillation (Müller et al., arXiv:1906.02629) |
| SAM | parked | twice the step cost; flatness in latent space is not shown to transfer to the signs |
| dropout, stochastic depth | parked | a capacity-limited L = 1 model; interacts with learned widths |
| rotation, smearing, constituent dropout | parked | approximate symmetries; after the exact reflection (M037) |
| pT sample reweighting | rejected here | the reference recipe uses no sample weights, and an earlier N = 8 study of this reweighting found it lowered accuracy |
| ΔR ordering | parked | predicted to matter at N ≤ 16, not at N = 64 |
| 80/20 split | parked | changes the validation set, which is the screen readout; a confirm-only question |
| class-set changes | rejected | a different task |
| working point | converted | a reporting rule (Z09) |
| tensor-wide widths | parked | channel granularity did better than tensor granularity in the constrained N = 8 ablation ([project README](../../README.md#post-conference-ebop-constrained-training)) |
| a lower init f0 | parked | after M012 |
| moving EBOPs target | parked | gradual budget schedules at N = 8, 16 and 64 gave no clear gain; reinstated only if M014 shows early-pressure harm |
| lowest-EBOPs selection | converted | re-selection from logs (Z05) |
| fixed 4-bit softmax output | parked | reinstated only if the reference reverts to the earlier quantizer set |
| sigmoid attention | parked | same mechanism as M005 and needs a table |
| A2Q training | rejected | no weight-magnitude lever after binarization; its metric is Z03 |
| clipvalue-only, β₂-only | aliased | read as the pair M074 inside the optimizer 2 × 2 |
| ParT pairwise features | parked | needs a cost pass first |

## Open questions

Each question names the evidence that would settle it and the entries that wait on it.

| # | question | evidence | waits |
| --- | --- | --- | --- |
| OQ-01 | Is 350k reachable at N = 64 by d32h4 with a nonzero attention branch? The floors say not with every attention channel alive | Z01 for the other architectures; per-channel widths of the reference at epoch 500 (Z02) | every 350k d32h4 cell |
| OQ-02 | Is the attention collapse Laatu et al. report at N = 64 pure budget geometry? | M049 against d32h4 at 5M with the attention diagnostics and the ablation delta | the reading of any collapse |
| OQ-03 | Is the fan-in-3 input projection the binding capacity loss? | Z04 distinct rows; M050 against the base at the same seeds | M024, M040, M067 |
| OQ-04 | Does the project's earlier recipe (LR 2e-5, batch 256) train signs at all? | C2I and flip-flop ratios from sign snapshots | M021, M033, M020 |
| OQ-05 | How does HGQ2 bill a ternary zero, and what is the LUT per EBOP of a ±1 layer at fan-in 32? | Z06; single-layer synthesis through place and route | M047; the iso-EBOPs framing |
| OQ-06 | Does a restart schedule beat one cosine of equal steps for binary weights? | flip-flop spikes against the best-checkpoint position; M032 at confirm | M068 to M073 |
| OQ-07 | Does the ordering at the trigger working point match top-1 accuracy and AUC? | Z09 at every confirm | every confirm report |
| OQ-08 | Literature: weights-only binary small transformers with multi-bit activations; Bop or flip-aware optimizers on transformers; SAM for binary networks; whether BinaryBERT's 2-bit to 1-bit cliff has a small-model analogue | table-level reads | M020, the parked SAM, the thesis framing |
| OQ-09 | The per-field bit width of L1 PUPPI candidates | the CMS Phase-2 L1 trigger TDR (CERN-LHCC-2020-004), Correlator Layer-2 firmware data formats | the parked input-width entries |
| OQ-10 | Does HGQ-LUT undercut a binary transformer's LUT count at comparable accuracy? | the tables of arXiv:2604.22293 checked against the PDF | the 0-DSP framing |
| OQ-11 | A2Q and A2Q+ at table level | full-text reads of arXiv:2308.13504 and arXiv:2401.10432 | the Z03 framing |
| OQ-12 | ReLU-attention, sigmoid-attention and Softermax numbers | full reads of arXiv:2309.08586, arXiv:2409.04431, arXiv:2103.09301 | M005, M059 |
| OQ-13 | Does hls4ml elide a 0-bit datalane or synthesize a 0-width wire? Can a runtime key mask pass the table softmax without breaking token folding? Is there an export path for a tanh table, a Linformer projection and a list-valued head? | a code read of the hls4ml conversion path and a toy conversion | the hardware status of M016, M039, M045, M004 |
| OQ-14 | Laatu et al. state the features as pT, η, φ; which coordinates (absolute or jet-relative) produced their Table 1? | the paper's authors or a code release with a license | the comparand's input set |
| OQ-15 | What does the jet-tagging distillation paper (arXiv:2311.14160) report, on which metric? | a full read | M035, M036 |
| OQ-16 | XNOR-Net++, Bi-Real Net, IR-Net, ReSTE, BinaryBERT, BiT: table numbers, not abstract claims | full reads | the B entries |
| OQ-17 | The cost of a ParT pairwise-feature bias in EBOPs and activation × activation products | a cost pass | whether to add it |
| OQ-18 | Resources of the 3-feature JEDI-linear row | the paper's tables | the parked JEDI-linear body |
| OQ-19 | How EMA latents are re-thresholded so that the exported weights equal the evaluated ones | a design note and a reload test | M034 |
| OQ-20 | The model and flow behind Sloot's ternary-over-binary result (FastML 2026) | the paper and its code | the M047 framing |
| OQ-21 | Is the MLP-Mixer row Laatu et al. quote from their reference 18 trained at 350k? | that reference | the parked MLP-Mixer |

## Choices still open

These are design choices made here with medium or low confidence. Each is revisited before the
wave it affects.

| choice | alternatives | confidence |
| --- | --- | --- |
| training-only levers screen at 5M on d32h4 and confirm at 5M and at 350k on d24h2 | screen them at 350k on d24h2 (h 0.399); screen at both targets (the 22 training-only singles and M046: 23 more cells, 56,000 more run-epochs at 4 seeds) | medium |
| seed count set from the measured spread, ranking mode if no n ≤ 8 qualifies | n = 8 with g_res; a smaller BH family (per code tier or family); a fixed threshold without FDR control | medium |
| the reference snapshot is the primary screen control; the replica takes over on the drift trigger | always pair against the in-wave replica (same code and GPU class; already budgeted) | medium |
| FF2 on d24h2 at 350k; FF1 on d32h4 at 5M | FF2 on d32h4 at 5M; FF1 on d24h2 at 350k | medium |
| teachers: d32h4, 90/10, reference recipe, 2,000 epochs, seed 101 | 7,000 epochs (10,000 more run-epochs); 500 epochs; no distillation in Delta | medium |
| values chosen here without a source: latent clip at [−1, 1] (BinaryConnect's range), EMA decay 0.999 (Mean Teacher range), PID gains p 2 and i 0.2, the collapse rule, a 2-bit table minimum, a cap of 12 confirm cells | set each in its wave from a small pilot | low |
| parked: SAM, label smoothing, batch-256 rescaling, JEDI-linear, MLP-Mixer, stochastic sign, ReSTE, ApproxSign | run them as singles (8 × 4 × 500 = 16,000 more run-epochs and two T2 patches) | medium |
| M027 and M036 at the 350k confirm need a d24h2-shaped teacher, while the planned teacher is d32h4 | train a second teacher; confirm those two at 5M only | open |

Also uncertain: every floor other than the two reference architectures until the trace; whether
d24h2 reaches 350k non-degenerately at all; and whether the reference LR of 3e-3 is stable. If it
is not, most of wave 2 re-bases on the earlier optimizer, the largest single risk to this queue.
