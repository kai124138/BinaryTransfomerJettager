# Delta: a pre-registered queue of methods for the binary-weight N = 64 tagger

**Status: designed. Nothing in Delta has been trained, and no number in these files is a result.**

Delta is a catalogue of 103 training, quantization, architecture and input changes, each written
down with its base configuration, its predicted direction, the diagnostic that would confirm the
mechanism, and the rule that decides whether it advances. The question it is built to answer:
which of these changes, alone or in pre-registered combinations, move the validation accuracy or
the 350,000-EBOP feasibility of a binary-weight transformer at N = 64 constituents, and which of
those survive an 8-seed confirmation over the full training schedule?

[Theory](theory.md) · [Rules](rules.md) · [Method table](methods.md) · [Catalogue](catalogue/delta.json) · [Project results](../../README.md)

## Why it exists

The September EBOP-constrained runs found feasible checkpoints under 350,000 EBOPs only at
N = 8. No N = 16 or N = 32 variant reached the target in 1,000 epochs
([final results](../current-work/TRAINING_RESULTS_20260923.md)), and the short N = 64 screen
found no feasible checkpoint either ([screen](../current-work/CONSTITUENT_SCREEN_20260923.md)).
Those runs changed one thing at a time, mostly on one seed. That is a slow way to search a
space where the cost model itself decides which arms can be feasible at all.

Arithmetic on layer shapes shows how tight the budget is. With binary weights, a 350k budget
and every activation channel at 0 bits, the softmax of a 4-head, 64-token attention block still
costs 343,040 EBOPs under the reference quantizer set (computed from layer shapes by
[`delta_spec.py`](catalogue/delta_spec.py); formula in [rules.md](rules.md#static-floors)). That
leaves 6,960 EBOPs for everything else, less than the 7,328 that the input projection and the
classifier head need with every channel at 1 bit. A 2-head, d_model 24 block has a 0-bit floor of
171,520 and leaves real room. So Delta separates two questions that the earlier runs mixed
together: which changes make 350k reachable, and which changes improve accuracy once it is.

Delta also exists to stop re-running ideas under new names. Every entry records whether it, or
something close to it, was tried before in this project, and why a retry is expected to differ.

## The reference configuration

Every entry is a change relative to one of three base configurations. They are trained in a
separate study; no result from that study is used here.

| | value |
| --- | --- |
| architectures | **d32h4**: d_model 32, 4 heads, 1 block, FFN 32, learned positional encoding. **d24h2**: d_model 24, 2 heads, 1 block, FFN 32, no positional encoding. Both: ReLU FFN, global average pooling, no normalization layers |
| bases | d24h2 at 350k (the 350k accuracy base), d32h4 at 350k (a feasibility reference close to its static floor), d32h4 at 5M (the 5M base) |
| weights | `binary_absmean`: centered absmean, one scale per tensor, weights in {−β, +β} |
| activations | widths learned per channel; activation and Q/K/V stream quantizers wrap on overflow and can reach 0 bits; learned softmax output; trainable exp and inverse tables with a 4-bit minimum; the exponential's input quantizer saturates |
| training | Adam with framework defaults; cosine restarts with peak LR 3e-3 every 500 epochs; batch 2,790; 7,000 epochs; a PID controller on β drives EBOPs to the target |
| data | [HLS4ML LHC Jet dataset](https://zenodo.org/records/3602260), N = 64, features pT, η_rel, φ_rel; constituents with pT < 2 GeV zeroed; 90/10 split of the 620,000-jet training archive (62,000 validation jets); 260,000 held-out jets |
| selection | validation accuracy among feasible, non-degenerate checkpoints |

The reference shares some settings with Laatu et al.
([arXiv:2510.24784](https://arxiv.org/abs/2510.24784), sections 2 and 3): three features per
particle, a 350,000-EBOP target, a PID controller on β, and the 620,000/260,000 split. It differs
in others. Laatu et al. use one attention head, HGQ weights with learned per-parameter widths,
value-wise activation widths (tested here as M011, not part of the base) and attention widths
floored at 1 bit; here the heads number 2 or 4, the weights are binary, and attention streams can
reach 0 bits. Like d24h2, their model has no positional encoding. The paper does not list the
remaining training settings. The two architectures, the binary weights and the quantizer set are
this project's.

## What is in the catalogue

The 103 entries fall into 50 singles and 53 combinations. The singles are 46 methods and 4
labelled non-binary baselines (ternary, int8 weights, HGQ learnable weight widths, an 8-bit input
projection), which are comparands and never the thesis. The combinations are 31 packages, each
read against its components on the same seeds, and 22 cells of two fractional factorials that
test whether effects add. By family:

| family | entries | what it varies | notes |
| --- | ---: | --- | --- |
| F, floor and attention cost | 20 | heads, table widths, Linformer, softmax-free attention, Deep Sets, N | decides which architectures have a live 350k base |
| Q, activation widths and the controller | 6 | width granularity and init, PID gains, open-loop β, width freeze, a tanh table | [activation-widths.md](activation-widths.md) |
| B, binarization | 16 | binarizer convention, STE variants, Bop, latent clipping, scale modes, warm start | [binarization.md](binarization.md) |
| R, recipe | 18 | peak LR, warm-up, restarts, weight decay, EMA, distillation | [recipe.md](recipe.md) |
| P, inputs and data | 12 | augmentation, pT gate, key masking, derived features, standardization | [inputs-data.md](inputs-data.md) |
| A, architecture | 5 | d_model, FFN width, depth, head, positional encoding | [architecture.md](architecture.md) |
| BL, labelled baselines | 4 | ternary, int8, HGQ weights, int8 first layer | never the thesis |
| FF1, FF2 | 15 + 7 | fractional-factorial cells | additivity tests |

Packages are counted under the family of their theme. Each family note holds the method cards
the entries were drawn from: the mechanism, the primary source with its arXiv id, what the
source reports and whether it transfers, the code change, and the hardware hazards. The
[theory note](theory.md) groups the cards by seven named mechanisms of accuracy loss (M1 to M7),
so each entry tests a stated cause rather than adding one more trick.

## How it runs: waves and two tiers

Delta runs in waves, and each wave gets its own pre-registration before launch. A wave may
tighten a rule in [rules.md](rules.md) but may not loosen one.

| wave | content |
| --- | --- |
| 0 | the reference configurations, trained separately, with snapshots every 500 epochs |
| 1 | no GPU: static floors from HGQ2's own counter, 0-bit channel maps, the diagnostic suite, data caches, unit tests for delayed treatments, and the seed-count readout |
| 2 | singles; the floor family goes first, because it decides which architectures have a 350k base |
| 3 | packages and factorial cells, on the same seeds as their singles |
| 4 | confirmation of the survivors |

Every entry passes through two tiers. The **screen** trains for a short horizon H (500 epochs,
one cosine cycle, unless the treatment starts later), on n seeds per cell with n set before the
wave from the measured seed spread (4, 6 or 8), and reads validation accuracy only. Screen numbers
never leave the screen. The **confirm** tier runs the full 7,000-epoch schedule on 8 seeds per
cell, selects on validation, and touches the 260,000 held-out jets once, after the last epoch,
with Holm correction across the confirm wave. The advance rule between the two, the
Benjamini-Hochberg families, the drift replicas and the stability gates are in
[rules.md](rules.md#screen-advance-rule).

The seed arithmetic is sobering. For 8 seeds to resolve a 0.3-point paired gap across the
43 singles screened at 5M, the epoch-500 accuracy spread of the base has to be at most
0.116 points (`catalogue/screen_power.py`). If it is larger, the screen runs as a declared
ranking at 4 seeds with no advance claims, and only the confirm tier tests anything. That
outcome is pre-registered too.

## Budget

In run-epochs, from `delta_spec.py`: the full screen costs 344,000 at 4 seeds per cell and
684,000 at 8, including two 2,000-epoch teacher runs for the distillation entries. A confirm wave
of 12 cells costs 672,000. A cheaper version (38 config-level and small-patch singles, 3 seeds,
one target each, always a ranking) costs 69,000. Seconds per epoch for these configurations
have not been measured, so no wall-clock or pod-hour figure is given here.

## What has run and what has not

Nothing has run. No Delta configuration has been trained, the wave-1 measurements have not been
made, and the static floors in [methods.md](methods.md) are shape arithmetic that HGQ2's own
counter has not yet checked. Several entries carry values marked `<set before launch>` (the Deep
Sets layer widths, the open-loop β schedule, the activation init f0); those are fixed in the
wave's own pre-registration. The rules assume the reference configurations reach 350k with
feasible, non-degenerate checkpoints; if d24h2 does not, the contingencies in
[rules.md](rules.md#the-350k-cells) apply.

## Code

[`code/delta/`](../../code/delta/README.md) holds 25 patches against the screen code in
`code/constituent-study-20260922` and six new modules, named after the catalogue's
`code_changes`. They cover 22 of the 38 code changes plus three post-run diagnostics; the
Linformer layer is there but not wired. The code for 34 entries is complete there and 11 need
none. The other 58 require the reference-configuration patches, which are not published here.
Every entry's base configuration also depends on that code, so no Delta cell can be run from
this repository alone. The code README lists which patches `convert_binary.py` refuses to
export and which checks were run.

## Files

| path | what it is |
| --- | --- |
| [theory.md](theory.md) | mechanisms of accuracy loss, cost-side reasoning, predicted signs, open questions |
| [rules.md](rules.md) | static floors, 350k cell roles, targets, tiers, seed count, advance rule, controls, lessons, parked ideas |
| [methods.md](methods.md) | the full method table, generated from the catalogue |
| [binarization.md](binarization.md), [recipe.md](recipe.md), [activation-widths.md](activation-widths.md), [architecture.md](architecture.md), [inputs-data.md](inputs-data.md) | method cards by family, with sources |
| [catalogue/delta.json](catalogue/delta.json) | one object per entry: config change, targets, horizon, pairing, prediction, diagnostic, floors, hardware labels |
| [catalogue/delta_spec.py](catalogue/delta_spec.py) | writes `delta.json` and prints the counts and budget quoted here |
| [catalogue/delta_tables.py](catalogue/delta_tables.py) | renders `methods.md` from `delta.json` |
| [catalogue/screen_power.py](catalogue/screen_power.py) | family sizes and the power constants of the advance rule (needs numpy and scipy) |

To regenerate the catalogue and the table:

```bash
cd docs/delta/catalogue
python3 delta_spec.py .
python3 delta_tables.py
python3 screen_power.py
```
