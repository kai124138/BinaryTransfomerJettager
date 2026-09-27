#!/usr/bin/env python3
"""Generate delta.json, the machine-readable Delta method catalogue.

Usage: python3 delta_spec.py [outdir]    (standard library only)

The script is the single source for the entry list, the static EBOPs floors, the 350k cell roles
and the screen budget. The Markdown tables in docs/delta/methods.md are rendered from its output
by delta_tables.py; screen_power.py reads the same file.

Static floors are computed from layer shapes only. No trace, training run or measurement enters
them. With binary weights (b_w = 1) and every activation channel at b_a bits:
  dense layer        N * fan_in * fan_out * b_a * b_w
  Q.K and A.V        T * S * d * b * b            (T = S = N; S = k for Linformer)
  softmax            16*H*T*S + 4*(H*T*S - H*T)   (exp x inv product and row sum, tables at 4 bits)
                     + H*T*S                      (the exponential's input quantizer saturates,
                                                   so every score entry costs at least 1 bit)
The 0-bit floor puts every activation channel at 0 bits, so only the softmax terms remain. The
1-bit-alive floor puts every channel at 1 bit. A small lookup-table term of the softmax (at most
a few tens of EBOPs) is not modelled.
"""
import itertools
import json
import math
import sys
from collections import Counter

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
K = 6                    # processes per pod, used only for the wall-clock estimate in units of s_e
T350, T5M, T1M4 = 350000, 5000000, 1400000
LABEL_FLOOR = "computed from layer shapes (delta_spec.py); not a measurement"


# ---------------------------------------------------------------- floors (computed from layer shapes)
def softmax_floor(H, T, S, tb=4, exp_in=1):
    hts = H * T * S
    return tb * tb * hts + tb * (hts - H * T) + exp_in * hts


def dense_1bit(N=64, nf=3, d=32, ffn=32, L=1, head=None, C=5, wbits=1, wbits_in=None):
    """Dense layers and head at 1-bit activations; weight bits multiply (EBOPs = b_a * b_w)."""
    wbits_in = wbits if wbits_in is None else wbits_in
    per_block = 4 * N * d * d + 2 * N * d * ffn
    head_cost = (d * d + d * C) if head is None else sum(a * b for a, b in zip([d] + head, head + [C]))
    return wbits_in * N * nf * d + wbits * (L * per_block + head_cost)


def arch_floors(N=64, nf=3, d=32, H=4, ffn=32, L=1, attn="softmax", k=None, tb=4, qk_min=0, head=None,
                exp_in=1, wbits=1, wbits_in=None, wbits_unknown=False):
    """Return (floor0, floor1_alive). attn is softmax, linformer, relu or none."""
    if attn == "none":
        return None, None
    S = k if attn == "linformer" else N
    sm = 0 if attn == "relu" else softmax_floor(H, N, S, tb, exp_in)
    qk1 = N * S * d
    proj = 2 * d * N * k if attn == "linformer" else 0
    f0 = L * (sm + qk_min * qk_min * qk1)
    if wbits_unknown:
        return f0, None
    f1 = dense_1bit(N, nf, d, ffn, L, head, wbits=wbits, wbits_in=wbits_in) + L * (sm + qk1 + qk1 + wbits * proj)
    return f0, f1


# hand substitutions quoted in docs/delta/rules.md
assert softmax_floor(4, 64, 64) == 262144 + 64512 + 16384 == 343040
assert dense_1bit() == 400544 and dense_1bit(d=24) == 251064
assert arch_floors() == (343040, 343040 + 400544 + 262144)
assert arch_floors(d=24, H=2) == (171520, 171520 + 251064 + 196608)

D32 = dict(N=64, nf=3, d=32, H=4, ffn=32, L=1, attn="softmax", k=None, tb=4, qk_min=0, head=None,
           exp_in=1, wbits=1, wbits_in=None, wbits_unknown=False)
D24 = dict(D32, d=24, H=2)
NEAR_FLOOR_H = 0.10


def headroom(f0, f1, target):
    if f0 is None or f1 is None:
        return None
    if f0 >= target:
        return "STATIC_INFEASIBLE"
    return min((target - f0) / (f1 - f0), 1.0)


# ---------------------------------------------------------------- planned code changes
# slug: (tier, size estimate in lines, where it acts, what it does). Names may change when the
# patch series is published.
PATCHES = {
 "lr-schedule-variants": ("T1", "~15", "learning-rate schedule", "peak decay per restart (m_mul), cycle growth (t_mul) and a linear warm-up on top of the reference cosine-restart schedule; unit test at restart boundaries"),
 "beta-schedule-runner": ("T1", "~12", "training loop", "honour train.ebops.controller 'schedule' (an open-loop beta schedule) in the constrained runner"),
 "softmax-table-min-bits": ("T1", "~5", "softmax quantizers", "expose the minimum width of the trainable exp and inverse tables"),
 "qk-stream-min-bits": ("T1", "~10", "Q/K stream quantizers", "lower bound on the Q/K stream widths (a width floor against collapse)"),
 "ebops-group-weight": ("T1", "~30", "EBOPs loss term", "per-layer-group multiplier on the beta*EBOPs term (attention group against the rest)"),
 "attn-linformer": ("T2", "~40-60 + export", "attention block, binary export", "binary sequence projections E, F (N -> k) on K and V"),
 "attn-relu-over-n": ("T2", "~80-150 + export", "softmax branch", "ReLU(scores)/N attention (a shift at N = 64); no exp or inverse tables"),
 "body-deepsets": ("T2", "~80-120 + export", "new model builder", "binary Deep Sets body (phi MLP, pooled context, rho MLP), written from the published topology"),
 "act-granularity-element": ("T1", "~10-20", "activation quantizers", "act_granularity 'element': one width per position and channel"),
 "act-init-f0": ("T1", "~20", "activation quantizers, matched initialization", "fixed activation init f0 with the integer bits tracking the range"),
 "pre-block-tanh-lut": ("T1", "~12 train + export", "block input, binary export", "table-based tanh before attention and before the FFN"),
 "head-dims": ("T1", "~25 + export", "classifier head, binary export", "arch.head_dims: a list of head hidden widths"),
 "binarizer-center-flag": ("T1", "~10", "binarizer", "quant.binary_center false: uncentered absmean"),
 "ste-variants": ("T1", "~25", "binarizer backward, epoch callback", "quant.ste in {bounded (default), clip_identity, ede}; EDE temperature restarted every LR cycle or annealed once"),
 "bop-optimizer": ("T2", "~60-100", "optimizer, variable routing", "Bop on the binary latents (flip on an EMA-gradient threshold); Adam on everything else"),
 "latent-clip": ("T1", "~10", "kernel constraint", "clip latent kernels to [-c, c] after each step"),
 "beta-mode": ("T1", "~30 incl. export", "binarizer, binary export", "quant.beta_mode in {absmean (default), absmean_pow2, learned_pow2}"),
 "channel-gain-pow2": ("T1", "~60 incl. export", "binary layers, binary export", "per-output-channel power-of-two gain (a shift) after the +-1 matmul; DSP audit required"),
 "pre-quant-shift": ("T1", "~20-30", "activation quantizer inputs", "learnable per-channel additive shift before each activation quantizer (the RSign shift, no slope)"),
 "init-from-checkpoint": ("T1", "~20", "matched initialization", "copy latents from a named checkpoint (the FP teacher) instead of the seed init"),
 "kd-constrained-runner": ("T1", "~30", "constrained runner, loss", "allow experiment.distillation in the constrained runner and supply teacher logits"),
 "kd-attention-map": ("T1", "~30-40", "loss, teacher tap", "attention-map distillation term (teacher with the same heads)"),
 "latent-ema-eval": ("T1", "~20", "epoch loop", "EMA of latents, re-binarized; candidate checkpoints evaluated on the EMA weights"),
 "augment-eta-phi-reflect": ("T1", "~15-25", "training batches only", "random eta_rel and phi_rel sign flips per jet; never on validation or held-out data"),
 "pt-gate-threshold": ("T1", "~5", "data loaders, cache builder", "gate threshold as a key (null = ungated); one cache per value"),
 "gated-key-mask": ("T1", "~20-40", "attention scores, cache", "additive mask on gated and padded key positions before the softmax"),
 "std-real-slots": ("T1", "~10", "cache builder", "standardization statistics over real (non-gated) slots only"),
 "derived-input-features": ("T1", "~25", "cache builder, config validation", "append log pT and Delta R computed from the three input features"),
 "weight-scheme-baselines": ("T1", "~25", "config validation, model builder", "admit quant.weight none / int8_absmax and quant.layer_weight_override for labelled baselines and teachers"),
 "ternary-absmean": ("T1", "~30", "binarizer (labelled ternary gate)", "ternary {-b, 0, +b} absmean baseline"),
 "hgq-learnable-weights": ("T2", "~40-60", "model builder", "HGQ learnable weight widths"),
 "accumulator-ebops-metric": ("T1", "~20", "EBOPs accounting", "closed-form accumulator bits b_act + ceil(log2 fan_in) per binary layer, reported beside native EBOPs"),
 "diag-sign-flips": ("T1", "~30", "epoch observer", "sign snapshot per layer: flip-flop ratio per epoch, C2I ratio"),
 "diag-attention": ("T1", "~40", "evaluation script", "attention entropy / log(n_valid) with gated keys masked in the diagnostic pass; attention-ablation delta (A.V replaced by mean V)"),
 "diag-beta-trajectory": ("T1", "~10", "epoch observer", "per-layer beta and latent |w|/beta histogram"),
 "diag-input-proj-rows": ("T1", "~15", "checkpoint reader", "distinct sign rows of input_proj"),
 "diag-latent-binary-gap": ("T1", "~20", "evaluation script", "validation accuracy with latent kernels in place of q*beta"),
 "screen-collapse-stop": ("T1", "~15", "epoch loop", "collapse rule: stop and record when validation accuracy <= 0.25 for 10 consecutive epochs after epoch 20"),
}
DIAG_ALWAYS = ["diag-attention", "screen-collapse-stop", "accumulator-ebops-metric"]

# ---------------------------------------------------------------- singles
S = []


def single(id_, name, fam, cards, mech, tier, delta, patches, tried, targets, pairing, pred, diag,
           horizon=500, arch=None, ref_keys=None, if_floor="5M-only", kind="single",
           targets_confirm=None, wave="W2", note=""):
    S.append(dict(id=id_, name=name, family=fam, tier=kind, cards=cards, mechanisms=mech, code_tier=tier,
                  combo_of=[], config_delta=delta, code_changes=patches, ref_keys=ref_keys or [],
                  tried=tried, targets=targets, targets_confirm=targets_confirm or targets,
                  pairing=pairing, prediction=pred, diagnostic=diag, horizon_screen=horizon,
                  arch=arch or {}, if_floor_fails=if_floor, wave=wave, note=note))


U = "untried"
TS = [T350, T5M]
TO, TOC = [T5M], [T5M, T350]
TBD = "<set before launch>"
PRIOR = "docs/current-work/TRAINING_RESULTS_20260923.md"

# F: floor and attention cost
single("M001", "Two heads at d32", "F", ["A01"], ["M5", "M7"], "T0", {"arch.n_heads": 2}, [],
       f"tried at N = 16 under the earlier quantizer set, never feasible at 350k ({PRIOR}); retried because the softmax floor halves with the head count",
       TS, "paired-if-hash (Wq/Wk/Wv reshape per head; kernel-hash check, else Welch)",
       "feasible at 350k in more seeds than d32h4; accuracy at 5M flat to slightly down",
       "EBOPs split (softmax / QK / AV / dense), Q/K and V 0-bit fractions, entropy/log n_valid",
       arch=dict(H=2), if_floor="drop")
single("M002", "One head at d32", "F", ["A01"], ["M5", "M7"], "T0", {"arch.n_heads": 1}, [],
       f"tried at N = 16 under the earlier quantizer set (3 seeds), never feasible at 350k ({PRIOR}); retried for the same reason as M001",
       TS, "paired-if-hash", "350k feasibility up; accuracy at 5M below M001", "as M001", arch=dict(H=1), if_floor="drop")
single("M003", "Softmax table minimum 2 bits", "F", ["Q08"], ["M7"], "T1", {"quant.softmax_table_min_bits": 2}, ["softmax-table-min-bits"],
       "untried: the softmax tables were fixed before the reference quantizer set made them trainable",
       TS, "paired (same shapes)", "lowers the d32h4 0-bit floor; accuracy at 5M flat", "table widths at the selected checkpoint, EBOPs split",
       arch=dict(tb=2), if_floor="drop", note="departs from the reference 4-bit table minimum; labelled")
single("M004", "Linformer projection k = 8", "F", ["A10"], ["M5", "M7"], "T2", {"arch.attn_kind": "linformer", "arch.linformer_k": 8}, ["attn-linformer"],
       "untried with binary weights", TS, "paired-if-hash (new E/F projections; other kernels shared)",
       "feasible at 350k with real headroom; accuracy at 5M within the interval of d32h4 at 5M",
       "EBOPs split, entropy over the k projected keys, attention-ablation delta", arch=dict(attn="linformer", k=8), if_floor="drop")
single("M005", "Softmax-free ReLU/N attention", "F", ["Q10"], ["M5", "M7"], "T2", {"arch.attn_kind": "relu_over_n"}, ["attn-relu-over-n"],
       "tried once in an early exploratory round (earlier dataset version, no EBOPs target), inconclusive; retried because it removes the whole softmax floor",
       TS, "paired-if-hash", "feasible at 350k; accuracy at 5M below d32h4 (competitive normalization lost)", "attention-ablation delta, EBOPs split",
       arch=dict(attn="relu"), if_floor="drop")
single("M006", "Binary Deep Sets body", "F", ["A09"], ["M5"], "T2", {"arch.body": "deepsets", "arch.deepsets_dims": TBD}, ["body-deepsets"],
       "untried with binary weights", TS, "unpaired (Welch): no shared attention tensors",
       "accuracy >= the reference at 350k (the Laatu et al. collapse argument); at 5M below d32h4",
       "no attention diagnostic applies; per-class AUC and the width map", arch=dict(attn="none"), if_floor="drop",
       note="layer widths set before launch from the published Deep Sets baselines; binary weights, no batch norm; the static floor needs the body's own shapes")
single("M007", "Q/K stream width floor 1 bit", "F", ["THEORY M5"], ["M5"], "T1", {"quant.qk_min_bits": 1}, ["qk-stream-min-bits"],
       U, [T5M], "paired", "forbids the Q/K 0-bit collapse; accuracy at 5M up if attention matters",
       "Q/K 0-bit fraction must be 0; entropy/log n_valid; attention-ablation delta",
       arch=dict(qk_min=1), if_floor="5M-only", note="0-bit floor on d32h4 is above 350k: 350k only in combination with M001 or M004")
single("M008", "Attention-group EBOPs weight x0.1", "F", ["THEORY M5, M7"], ["M5", "M7"], "T1", {"quant.ebops_group_weight": {"attention": 0.1, "rest": 1.0}}, ["ebops-group-weight"],
       U, TS, "paired", "350k: attention survives longer, dense layers pruned harder; sign depends on the budget",
       "EBOPs split by group against epoch, Q/K/V 0-bit fractions")
single("M009", "N = 32 constituents (crosses N by design)", "F", ["P01"], ["M7"], "T0", {"arch.n_part": 32}, [],
       f"tried at N = 32 under the earlier quantizer set, never feasible at 350k ({PRIOR}); retried because the N = 32 softmax floor is far below 350k",
       TS, "unpaired (Welch): different inputs; crosses N by design, labelled",
       "feasible at 350k with nearly every channel alive; accuracy against the reference undetermined (context against bits)", "EBOPs split, 0-bit fractions",
       arch=dict(N=32), if_floor="drop", note="needs a gated 90/10 N = 32 cache (CPU job)")
single("M010", "Target 1.4M (ladder rung)", "F", ["Q07"], ["M7"], "T0", {"train.ebops.pid.target_ebops": 1400000}, [],
       "untried rung between d32h4 at 350k and d32h4 at 5M", [T1M4],
       "paired by seed to d32h4 at 350k at a different target: a ladder point with no same-target control, so G3 does not apply",
       "accuracy between d32h4 at 350k and at 5M; locates the knee (descriptive; never advances on accuracy)", "feasible count, EBOPs split, 0-bit fractions",
       if_floor="n/a", note="ladder point only: G2 counts and the ladder plot are reported, G3 is not applied")
# Q: activation widths and the controller
single("M011", "Per-value activation widths", "Q", ["Q02"], ["M7"], "T1", {"quant.act_granularity": "element"}, ["act-granularity-element"],
       U, TS, "paired", "up at 350k (bits placed per particle rank), flat at 5M", "0-bit fraction per position; EBOPs split",
       note="value-wise heterogeneous quantization as in Laatu et al. (arXiv:2510.24784, section 2.2); breaks permutation invariance")
single("M012", "Wider activation init (fixed f0)", "Q", ["Q03"], ["M7"], "T0a", {"quant.act_f0": TBD}, ["act-init-f0"],
       U, TS, "paired", "slower EBOPs descent from a wider start; selected accuracy within the interval", "EBOPs against epoch, first feasible epoch",
       ref_keys=["quantizer-set"], note="f0 is set before launch; the integer bits track the data range and only f feels the EBOPs gradient; T0a if the reference quantizer code exposes f0, else the patch")
single("M013", "PID gains p 2, i 0.2", "Q", ["Q04"], ["M7"], "T0", {"train.ebops.pid.p": 2.0, "train.ebops.pid.i": 0.2}, [],
       "untried (a gentler PID was planned once and never run)", [T350], "paired",
       "more seeds feasible by epoch 500; accuracy flat", "beta trajectory, first feasible epoch, EBOPs overshoot below target",
       if_floor="350k only: on d24h2 by default; a feasibility probe on d32h4 at 350k if d24h2 fails at 350k",
       note="350k only: its purpose is reaching the budget; at 5M the reference PID already reaches the target")
single("M014", "Open-loop beta schedule", "Q", ["Q04"], ["M7"], "T1", {"train.ebops.controller": "schedule", "train.ebops.beta_schedule": TBD}, ["beta-schedule-runner"],
       "tried once in an archived round with a different input set; not comparable", [], "paired",
       "fewer seeds feasible than the PID at matched epochs (an open loop does not aim at a target)", "EBOPs against epoch, feasible count",
       targets_confirm=[T350], wave="confirm-only",
       note="the schedule spans the full 7,000 epochs, so no screen horizon measures it; no screen cell; offered at the confirm decision as a confirm-only cell (7,000 epochs, 8 seeds, against the 350k base at the terminal epoch)")
single("M015", "Fixed-width recovery after epoch 500", "Q", ["Q14"], ["M7"], "T0", {"experiment.recovery_after_epochs": 500}, [],
       "tried at N = 8 on the earlier recipe: the selected checkpoint (epoch 205, results/post_conference/ablation_metrics.json) preceded the freeze at epoch 800, so the treatment was never measured; retried with a horizon that covers it",
       TS, "paired (against the reference epoch-1,000 snapshot)", "accuracy up after the freeze at unchanged EBOPs",
       "onset = the logged freeze epoch (first feasible epoch >= 500); the selected epoch must be later, else 'treatment not measured'", horizon=1000)
single("M016", "tanh table before attention and FFN", "Q", ["Q09", "A14"], ["M6"], "T1", {"arch.pre_block_act": "tanh_lut"}, ["pre-block-tanh-lut"],
       U, TS, "paired-if-hash", "small accuracy gain (range control on a norm-free graph)", "saturation fraction at the following quantizer; table EBOPs term",
       arch=dict(extra="tanh table"), note="no export path for a table layer yet")
# B: binarization (training-only levers screen at 5M and confirm at 5M and 350k)
single("M017", "Uncentered absmean (BitNet/XNOR convention)", "B", ["B02"], ["M3"], "T1", {"quant.binary_center": False}, ["binarizer-center-flag"],
       "untried as a training arm (the two formulas were compared on one stored checkpoint only)", TO, "paired",
       "flat; fewer collective flips (the threshold no longer moves with the tensor mean)", "FF ratio per epoch", targets_confirm=TOC)
single("M018", "Clipped-identity STE (hard-tanh)", "B", ["B04"], ["M2"], "T1", {"quant.ste": "clip_identity"}, ["ste-variants"],
       "untried; the current bounded STE replaced an earlier backward that produced NaNs", TO, "paired",
       "flat to up at peak LR 3e-3; fewer divergences", "STE/latent gradient cosine, latent-vs-binary gap", targets_confirm=TOC)
single("M019", "EDE annealed STE, restarted each LR cycle", "B", ["B07"], ["M2"], "T1", {"quant.ste": "ede", "quant.ste_ede_period": 500}, ["ste-variants"],
       U, TO, "paired", "up (sharpening synced to the cosine cycle)", "latent-vs-binary gap per cycle", targets_confirm=TOC)
BOP_NOTE = ("gamma 1e-4 (undecayed) and tau 1e-8 are a starting point, not tuned for this setup. Sources: Helwegen et al., arXiv:1906.02107 "
            "section 5.2 (CIFAR-10: gamma 1e-4 decayed x0.1 every 100 epochs, tau 1e-8) and section 5.3 (ImageNet: gamma 1e-4 to 1e-6 linear, "
            "tau 1e-8); Larq larq.optimizers.Bop defaults threshold 1e-8, gamma 1e-4, no decay (larq/larq master, commit "
            "3d7de8832a477285bbf3c36252e24fcb9299a959, optimizers.py l. 314-316; a master commit, not a tagged release). The paper tuned "
            "W1A1 convolutional networks at batch 50 and 1,024, and tau is an absolute threshold on the EMA gradient, whose scale differs here "
            "(docs/delta/binarization.md, Bop hyperparameters). The |m| distribution of the reference binary latents is measured before launch "
            "and may set a tau scan. The flip rule is fixed before launch: (1) reflect the latent about its mean, w <- 2*alpha - w, which keeps "
            "|w - alpha| and so the layer scale beta (not the paper's rule); or (2) Bop as published (arXiv:1906.02107, Algorithm 2), where the "
            "weight is the sign itself and a flip sets w <- -w with latents at +-1, so beta becomes 1 - alpha^2")
single("M020", "Bop optimizer on binary latents", "B", ["B10"], ["M3"], "T2", {"train.binary_optimizer": "bop", "train.bop_gamma": 1e-4, "train.bop_tau": 1e-8}, ["bop-optimizer"],
       U, TO, "paired (same init; different update rule)", "sign unknown (designed for W1A1)", "FF ratio, C2I ratio", targets_confirm=TOC, note=BOP_NOTE)
single("M021", "Latent clip to [-1, 1]", "B", ["THEORY M3", "B09"], ["M3"], "T1", {"quant.latent_clip": 1.0}, ["latent-clip"],
       U, TO, "paired", "up late in training (inertia capped); partly redundant with restarts", "latent |w|/beta histogram, FF ratio late in each cycle", targets_confirm=TOC,
       note="the [-1, 1] range follows BinaryConnect (arXiv:1511.00363)")
single("M022", "Beta decoupled from latent magnitude (learned, power of two)", "B", ["THEORY M3, M6"], ["M3", "M6"], "T1", {"quant.beta_mode": "learned_pow2"}, ["beta-mode"],
       U, TO, "paired", "up; beta stops tracking latent growth", "beta against integer-bit trajectory correlation", targets_confirm=TOC)
single("M023", "Absmean beta rounded to a power of two (Libra-PB scale)", "B", ["B06"], ["M4"], "T1", {"quant.beta_mode": "absmean_pow2"}, ["beta-mode"],
       U, TO, "paired", "non-inferior accuracy; the beta-restore affines become shifts (LUT/DSP, not EBOPs)", "non-inferiority only; the cost claim needs synthesis", targets_confirm=TOC,
       note="advances on non-inferiority to a synthesis check, not to an accuracy confirm")
single("M024", "Power-of-two per-channel gain on input_proj", "B", ["THEORY M1", "B03"], ["M1", "M4"], "T1", {"quant.channel_gain": {"layers": ["input_proj"], "mode": "pow2"}}, ["channel-gain-pow2"],
       U, TO, "paired", "up (the 8 expressible input directions get distinct gains)", "distinct sign rows of input_proj, per-channel gain histogram", targets_confirm=TOC,
       note="two symmetric values per channel; DSP audit before any hardware claim")
single("M025", "Power-of-two per-channel gain on every binary layer", "B", ["THEORY M4", "B03"], ["M4"], "T1", {"quant.channel_gain": {"layers": "all_binary", "mode": "pow2"}}, ["channel-gain-pow2"],
       U, TO, "paired", "up; larger than M024 if the scale loss is general (M4) rather than first-layer (M1)", "least-squares per-channel alpha spread against per-tensor beta", targets_confirm=TOC)
single("M026", "Per-channel shift before activation quantizers (RSign)", "B", ["B13"], ["M4", "M6"], "T1", {"quant.pre_quant_shift": "channel"}, ["pre-quant-shift"],
       U, TO, "paired", "up; possibly lower achieved widths at equal accuracy", "dead-ReLU fraction, width map", targets_confirm=TOC)
WHY_WS = ("earlier runs that peaked unconstrained and were then squeezed to a budget far below the peak's cost did not recover; this entry "
          "screens at 5M, where the squeeze from the unconstrained start is much milder, and the 350k claim is tested only at confirm. If the "
          "student peaks at the start and falls by 5 pt or more while its EBOPs fall, that is recorded as the early-peak-then-squeeze pattern, not as a null")
single("M027", "Warm start from the FP teacher (two-stage)", "B", ["B11", "B12", "THEORY 5.1"], ["M2", "M3"], "T1", {"experiment.init_checkpoint": "teacher-fp32-d32h4"}, ["init-from-checkpoint", "weight-scheme-baselines"],
       U, TO, "unpaired (Welch): init not seed-derived; data order shared", "up; faster to a given accuracy", "C2I ratio (from teacher signs), latent-vs-binary gap", targets_confirm=TOC,
       note="needs the FP teacher job; " + WHY_WS)
# R: recipe
WHY_LR = ("earlier binary runs on the archived recipe (batch 256) collapsed within the first epochs at a peak LR of 2e-4 or more; the reference "
          "runs batch 2,790 (about 11x fewer steps per epoch), a cosine decay to 1e-6 inside each 500-epoch cycle and an EBOPs term. These "
          "are documented differences, not a shown cause. Both rungs sit between the collapse range and the reference 3e-3: they locate the "
          "stability edge if the reference collapses and test whether a lower peak helps if it does not")
single("M028", "Peak LR 1e-3", "R", ["R06"], ["M2", "M3"], "T0a", {"train.lr": 1e-3}, [],
       "earlier binary runs at LR >= 2e-4 on the archived recipe collapsed early", TO, "paired",
       "fewer divergences than the base; accuracy flat or up", "divergence and collapse count, FF ratio", ref_keys=["schedule"], targets_confirm=TOC,
       note="T0a assumes the reference schedule reads train.lr as the peak; " + WHY_LR)
single("M029", "Peak LR 3e-4", "R", ["R06"], ["M2", "M3"], "T0a", {"train.lr": 3e-4}, [],
       "as M028", TO, "paired", "no divergence; accuracy below the base if the base is stable", "as M028", ref_keys=["schedule"], targets_confirm=TOC, note=WHY_LR)
single("M030", "Linear warm-up, 10 epochs", "R", ["R04"], ["M2"], "T1", {"train.lr_warmup_epochs": 10}, ["lr-schedule-variants"],
       "untried on this schedule (the project's earlier recipe warms up for 1 epoch)", TO, "paired", "fewer early collapses; accuracy flat",
       "epoch of best validation in cycle 1, divergence count", targets_confirm=TOC)
single("M031", "Peak decay m_mul 0.85 per restart", "R", ["R03"], ["M3"], "T1", {"train.lr_m_mul": 0.85}, ["lr-schedule-variants"],
       U, TO, "paired (against the reference epoch-1,500 snapshot)", "fewer late divergences; accuracy flat or up", "FF spike height at each restart", horizon=1500, targets_confirm=TOC,
       note="cycle 1 is identical to the reference, so a 500-epoch screen would measure nothing")
single("M032", "No restarts: one cosine over the horizon", "R", ["THEORY 6.6", "R02", "R05"], ["M3"], "T0a", {"train.lr_cycle_epochs": "=horizon"}, [],
       U, TO, "paired (against the reference epoch-2,000 snapshot)", "sign open (THEORY 6.6)", "FF ratio against epoch; within-cycle position of the best checkpoint",
       horizon=2000, ref_keys=["schedule"], targets_confirm=TOC, note="the screen proxy is one cosine over 2,000 epochs; the confirm is one cosine over 7,000")
single("M033", "Weight decay 0.01 only", "R", ["R07", "THEORY 2"], ["M3", "M6"], "T0a", {"train.optimizer": "adam", "train.beta2": 0.999, "train.weight_decay": 0.01, "train.clipvalue": None}, [],
       "untried alone (the project's earlier optimizer changes three fields at once)", TO, "paired", "higher FF ratio, smaller beta; accuracy sign open", "FF ratio, beta trajectory",
       ref_keys=["optimizer"], targets_confirm=TOC, note="stated on the explicit optimizer path so weight_decay is never silently ignored")
single("M034", "EMA of latents, re-binarized for evaluation", "R", ["R08"], ["M3"], "T1", {"experiment.latent_ema_decay": 0.999}, ["latent-ema-eval"],
       U, TO, "paired", "small gain; less checkpoint jitter", "sign agreement EMA against raw; validation jitter across epochs", targets_confirm=TOC,
       note="decay 0.999 is a design choice in the Mean Teacher range (arXiv:1703.01780)")
single("M035", "Logit KD from an FP teacher (T 2, coefficient 0.5)", "R", ["R09"], ["M2"], "T1", {"experiment.distillation": {"teacher_artifact": "teacher-fp32-d32h4", "temperature": 2, "coefficient": 0.5}}, ["kd-constrained-runner", "weight-scheme-baselines"],
       "tried at N = 8 in the constrained ablation on the earlier recipe (project README); retried on the reference recipe with a new teacher trained on the student's training split",
       TO, "paired", "up, larger at the lower budget", "per-class gap to the teacher", targets_confirm=TOC,
       note="temperature and coefficient as in code/hgq2/configs/post_conference_budget350k-knowledge_distillation-w1a8.json")
single("M036", "Attention-map KD", "R", ["R09", "THEORY M5"], ["M5"], "T1", {"experiment.distillation": {"teacher_artifact": "teacher-fp32-d32h4", "temperature": 2, "coefficient": 0.0, "attention_coefficient": 1.0}}, ["kd-constrained-runner", "kd-attention-map", "weight-scheme-baselines"],
       U, TO, "paired", "attention stays non-uniform; accuracy up", "entropy/log n_valid against the teacher, attention-ablation delta", targets_confirm=TOC)
# P: inputs and data
single("M037", "eta/phi reflection augmentation", "P", ["R13", "P09"], ["M1"], "T1", {"data.augment": {"reflect_eta": True, "reflect_phi": True}}, ["augment-eta-phi-reflect"],
       "untried (no augmentation in any earlier run)", TO, "paired", "narrower seed spread; mean flat or up", "paired sd of the gap; per-class accuracy", targets_confirm=TOC)
single("M038", "pT gate off", "P", ["P02"], ["M5"], "T1", {"arch.pt_gate_gev": None}, ["pt-gate-threshold"],
       "untried on this recipe (earlier runs were ungated, under other recipes)", TS, "paired (same shapes; inputs differ)",
       "sign open: more real low-pT constituents against more noise in the statistics", "padding and gated fraction per jet, entropy/log n_valid",
       ref_keys=["pt-gate"], note="needs an ungated 90/10 cache; null means ungated and must stay explicit so that it overrides the reference 2 GeV gate")
single("M039", "Mask gated and padded keys", "P", ["P03", "THEORY M5"], ["M5"], "T1", {"arch.mask_gated_keys": True}, ["gated-key-mask"],
       U, TS, "paired", "up; the entropy diagnostic is no longer inflated by padding", "entropy/log n_valid; token-folding hazard noted for export")
NOHW = ("not a hardware candidate until the derived-feature path is costed: log pT and Delta R are computed off-model, so HGQ2 never bills them; "
        "the iso-EBOPs match excludes the feature computation, and on chip the squares or tables are uncosted (DSP risk)")
single("M040", "Derived features log pT and Delta R (5 inputs)", "P", ["P05", "THEORY M1"], ["M1"], "T1", {"arch.n_feat": 5, "arch.derived_features": ["log_pt", "delta_r"]}, ["derived-input-features"],
       "untried at fixed N and input set", TS, "paired-if-hash (input_proj fan-in changes); crosses the input set by design, labelled",
       "up (more expressible first-layer directions)", "distinct sign rows of input_proj", arch=dict(nf=5), note=NOHW)
single("M041", "Standardize over real slots only", "P", ["P06"], ["M6"], "T1", {"data.std_scope": "real_slots"}, ["std-real-slots"],
       U, TO, "paired", "up (the operating point of the first binary layer is no longer set by padding)", "input range at the first quantizer, saturation fraction", targets_confirm=TOC)
# A: architecture
single("M042", "d_model 16", "A", ["A01"], ["M1", "M7"], "T0", {"arch.d_model": 16}, [],
       "tried at N = 16 and 64 under the earlier quantizer set, never feasible at 350k", TS, "paired-if-hash",
       "350k: more headroom for widths than d32h4, still near-floor; 5M below d32h4", "effective width, EBOPs split", arch=dict(d=16))
single("M043", "FFN 64", "A", ["A03"], ["M1"], "T0", {"arch.ffn_dim": 64}, [],
       "tried at N = 8 at 350k (single seed, earlier quantizer): FFN 32 did better than 64 (README, post-conference table)", TS, "paired-if-hash",
       "350k worse; 5M flat or up", "EBOPs split", arch=dict(ffn=64))
single("M044", "Two blocks (L = 2)", "A", ["A02"], ["M1"], "T0", {"arch.n_layers": 2}, [],
       f"tried at 350k under the earlier quantizer set; no N >= 16 run was feasible ({PRIOR})", [T5M], "paired-if-hash", "5M: up if depth matters", "EBOPs split per block",
       arch=dict(L=2), if_floor="5M-only", note="0-bit floor above 350k")
single("M045", "Head with three hidden layers of width 32", "A", ["A08"], ["M1"], "T1", {"arch.head_dims": [32, 32, 32]}, ["head-dims"],
       U, TS, "paired-if-hash", "flat", "width map of the head", arch=dict(head=[32, 32, 32]))
single("M046", "No positional encoding (5M cell)", "A", ["A04"], ["M5"], "T0", {"arch.pos_enc": "none"}, [],
       "tried at N = 16 under the earlier quantizer set, inconclusive", [T5M],
       "paired if the kernel-hash check shows only the position table differs, else Welch",
       "flat or up (permutation invariance); gated tokens become exact duplicates", "duplicated-token count, entropy/log n_valid",
       note="completes the gate x PE x mask cube at 5M; the 350k base d24h2 already has no positional encoding")
# BL: labelled non-binary baselines (never the thesis)
single("M047", "Ternary absmean (labelled baseline)", "BL", ["B14"], ["M1", "M4"], "T1", {"quant.weight": "ternary_absmean"}, ["ternary-absmean", "weight-scheme-baselines"],
       "untried at this architecture (an early ternary attempt on an earlier dataset version was abandoned)", TS, "paired",
       "ternary >= binary on accuracy (the direction Sloot, FastML 2026, reports for an MLP)", "zero fraction per layer; how HGQ2 counts the zeros",
       kind="baseline", arch=dict(wbits_unknown=True), note="1-bit-alive floor not computable until it is known how HGQ2 bills a ternary weight")
single("M048", "int8 static weights, all layers (matched non-binary)", "BL", ["THEORY 6.2"], ["M1"], "T1", {"quant.weight": "int8_absmax"}, ["weight-scheme-baselines"],
       "run only in the pre-conference fixed-precision study (no EBOPs target)", TS, "paired",
       "above d32h4 at 5M; at 350k the weight bits cost more EBOPs", "EBOPs split; the same attention diagnostics", kind="baseline", arch=dict(wbits=8),
       note="despite its name, int8_absmax in this code base is a fixed fixed<8,3> grid (code/hgq2/bnhgq2/qat.py, _static_w8); the 1-bit-alive floor prices every dense and head layer at b_w = 8")
single("M049", "HGQ learnable weight widths (the Laatu et al. weight scheme)", "BL", ["THEORY 6.2"], ["M5", "M7"], "T2", {"quant.weight": "hgq_learnable"}, ["hgq-learnable-weights"],
       "untried in this pipeline", TS, "paired-if-hash",
       "collapses at 350k as in Laatu et al. if the collapse is budget geometry", "Q/K/V 0-bit fractions, entropy, ablation delta", kind="baseline",
       arch=dict(wbits_unknown=True), note="1-bit-alive floor not computable: learned weight widths, 0 bits reachable, no fixed b_w")
single("M050", "8-bit input_proj only", "BL", ["THEORY M1, 6.3"], ["M1"], "T1", {"quant.layer_weight_override": {"input_proj": "int8_absmax"}}, ["weight-scheme-baselines"],
       U, TS, "paired", "the gap to the base measures the fan-in-3 first-layer loss", "distinct sign rows (base) against this arm", kind="baseline", arch=dict(wbits_in=8),
       note="the 1-bit-alive floor prices input_proj at b_w = 8")

ENTRY = {e["id"]: e for e in S}

# ---------------------------------------------------------------- combinations
C = []
TIER_ORDER = {"T0": 0, "T0a": 1, "T1": 2, "T2": 3}


def merge(ids):
    d, p, rk, arch = {}, [], [], {}
    for i in ids:
        e = ENTRY[i]
        for k_, v in e["config_delta"].items():
            if k_ in d and d[k_] != v and isinstance(d[k_], dict) and isinstance(v, dict):
                d[k_] = {**d[k_], **v}
            else:
                d[k_] = v
        p += [x for x in e["code_changes"] if x not in p]
        rk += [x for x in e["ref_keys"] if x not in rk]
        arch.update(e["arch"])
    return d, p, rk, arch


def combo(id_, name, fam, ids, ladder, targets, pred, diag, horizon=None, pairing=None, kind="package",
          targets_confirm=None, wave="W3", note="", if_floor="drop", base=None, extra_delta=None, ref_keys=None):
    d, p, rk, arch = merge(ids)
    rk += [x for x in (ref_keys or []) if x not in rk]
    if extra_delta:
        d.update(extra_delta)
    ids_ct = ids or ["M033"]
    hz = horizon or max([ENTRY[i]["horizon_screen"] for i in ids] + [500])
    pr = pairing or ("paired" if all(ENTRY[i]["pairing"].startswith("paired (") or ENTRY[i]["pairing"] == "paired" for i in ids)
                     else "strictest component rule (paired / paired-if-hash / Welch)")
    C.append(dict(id=id_, name=name, family=fam, tier=kind, cards=[], mechanisms=sorted({m for i in ids for m in ENTRY[i]["mechanisms"]}),
                  code_tier=max([ENTRY[i]["code_tier"] for i in ids_ct], key=lambda t: TIER_ORDER[t]),
                  combo_of=list(ids), ladder=ladder, config_delta=d, code_changes=p, ref_keys=rk,
                  tried="untried (combination)", targets=targets, targets_confirm=targets_confirm or targets, pairing=pr,
                  prediction=pred, diagnostic=diag, horizon_screen=hz, arch=arch,
                  if_floor_fails=if_floor, wave=wave, note=note, base=base, extra_delta=extra_delta or {}))
    return id_


cid = [51]


def nid():
    s = f"M{cid[0]:03d}"
    cid[0] += 1
    return s


# floor crosses (THEORY 5.4)
X_h2tb = combo(nid(), "Two heads + table floor 2 bits", "F", ["M001", "M003"], ["M001", "M003"], TS, "350k headroom beyond either single", "EBOPs split, Q/K/V 0-bit fractions")
combo(nid(), "One head + table floor 2 bits", "F", ["M002", "M003"], ["M002", "M003"], TS, "largest attention-keeping headroom among softmax arms", "as above")
X_lh2 = combo(nid(), "Linformer k8 + two heads", "F", ["M004", "M001"], ["M004", "M001"], TS, "feasible with the most headroom; accuracy against M004 flat", "EBOPs split, entropy over projected keys")
X_ltb = combo(nid(), "Linformer k8 + table floor 2 bits", "F", ["M004", "M003"], ["M004", "M003"], TS, "redundant: the table term is already small under Linformer", "EBOPs split")
combo(nid(), "Q/K width floor + two heads", "F", ["M007", "M001"], ["M007 (5M only)", "M001"], TS, "feasible at 350k with attention forced alive, but near-floor", "Q/K 0-bit fraction 0; entropy; ablation delta",
      note="at 350k the decomposition is one-sided: M007 alone is statically infeasible there")
combo(nid(), "Q/K width floor + Linformer k8", "F", ["M007", "M004"], ["M007 (5M only)", "M004"], TS, "feasible at 350k with attention alive", "as above",
      note="at 350k the decomposition is one-sided")
combo(nid(), "Attention-group weight + two heads", "F", ["M008", "M001"], ["M008", "M001"], TS, "attention survives at 350k", "EBOPs split by group")
combo(nid(), "N = 32 + two heads (crosses N)", "F", ["M009", "M001"], ["M009", "M001"], TS, "every channel alive at 350k with attention", "as M009", pairing="unpaired (Welch); crosses N, labelled")
combo(nid(), "ReLU/N attention + Q/K width floor", "F", ["M005", "M007"], ["M005", "M007 (5M only)"], TS, "softmax-free attention kept alive at 350k", "ablation delta",
      note="at 350k the decomposition is one-sided")
combo(nid(), "Floor stack: Linformer + two heads + table 2 bits", "F", ["M004", "M001", "M003"], [X_lh2, X_ltb, X_h2tb], TS, "no gain over the best pair (redundant)", "EBOPs split")
# KD x progressive binarization (THEORY 5.1)
X_kdw = combo(nid(), "Logit KD + warm start from teacher", "R", ["M035", "M027"], ["M035", "M027"], TO, "synergy; larger at the low budget (tested at confirm)", "per-class gap to teacher, C2I",
              pairing="unpaired (Welch): warm start", targets_confirm=TOC)
X_akw = combo(nid(), "Attention KD + warm start", "R", ["M036", "M027"], ["M036", "M027"], TO, "synergy", "entropy against teacher", pairing="unpaired (Welch): warm start", targets_confirm=TOC)
combo(nid(), "Logit KD from the int8 teacher (staged, BiT-style)", "R", ["M035"], ["M035", "M048"], TO, "a closer teacher helps more than the FP teacher", "per-class gap to both teachers",
      extra_delta={"experiment.distillation": {"teacher_artifact": "teacher-int8-d32h4", "temperature": 2, "coefficient": 0.5}}, targets_confirm=TOC,
      note="teacher variant of M035; needs the int8 teacher job")
combo(nid(), "Teacher stack: logit + attention KD + warm start", "R", ["M035", "M036", "M027"], [X_kdw, X_akw], TO, "no gain beyond the best pair", "as components",
      pairing="unpaired (Welch): warm start", targets_confirm=TOC)
# gain x widths (THEORY 5.2, 5.6)
combo(nid(), "Per-channel pow2 gain + per-value widths", "B", ["M025", "M011"], ["M025", "M011"], TO, "synergy (the gain sets relative contributions, the widths re-adapt)", "gain histogram, width map", targets_confirm=TOC)
combo(nid(), "Per-channel pow2 gain + shift (binary-safe affine)", "B", ["M025", "M026"], ["M025", "M026"], TO, "additive or mildly synergistic; a binary-safe analogue of a fused batch norm", "dead-ReLU fraction, gain histogram", targets_confirm=TOC)
X_fg = combo(nid(), "Derived features + pow2 input_proj gain", "P", ["M040", "M024"], ["M040", "M024"], TO, "redundant at the margin (both enlarge the input directions, THEORY 5.6)", "distinct sign rows", targets_confirm=TOC)
# restarts x latent treatments (THEORY 5.3)
X_nrc = combo(nid(), "No restarts + latent clip", "R", ["M032", "M021"], ["M032", "M021"], TO, "partly redundant", "FF ratio", targets_confirm=TOC)
combo(nid(), "No restarts + weight decay", "R", ["M032", "M033"], ["M032", "M033"], TO, "partly redundant", "FF ratio, beta trajectory", targets_confirm=TOC)
combo(nid(), "Latent clip + weight decay", "R", ["M021", "M033"], ["M021", "M033"], TO, "redundant (both cap inertia)", "latent histogram", targets_confirm=TOC)
combo(nid(), "Bop + no restarts", "B", ["M020", "M032"], ["M020", "M032"], TO, "restarts do not act on Bop; no interaction expected", "FF ratio", targets_confirm=TOC)
X_ede1 = combo(nid(), "EDE annealed once + no restarts", "B", ["M019", "M032"], ["M019", "M032"], TO, "up over EDE per cycle only if restarts fight the anneal", "latent-vs-binary gap", targets_confirm=TOC,
               extra_delta={"quant.ste_ede_period": "=horizon"})
combo(nid(), "Inertia stack: EDE + latent clip + no restarts", "B", ["M019", "M021", "M032"], [X_nrc, X_ede1], TO, "no gain beyond the best pair", "FF ratio", targets_confirm=TOC,
      extra_delta={"quant.ste_ede_period": "=horizon"})
# optimizer decomposition
combo(nid(), "clipvalue 1 + beta2 0.98 (earlier optimizer minus weight decay)", "R", [], ["M033", "M103"], TO, "with M033 and M103: a 2x2 in {weight decay} x {clip + beta2}; additivity checked", "divergence count, FF ratio",
      extra_delta={"train.optimizer": "adam", "train.beta2": 0.98, "train.weight_decay": None, "train.clipvalue": 1.0}, targets_confirm=TOC, ref_keys=["optimizer"])
# gate x PE x mask: the full 2^3 at 5M; at 350k the gate x mask square on d24h2
NO_PE_350 = "M046 (5M; the 350k cell is dropped because d24h2 has no positional encoding)"
X_gp = combo(nid(), "Gate off + no PE", "P", ["M038", "M046"], ["M038", NO_PE_350], TS, "2^3 cell", "entropy/log n_valid, duplicated-token count")
X_gm = combo(nid(), "Gate off + masking", "P", ["M038", "M039"], ["M038", "M039"], TS, "2^3 cell", "as above")
X_pm = combo(nid(), "No PE + masking", "P", ["M046", "M039"], [NO_PE_350, "M039"], TS, "2^3 cell (gated tokens are exact duplicates without PE)", "as above")
combo(nid(), "Gate off + no PE + masking", "P", ["M038", "M046", "M039"], [X_gp, X_gm, X_pm], TS, "2^3 cell", "as above")
X_ms = combo(nid(), "Masking + real-slot standardization", "P", ["M039", "M041"], ["M039", "M041"], TO, "natural pair (absent slots treated consistently)", "entropy/log n_valid", targets_confirm=TOC)
combo(nid(), "First-layer stack: features + gain + mask + real-slot std", "P", ["M040", "M024", "M039", "M041"], [X_fg, X_ms], TO, "no gain beyond the best pair", "distinct rows, entropy/log n_valid", targets_confirm=TOC)

# FF1: 2^(5-1), resolution V, generator E = -ABCD (I = -ABCDE); the all-low cell is the in-wave d32h4 5M replica
FF1 = dict(A="M019", B="M040", C="M035", D="M042", E="M037")
ff1_cells = []
for a, b, c, d in itertools.product([-1, 1], repeat=4):
    lv = dict(A=a, B=b, C=c, D=d, E=-a * b * c * d)
    ff1_cells.append([f for f in "ABCDE" if lv[f] == 1])
assert sorted(len(h) for h in ff1_cells) == [0] + [2] * 10 + [4] * 5
for highs in sorted([h for h in ff1_cells if h], key=lambda h: (len(h), h)):
    ids = [FF1[f] for f in highs]
    combo(nid(), "FF1 cell " + "".join(highs) + ": " + " + ".join(ENTRY[i]["name"].split(" (")[0] for i in ids), "FF1", ids,
          ["FF1 linear model (15 new cells + the in-wave d32h4 5M replica as the all-low cell; seed blocks)"] + ids, [T5M],
          "main effects and two-factor interactions estimated from the 16-cell fraction; additivity (THEORY 5, 'expected additive') tested, not assumed",
          "each component's diagnostic", kind="factorial-cell", targets_confirm=TOC,
          pairing="paired-if-hash (seed blocks); analysed as a blocked 2^(5-1) design", base="d32h4", horizon=500)

# FF2: 2^(4-1), resolution IV, generator D = ABC (I = ABCD), on d24h2 at 350k; the all-low cell is the in-wave d24h2 replica
FF2 = dict(A="M011", B="M012", C="M016", D="M045")
ff2_cells = []
for a, b, c in itertools.product([-1, 1], repeat=3):
    lv = dict(A=a, B=b, C=c, D=a * b * c)
    ff2_cells.append([f for f in "ABCD" if lv[f] == 1])
assert sorted(len(h) for h in ff2_cells) == [0] + [2] * 6 + [4]
for highs in sorted([h for h in ff2_cells if h], key=lambda h: (len(h), h)):
    ids = [FF2[f] for f in highs]
    combo(nid(), "FF2 cell " + "".join(highs) + " on d24h2: " + " + ".join(ENTRY[i]["name"].split(" (")[0] for i in ids), "FF2", ids,
          ["FF2 linear model (7 new cells + the in-wave d24h2 350k replica as the all-low cell; seed blocks)"] + ids, [T350],
          "main effects clear of two-factor interactions; the interactions are aliased in pairs AB = CD, AC = BD, AD = BC", "each component's diagnostic",
          kind="factorial-cell", pairing="paired-if-hash against d24h2 at 350k (seed blocks)", base="d24h2", targets_confirm=[T350],
          extra_delta={"arch.d_model": 24, "arch.n_heads": 2, "arch.pos_enc": "none"}, horizon=500)

# completes the {weight decay} x {clip + beta2} 2x2 at 5M
combo(nid(), "Earlier optimizer at 5M (beta2 0.98, weight decay 0.01, clipvalue 1)", "R", [], ["M033", "M074"], TO,
      "2x2 corner; with the base, M033 and M074 gives the additivity check of the three optimizer fields", "divergence count, FF ratio, beta trajectory",
      extra_delta={"train.optimizer": "adam", "train.beta2": 0.98, "train.weight_decay": 0.01, "train.clipvalue": 1.0}, targets_confirm=TOC, ref_keys=["optimizer"])

ALL = S + C
for e in C:
    notes = [e["note"]] if e["note"] else []
    if "M027" in e["combo_of"]:
        notes.append(WHY_WS)
    if "M020" in e["combo_of"]:
        notes.append(BOP_NOTE)
    if "M040" in e["combo_of"]:
        notes.append(NOHW)
    e["note"] = "; ".join(notes)

# ---------------------------------------------------------------- floors and 350k roles per entry
ROLE = {
    "d24h2-base": "factorial cell built on d24h2 at 350k and paired with it",
    "own-arch": ("floor family: runs on its own d32h4-derived architecture; read for feasibility against d32h4 at 350k (count rule G2) "
                 "and for accuracy against d24h2 at 350k (Welch, labelled cross-architecture); never against the accuracy of d32h4 at 350k"),
    "near-floor-d24h2": ("near-floor on d32h4 (headroom below 0.10): the change is applied to d24h2 at 350k and paired with it "
                         "(accuracy screen, G0-G3); d32h4 at 350k stands for the d32h4 version, which is not run"),
    "near-floor-d24h2-probe": ("near-floor on d32h4 and on d24h2, or not computable: applied to d24h2 at 350k as a feasibility probe only "
                               "(G2 counts, EBOPs split, attention state; no accuracy reading)"),
}
BASE_350 = {
    "d24h2-base": "d24h2@350k",
    "near-floor-d24h2": "d24h2@350k (change applied to d24h2; accuracy screen)",
    "near-floor-d24h2-probe": "d24h2@350k (change applied to d24h2; feasibility probe)",
    "own-arch": "own architecture; feasibility against d32h4@350k, accuracy against d24h2@350k (Welch)",
}


def arch_args(e, on_d24=False):
    a = dict(D24 if (on_d24 or e.get("base") == "d24h2") else D32)
    ar = dict(e["arch"])
    extra = ar.pop("extra", None)
    a.update(ar)
    return a, extra


def hround(v):
    return round(v, 3) if isinstance(v, float) else v


for e in ALL:
    a, extra = arch_args(e)
    f0, f1 = arch_floors(**a)
    fd = {"floor0": f0, "floor1_alive": f1, "extra_term": extra, "label": LABEL_FLOOR}
    if a.get("wbits_unknown"):
        fd["floor1_note"] = "not computable: weight bits not fixed"
    h = {str(t): headroom(f0, f1, t) for t in e["targets"]}
    fd["headroom_by_target"] = {k_: hround(v) for k_, v in h.items()}
    if T350 in e["targets"] and h.get(str(T350)) == "STATIC_INFEASIBLE":
        e["targets"] = [t for t in e["targets"] if t != T350]
        e["note"] = "; ".join(x for x in [e["note"], "350k removed: 0-bit floor at or above 350k"] if x)
    role = None
    if T350 in e["targets"]:
        h350 = h.get(str(T350))
        if e.get("base") == "d24h2":
            role = "d24h2-base"
        elif (isinstance(h350, float) and h350 < NEAR_FLOOR_H) or (h350 is None and f1 is None and f0 is not None):
            role = "near-floor-d24h2"
        else:
            role = "own-arch"
        if role == "near-floor-d24h2":
            aE, _ = arch_args(e, on_d24=True)
            g0, g1 = arch_floors(**aE)
            hE = headroom(g0, g1, T350)
            fd["on_d24h2"] = {"floor0": g0, "floor1_alive": g1, "headroom_350000": hround(hE), "label": LABEL_FLOOR}
            if not (isinstance(hE, float) and hE >= NEAR_FLOOR_H):
                e["note"] = "; ".join(x for x in [e["note"], "near-floor or not computable on d24h2 as well: the 350k cell is a feasibility probe"] if x)
                role = "near-floor-d24h2-probe"
            if e["config_delta"].get("arch.pos_enc") == "none":
                e["targets"] = [t for t in e["targets"] if t != T350]
                e["note"] = "; ".join(x for x in [e["note"], "350k dropped: d24h2 has no positional encoding, so this cell duplicates a cell of the gate x mask square"] if x)
                del fd["on_d24h2"]
                role = None
    e["role_key"] = role
    e["floor"] = fd
    e["seeds_screen"] = [1, 2, 3, 4]
    e["seeds_confirm"] = list(range(1, 9))

# ---------------------------------------------------------------- architecture floor table
ARCH_ROWS = [
    ("d32h4 (d_model 32, 4 heads, learned PE): the 5M base and the 350k feasibility reference", {}),
    ("d24h2 (d_model 24, 2 heads, no PE): the 350k base", dict(d=24, H=2)),
    ("d32h4 with FFN 64 (M043)", dict(ffn=64)),
    ("d32h4 with d_model 16 (M042)", dict(d=16)),
    ("d32h4, head 32/32/32 (M045)", dict(head=[32, 32, 32])),
    ("d32h4, 5 input features (M040)", dict(nf=5)),
    ("d32h4, int8 weights (M048)", dict(wbits=8)),
    ("d32h4, int8 input_proj (M050)", dict(wbits_in=8)),
    ("d32h4 with 2 heads (M001)", dict(H=2)),
    ("d32h4 with 1 head (M002)", dict(H=1)),
    ("d32h4, table floor 2 bits (M003)", dict(tb=2)),
    ("d32h4, Linformer k = 8 (M004)", dict(attn="linformer", k=8)),
    ("d32h4, ReLU/N attention (M005)", dict(attn="relu")),
    ("d32h4 at N = 32 (M009)", dict(N=32)),
    ("d32h4 + Q/K floor 1 bit (M007)", dict(qk_min=1)),
    ("d32h4, L = 2 (M044)", dict(L=2)),
    ("2 heads + table 2 bits (M051)", dict(H=2, tb=2)),
    ("1 head + table 2 bits (M052)", dict(H=1, tb=2)),
    ("Linformer + 2 heads (M053)", dict(attn="linformer", k=8, H=2)),
    ("Linformer + table 2 bits (M054)", dict(attn="linformer", k=8, tb=2)),
    ("Q/K floor + 2 heads (M055)", dict(qk_min=1, H=2)),
    ("Q/K floor + Linformer (M056)", dict(qk_min=1, attn="linformer", k=8)),
    ("N = 32 + 2 heads (M058)", dict(N=32, H=2)),
    ("ReLU/N + Q/K floor (M059)", dict(attn="relu", qk_min=1)),
    ("floor stack: Linformer + 2 heads + table 2 bits (M060)", dict(attn="linformer", k=8, H=2, tb=2)),
]
ARCH_FLOORS = []
for label, kw in ARCH_ROWS:
    a = dict(D32)
    a.update(kw)
    f0, f1 = arch_floors(**a)
    hh = headroom(f0, f1, T350)
    if hh == "STATIC_INFEASIBLE":
        cls = "STATIC_INFEASIBLE"
    elif f1 is not None and f1 < T350:
        cls = "every channel alive at 1 bit"
    elif isinstance(hh, float) and hh < NEAR_FLOOR_H:
        cls = "near-floor"
    else:
        cls = "headroom >= 0.10"
    ARCH_FLOORS.append({"architecture": label, "floor0": f0, "floor1_alive": f1, "headroom_350000": hround(hh),
                        "class_350000": cls, "label": LABEL_FLOOR})

# ---------------------------------------------------------------- budget (run-epochs; seconds per epoch not measured)
SEED_CHOICES = (4, 6, 8)


def run_epochs(entries, seeds=4):
    return sum(seeds * len(e["targets"]) * e["horizon_screen"] for e in entries)


W2 = list(S)
W3 = list(C)
DRIFT = {
    "W2": [("d24h2@350k", T350, 1000, "350k accuracy cells and M015 at 1,000 epochs; drift"),
           ("d32h4@350k", T350, 500, "floor-family feasibility count (G2); drift"),
           ("d32h4@5M", T5M, 2000, "5M cells, M031 at 1,500 and M032 at 2,000 epochs")],
    "W3": [("d24h2@350k", T350, 500, "350k packages; FF2 all-low cell"),
           ("d32h4@350k", T350, 500, "floor crosses' feasibility count (G2); drift"),
           ("d32h4@5M", T5M, 2000, "5M packages, M068, M069, M071-M073 at 2,000 epochs; FF1 all-low cell")],
}
REP = {w: {} for w in DRIFT}
for w, reps in DRIFT.items():
    for _, _, hz, _ in reps:
        REP[w][hz] = REP[w].get(hz, 0) + 1
TEACHERS = 2 * 1 * 2000   # FP and int8 teachers, 1 seed, 2,000 epochs


def budget(n):
    rw2 = n * sum(h * c for h, c in REP["W2"].items())
    rw3 = n * sum(h * c for h, c in REP["W3"].items())
    re_ = dict(W2=run_epochs(W2, n) + rw2, W3=run_epochs(W3, n) + rw3, teachers=TEACHERS)
    re_["screen_total"] = re_["W2"] + re_["W3"] + re_["teachers"]
    return re_


RE_BY_N = {n: budget(n) for n in SEED_CHOICES}


def runs_by_h(entries, n=4):
    out = {}
    for e in entries:
        if e["targets"]:
            out[e["horizon_screen"]] = out.get(e["horizon_screen"], 0) + n * len(e["targets"])
    return out


def wall(entries, reps, P, n=4):
    rb = runs_by_h(entries, n)
    for h, c in reps.items():
        rb[h] = rb.get(h, 0) + n * c
    return sum(math.ceil(m / (P * K)) * h for h, m in rb.items())


WALL = {(n, P): wall(W2, REP["W2"], P, n) + wall(W3, REP["W3"], P, n) + 2000 for n in SEED_CHOICES for P in (2, 10)}
CHEAP = [e for e in S if e["code_tier"] in ("T0", "T0a", "T1") and e["tier"] == "single" and e["id"] not in ("M027", "M035", "M036") and e["targets"]]
RE_CHEAP = sum(3 * 1 * e["horizon_screen"] for e in CHEAP) + 3 * 500 * 2
CONF_CELLS = 12
RE_CONF = CONF_CELLS * 8 * 7000

# ---------------------------------------------------------------- hardware-risk labels
NO_EXPORT_GUARDED = {"quant.binary_center": "uncentered binarizer", "quant.beta_mode": "beta modes",
                     "quant.channel_gain": "per-channel gain", "arch.head_dims": "head widths",
                     "arch.pre_block_act": "tanh table", "quant.weight": "non-binary weight scheme",
                     "quant.layer_weight_override": "non-binary weight scheme"}
NO_EXPORT_OTHER = {"arch.attn_kind": "Linformer or ReLU/N attention", "arch.body": "Deep Sets body",
                   "arch.mask_gated_keys": "runtime key mask"}


def hw_labels(e):
    d = e["config_delta"]
    out = []
    if "quant.channel_gain" in d:
        out.append("DSP audit required before any hardware claim: per-channel power-of-two gain")
    g = sorted({v for k_, v in NO_EXPORT_GUARDED.items() if k_ in d})
    if g:
        out.append("no HLS export path yet; export is refused for: " + ", ".join(g))
    o = sorted({v for k_, v in NO_EXPORT_OTHER.items() if k_ in d})
    if o:
        out.append("no HLS export path yet: " + ", ".join(o))
    if "arch.derived_features" in d:
        out.append("not a hardware candidate until the derived-feature path is costed (off-model features, DSP risk)")
    return out


for e in ALL:
    e["hw_labels"] = hw_labels(e)

# ---------------------------------------------------------------- write
entries = []
for e in ALL:
    base = {"350000": BASE_350.get(e["role_key"], "no 350k screen cell"), "5000000": "d32h4@5M"}
    if T1M4 in e["targets"] + e["targets_confirm"]:
        base["1400000"] = "d32h4 at 1.4M, paired by seed to d32h4@350k (target differs)"
    entries.append({
        "id": e["id"], "name": e["name"], "family": e["family"], "tier": e["tier"], "code_tier": e["code_tier"],
        "combo_of": e["combo_of"], "targets": e["targets"], "targets_confirm": list(e["targets_confirm"]),
        "seeds_screen": e["seeds_screen"], "seeds_confirm": e["seeds_confirm"],
        "horizon_screen_epochs": e["horizon_screen"], "base": base,
        "config_delta": e["config_delta"], "extra_delta": e.get("extra_delta", {}), "code_changes": e["code_changes"],
        "reference_code_required": e["ref_keys"], "pairing": e["pairing"], "tried": e["tried"],
        "prediction": e["prediction"], "mechanism_diagnostic": e["diagnostic"], "mechanisms": e["mechanisms"],
        "cards": e["cards"], "ladder": e.get("ladder", []), "floor": e["floor"],
        "role_350k": e["role_key"], "role_350k_text": ROLE.get(e["role_key"]), "if_floor_fails": e["if_floor_fails"],
        "wave": e["wave"], "note": e["note"], "hw_labels": e["hw_labels"],
    })

catalogue = {
    "delta_id": "2026-09-26-delta",
    "status": "designed; nothing has been trained",
    "reference": {
        "architectures": {
            "d32h4": "d_model 32, 4 heads, 1 block, FFN 32, learned positional encoding, ReLU FFN, global average pooling, no normalization layers",
            "d24h2": "d_model 24, 2 heads, 1 block, FFN 32, no positional encoding, otherwise as d32h4",
        },
        "bases": {"d24h2@350k": "the 350k accuracy base", "d32h4@350k": "the 350k feasibility reference (near its static floor)",
                  "d32h4@5M": "the 5M base"},
        "weights": "binary_absmean: centered absmean, one scale per tensor, weights in {-beta, +beta}",
        "quantizers": ("activation and Q/K/V stream widths learned per channel, wrap-around overflow, 0 bits reachable; learned softmax output; "
                       "trainable exp and inverse tables with a 4-bit minimum; the exponential's input quantizer saturates"),
        "recipe": ("Adam with framework defaults; cosine restarts with peak LR 3e-3 every 500 epochs; batch 2,790; 7,000 epochs; "
                   "PID controller on beta to the EBOPs target"),
        "data": ("HLS4ML LHC jet dataset, N = 64 constituents, features pT, eta_rel, phi_rel; constituents with pT < 2 GeV zeroed; "
                 "90/10 split of the 620,000-jet training archive (62,000 validation jets); 260,000 held-out jets"),
        "selection": "validation accuracy among feasible, non-degenerate checkpoints",
        "status": "the reference configuration is trained in a separate study; no result from it is used here",
    },
    "drift_replicas": {
        "rule": ("every screen wave re-runs its base configurations at the Delta code version, seeds 1-n, each to the longest horizon that pairs "
                 "with it; snapshots every 500 epochs give every shorter horizon. The replicas take the first pods of the wave, and the paired sd "
                 "of replica minus reference snapshot is read at epoch 500 before any other entry of the wave starts. If the 95 % interval of "
                 "replica minus reference excludes 0, or |mean| > 0.3 pt, the wave's primary pairing switches to the replica at every horizon"),
        **{w: [{"base": a, "target": t, "horizon_epochs": h, "serves": why} for a, t, h, why in reps] for w, reps in DRIFT.items()},
    },
    "architecture_floors": ARCH_FLOORS,
    "screen_seed_rule": {
        "seed_choices": list(SEED_CHOICES), "lower_bound": 4, "upper_bound": 8,
        "rule": ("sd_plan = sqrt(2) * epoch-500 validation-accuracy sd of the base (d32h4@5M for 5M, d24h2@350k for 350k; seeds 1-8, feasible "
                 "only), or max(sd_plan, sd_null) once the drift replicas are read; n = the smallest of 4, 6, 8 with k_joint(n, alpha_eff) * sd_plan "
                 "<= g_0 = 0.3 pt, where k_joint sizes the whole advance rule, (i) BH and (ii) mean g >= g_0 / 2, for 80 % power at a true gap of "
                 "g_0; alpha_eff = q / m one-sided (q = 0.10, m = cells with a G3 test in the wave x target family); n_W3 <= n_W2 at the same "
                 "target; if no n qualifies, the wave runs at n = 4 as a declared ranking with no advance claims"),
        "gate_ii": "mean g >= g_0 / 2 (0.15 pt); g_0 is the minimum detectable effect the seed count is sized for, not the gate",
        "k_table": "screen_power.py",
    },
    "config_delta_semantics": ("dotted config keys relative to the base named in base[target]. '=horizon' means the screen horizon (7,000 at "
                               "confirm). '<set before launch>' marks a value fixed in the wave's own pre-registration. null means unset "
                               "(train.clipvalue, train.weight_decay) or ungated (arch.pt_gate_gev). T0a keys are placeholder names until the "
                               "reference configuration's code is published; reference_code_required names the piece each depends on"),
    "tier_legend": {"single": "one method against its base", "baseline": "labelled non-binary comparand, never the thesis",
                    "package": "pre-registered combination, read against its decomposition ladder", "factorial-cell": "cell of FF1 or FF2"},
    "code_tier_legend": {"T0": "config-only on a key the constrained runner reads today",
                         "T0a": "config-only once the reference configuration's code is published",
                         "T1": "small patch, about 80 lines or less", "T2": "new module"},
    "patches": {k_: {"code_tier": v[0], "size_lines": v[1], "acts_on": v[2], "what": v[3]} for k_, v in PATCHES.items()},
    "always_on_diagnostics": DIAG_ALWAYS,
    "prerequisite_jobs": {
        "teacher-fp32-d32h4": {"what": "FP teacher: d32h4, N = 64, unquantized weights, the reference activation and softmax quantizers at their init widths, same split and recipe, no EBOPs pressure", "epochs": 2000, "seeds": [101], "selection": "validation accuracy"},
        "teacher-int8-d32h4": {"what": "int8 teacher: as the FP teacher with quant.weight int8_absmax", "epochs": 2000, "seeds": [101]},
        "caches": ["gated 90/10 N = 32", "ungated 90/10 N = 64", "N = 64 with derived features", "N = 64 with real-slot standardization"],
    },
    "budget_run_epochs": {str(n): RE_BY_N[n] for n in SEED_CHOICES},
    "entries": entries,
}
with open(f"{OUT}/delta.json", "w") as fh:
    json.dump(catalogue, fh, indent=1)
    fh.write("\n")

# ---------------------------------------------------------------- checks and summary
slugs = [p for e in ALL for p in e["code_changes"]]
assert all(p in PATCHES for p in slugs), set(slugs) - set(PATCHES)
ids = [e["id"] for e in ALL]
assert len(ids) == len(set(ids)) == 103

print("entries", len(ALL), "singles", len(S), "combinations", len(C))
print("tier", dict(Counter(e["tier"] for e in ALL)))
print("family", dict(Counter(e["family"] for e in ALL)))
print("code tier, all", dict(Counter(e["code_tier"] for e in ALL)))
print("code tier, singles", dict(Counter(e["code_tier"] for e in S)))
print("code tier, combinations", dict(Counter(e["code_tier"] for e in C)))
print("350k roles", dict(Counter(e["role_key"] for e in ALL if e["role_key"])))
for n in SEED_CHOICES:
    print("n", n, "run-epochs", RE_BY_N[n], "wall (units of s_e) P=2", WALL[(n, 2)], "P=10", WALL[(n, 10)])
print("runs W2", sum(4 * len(e["targets"]) for e in W2), "+ replicas", 4 * sum(REP["W2"].values()),
      "runs W3", sum(4 * len(e["targets"]) for e in W3), "+ replicas", 4 * sum(REP["W3"].values()))
print("runs by horizon at n = 4: W2", runs_by_h(W2), "W3", runs_by_h(W3))
print("cheap version: entries", len(CHEAP), "run-epochs", RE_CHEAP, "; confirm (12 cells)", RE_CONF)
used = sorted(set(slugs) | set(DIAG_ALWAYS) | {"diag-sign-flips", "diag-beta-trajectory", "diag-input-proj-rows", "diag-latent-binary-gap"})
print("planned changes used", len(used), "defined", len(PATCHES))
for r in ARCH_FLOORS:
    print("floor", r["architecture"], r["floor0"], r["floor1_alive"], r["headroom_350000"], r["class_350000"])
for e in ALL:
    if e["role_key"] in ("near-floor-d24h2", "near-floor-d24h2-probe"):
        print("on d24h2", e["id"], e["floor"]["on_d24h2"])
