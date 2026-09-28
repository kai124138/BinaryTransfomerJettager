# Delta code: patches and modules for the method queue

This directory implements part of the [Delta catalogue](../../docs/delta/README.md). It holds 25
patches against the September screen code in
[`code/constituent-study-20260922`](../constituent-study-20260922/), six new modules, and the
checks that test them. No Delta configuration has been trained with this code. The checks run on
synthetic inputs on a CPU, and nothing they print is a result.

Every Delta key is opt-in, so a config with no Delta key should build and train as before. On
seven screen configurations the patched tree reproduces the unpatched tree's initial kernels,
model, EBOPs, predictions and one optimizer epoch, hash for hash (the invariance check below).

## Building the patched tree

```bash
code/delta/apply.sh /tmp/delta-build          # the directory must be empty or absent
code/delta/apply.sh /tmp/delta-0006 0006      # stop after patch 0006
```

`apply.sh` checks a content hash of `code/constituent-study-20260922`, copies it to
`<dir>/code`, adds `newmods/`, and applies the patches in order with `git apply --3way` inside a
throwaway git repository in `<dir>`. After a full build it copies `tests/` to `<dir>/code/tests`.
It never writes to this repository. The patch paths are rooted at `code/` of the build directory.
To patch the repository copy in place instead, copy `newmods/*.py` into
`code/constituent-study-20260922/newmods/` and run
`git apply -p2 --directory=code/constituent-study-20260922` on each patch in order.

## The patches

Patch names are the `code_changes` names in [`delta.json`](../../docs/delta/catalogue/delta.json),
except the three rows marked "infrastructure". The last column says whether `convert_binary.py`
refuses to export a config that sets the key (see [Export to HLS](#export-to-hls)).

| patch | what it changes | export |
| --- | --- | --- |
| 0001 delta-keys-registry (infrastructure) | `bnhgq2/delta_keys.py`: one table of Delta keys and their validators, called from `run_engram.validate_cfg`; `convert_binary.py` asks it for export blockers | |
| 0002 binarizer-center-flag | `quant.binary_center: false` binarizes the uncentered latent; `binarize.py` follows the forward pass | refused |
| 0003 ste-variants | `quant.ste`: `bounded` (default), `clip_identity` or `ede`; `quant.ste_ede_period` restarts the EDE temperature ramp (0.1 to 10, as in IR-Net) every that many epochs | not refused |
| 0004 latent-clip | `quant.latent_clip` c clips binary latents to [−c, c] after each optimizer step | not refused |
| 0005 beta-mode | `quant.beta_mode`: `absmean` (default), `absmean_pow2`, or `learned_pow2` (a learned log2 scale per layer, initialized from the absmean) | refused |
| 0006 channel-gain-pow2 | `quant.channel_gain`: a power-of-two gain per output channel, 1 at initialization; the binary gate checks each channel separately | refused |
| 0007 weight-scheme-baselines | admits `quant.weight` `none` or `int8_absmax` and `quant.layer_weight_override`, for labelled baselines and teachers; `int8_absmax` is the screen code's static fixed<8,3> grid | refused |
| 0008 init-from-checkpoint | `experiment.init_checkpoint` copies latents from a named checkpoint before calibration; the checkpoint's `input_std.json` must match the cache | not refused |
| 0009 kd-constrained-runner | admits `experiment.distillation` in the constrained runner; the teacher is loaded from an artifact and run on every training batch, augmented or not | not refused |
| 0010 kd-attention-map | `experiment.distillation.attention_coefficient` adds KL(teacher ‖ student) over the softmax maps to the loss | not refused |
| 0011 augment-eta-phi-reflect | `data.augment` flips the signs of η_rel and φ_rel per jet, on training batches only | not refused |
| 0012 latent-ema-eval | `experiment.latent_ema_decay` keeps an EMA of the binary latents; candidate checkpoints are evaluated and saved with the EMA latents | not refused |
| 0013 beta-schedule-runner | `train.ebops.controller: "schedule"` replaces the PID controller with `train.ebops.beta_schedule` (hgq2 `PieceWiseSchedule`) | not refused |
| 0014 screen-collapse-stop | `experiment.collapse_stop` {threshold, patience, after_epoch} stops a collapsed run and writes `COLLAPSED.json` (rule G1 in [rules.md](../../docs/delta/rules.md)) | not refused |
| 0015 head-dims | `arch.head_dims`: hidden widths of the classifier head | refused |
| 0016 pre-block-tanh-lut | `arch.pre_block_act: "tanh_lut"`: a trainable tanh table before attention and before the FFN | refused |
| 0017 ternary-absmean | `quant.weight: "ternary_absmean"`: the labelled ternary baseline, with its own gate, billed at 2 bits per weight | refused |
| 0018 hgq-learnable-weights | `quant.weight: "hgq_learnable"`: HGQ learnable weight widths, a labelled baseline | refused |
| 0019 accumulator-ebops-metric | `experiment.accumulator_metric` logs b_acc = b_act + ⌈log₂ fan_in⌉ for each binary layer beside the native EBOPs (Z03) | not refused |
| 0020 diag-sign-flips | `experiment.diagnostics: ["sign_flips"]` logs the per-layer sign-flip fraction per epoch and the C2I ratio (Z08) | not refused |
| 0021 diag-beta-trajectory | `experiment.diagnostics: ["beta_trajectory"]` logs β per layer and a histogram of \|latent\|/absmean (Z08) | not refused |
| 0022 strict-keys (infrastructure) | the validator refuses, by name, any config key that no code path in the tree reads; a key whose code is not published says so in the message | |
| 0023 body-deepsets | wires `newmods/deepsets.py` into the builder and the binary gate: `arch.body: "deepsets"`, `arch.deepsets_dims` | refused |
| 0024 bop-optimizer | wires `newmods/bop.py` at both optimizer call sites: `train.binary_optimizer: "bop"`, `train.bop_gamma`, `train.bop_tau` | not refused |
| 0025 run-pack-names-roots (infrastructure) | `run_study.py` and `run_pack.py` read their roots from `BNJ_DATA_ROOT`, `BNJ_RUN_ROOT` and `BNJ_CAMPAIGN_DIR`, and pack entries may be run names; unset, they behave as before | |

## New modules

| module | catalogue name | status |
| --- | --- | --- |
| `newmods/deepsets.py` | body-deepsets | binary Deep Sets body; the default layer widths follow the `get_gnn` Deep Sets model in the HGQ2 examples (jsc150), re-implemented without copying code; wired by 0023 |
| `newmods/bop.py` | bop-optimizer | Bop (Helwegen et al., [arXiv:1906.02107](https://arxiv.org/abs/1906.02107)) on the binary latents, Adam on everything else; wired by 0024 |
| `newmods/linformer.py` | attn-linformer | the attention layer with binary sequence projections; **not wired**, because its wiring requires the reference-configuration patches, not published here; the validator refuses `arch.attn_kind` |
| `newmods/diag_attention.py` | diag-attention | post-run script: attention entropy over log n_valid with gated keys masked, and the change in the metric when A·V is replaced by the mean of V |
| `newmods/diag_input_proj_rows.py` | diag-input-proj-rows | post-run script: distinct sign patterns of the binary input projection (Z04) |
| `newmods/diag_latent_binary_gap.py` | diag-latent-binary-gap | post-run script: accuracy with the float latents in place of the binary kernels |

The three diagnostics run on any saved checkpoint; nothing in `run_study.py` calls them.

## Which catalogue entries this code covers

The counts below come from `delta.json` (`code_tier`, `code_changes`, `reference_code_required`),
103 entries in all.

- **Config-only, 11 entries** (`code_tier` T0): M001, M002, M009, M010, M013, M015, M042, M043,
  M044, M046, M058. They change keys the screen runner already reads.
- **Code complete in this series, 34 entries**: M006, M014, M016–M025, M027, M034–M037, M045,
  M047–M050, M061–M064, M082–M084, M088–M090, M094, M101.
- **Requires the reference-configuration patches, not published here, 58 entries.** Twenty-one
  name the reference schedule, optimizer, pT gate or quantizer set in `reference_code_required`.
  The rest use one of 13 code changes that exist only on top of that code: lr-schedule-variants,
  softmax-table-min-bits, qk-stream-min-bits, ebops-group-weight, attn-linformer (the wiring),
  attn-relu-over-n, act-granularity-element, act-init-f0, pre-quant-shift, pt-gate-threshold,
  gated-key-mask, std-real-slots and derived-input-features.

One caveat covers all three groups. Every Delta entry is a change to one of the reference
configurations (d32h4 and d24h2 in [the Delta README](../../docs/delta/README.md#the-reference-configuration)),
and those use a quantizer set, learning-rate schedule and optimizer that are not in this
repository. Applied here, a Delta key acts on the screen code's quantizers and schedule instead.
A run built from this tree alone is therefore not a Delta cell, even for the 45 entries whose own
code is complete. The checks build d32h4-shaped models (the `const0922-a07-n64` screen config,
d_model 32 and 4 heads) and d24h2-shaped ones (d_model 24, 2 heads, no positional encoding) on
the screen quantizers.

## Export to HLS

`convert_binary.py` calls `delta_keys.export_blockers` and stops on any config that sets a key
marked "refused" above: the binarizer, scale and gain changes (0002, 0005, 0006), the
non-binary weight schemes (0007, 0017, 0018), the deeper head (0015), the tanh table (0016) and
the Deep Sets body (0023). `binarize.py` carries the math for 0002, 0005 and 0006, and the
per-patch tests compare its effective weights with the training forward pass for 0005, 0006 and
0015, but `convert_binary.py` has not been run on any of them. The Linformer layer has no
export path at all: hgq2 0.1.9 has no matching layer and `convert_binary.py` knows no sequence
projection.

Keys marked "not refused" change only training, logging or the runner. A model trained with
them has the screen architecture and goes through the unchanged export path. No Delta model has
been converted, synthesized or placed on an FPGA, so this code supports no resource, latency or
DSP claim.

## Checks

All three run on a CPU in the pinned environment of `checks/pyenv.sh`: Python 3.12 through `uv`,
tensorflow 2.21.0, keras 3.15.0, hgq2 0.1.9, quantizers 1.2.2, numpy 2.5.0, one thread,
deterministic ops. Run them from the repository root:

```bash
code/delta/apply.sh /tmp/d0 0000              # screen code + newmods, no patches
code/delta/apply.sh /tmp/d
code/delta/checks/pyenv.sh code/delta/checks/gate_invariance.py --tree /tmp/d0/code --out fp0.json
code/delta/checks/pyenv.sh code/delta/checks/gate_invariance.py --tree /tmp/d/code --out fp.json
python3 code/delta/checks/compare_fp.py fp0.json fp.json
code/delta/checks/pyenv.sh code/delta/checks/slug_tests.py --tree /tmp/d/code --out slug.json
(cd /tmp/d/code && "$OLDPWD"/code/delta/checks/pyenv.sh -m pytest -q tests)
```

**Invariance.** For seven screen configurations (a07-n64, a00-n8, a01-n64, b02-n64, b04-n8,
a08-n64, e03-n8), `gate_invariance.py` records the config digest, the initial kernel hashes, the
model JSON, the initial EBOPs, the predictions on a fixed synthetic sample, and the hashes of
every variable after one epoch of two optimizer steps. The unpatched and patched trees must agree
on every field. On 27 September 2026 all seven agreed (`INVARIANCE_GATE_PASS`).

**Per-patch tests.** `slug_tests.py` has one test per patch, named by the catalogue name, plus a
d24h2-shape sweep and a strict-keys test: 24 in all. Each checks the mechanism against
a hand computation where one exists, builds a config with the key set, applies the binary gate
(or the labelled gate for the non-binary baselines), and reloads a saved model with predictions
within 1e-7. Most also train for one or two optimizer steps or a short run and gate again. The Deep Sets, Bop and strict-keys tests build entries
M004, M006 and M020 from their `config_delta` in `delta.json`; the Deep Sets widths, which the
catalogue leaves to be set before launch, get test values. On 27 September 2026 all 24 passed (`SLUG_TESTS_ALL_PASS`), in about 11 minutes on one thread.

**Module tests.** `tests/` holds 13 tests of the new modules (Bop 4, Deep Sets 3, diagnostics
4, Linformer 2), and patch 0025 adds 6 for the runner. On 27 September 2026 all 19 passed.

## What is not here

The 13 code changes listed above, and the reference-configuration code they sit on, are not in
this repository. Neither are the Delta config generator, the generated configs or the job
manifests: they are built on the reference configurations. Several entries still carry values
marked `<set before launch>` in the catalogue (the Deep Sets widths, the open-loop β schedule,
the activation init f0). `DEFAULT_DIMS` in `deepsets.py` is a proposal for the Deep Sets widths,
not a decision.
