# Frozen Engram experiment source

This directory contains the runtime source used by the E00–E03 pilot and full-length continuation. All 22 files listed in [source_manifest.json](../../results/engram/source_manifest.json) match their recorded SHA-256 hashes. The manifest also records the Linux production dependency versions. The hash includes those versions, so a different environment cannot resume the same run identity.

[Study, results and limitations](../../docs/current-work/ENGRAM_STUDY.md) · [Final status](../../results/engram/status-20260923.json) · [Earlier failure evidence](../../results/engram/status-20260921.json) · [Supplementary file hashes](../../results/engram/supplementary_files.json)

This is a frozen experiment package. Invoke its runner directly as below so Python imports this directory's `bnhgq2`; the root editable installation exposes the older `code/hgq2` package. Do not launch memory configurations with the generic ablation runner. E00 finalization succeeded; E01–E03 fail metric-reproduction checks, which the code preserves for investigation. HLS conversion of the memory module is unsupported.

## Environment and synthetic checks

From the repository root, use a dedicated Python 3.12 environment. The CPU check exercises synthetic inputs only; it does not validate final jet accuracy.

```bash
python3.12 -m venv .venv-engram
source .venv-engram/bin/activate
python -m pip install -r code/engram/requirements-cpu.txt
KERAS_BACKEND=tensorflow WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=-1 \
  python code/engram/run_engram.py preflight --out /tmp/engram-preflight
```

For Linux GPU training, install `code/engram/requirements-training.txt` instead. The recorded production versions include TensorFlow 2.21.0, Keras 3.15.0, HGQ2 0.1.9, quantizers 1.2.2, NumPy 2.5.0 and scikit-learn 1.9.0. Existing checkpoint resume requires the exact source, configuration, data and runtime-version identities.

## Prepare the shared data cache

Obtain the training archive described in the [main README](../../README.md#dataset-and-metrics), and place the 62 extracted training HDF5 files in `data/train`. This operation loads the real training data and belongs on a suitable compute host. It creates the standard 496,000/124,000 training/internal-validation split and records hashes of all four arrays. No held-out test data is used.

```bash
KERAS_BACKEND=tensorflow WANDB_MODE=disabled \
  BNHGQ2_CODE_SHA256=918e57c18ecf9a1094531f83f75ae67ca5a346162747e850e6e65f567f0cd366 \
  python code/engram/prepare_batch_cache.py \
  --configs code/hgq2/configs/batch20260917 \
  --raw data/train --root outputs/engram-cache --n-parts 16
```

The helper is an accompanying cache-preparation utility, outside the 22-file trainer manifest. It matches the shared-cache workflow used by the experiment. Existing cache identities must pass the runner's checks; do not edit metadata to bypass them.

## Run or resume an experiment

On a GPU host, the following trains E02 through its configured 1,000 epochs, resuming a matching existing output directory when present:

```bash
KERAS_BACKEND=tensorflow TF_FORCE_GPU_ALLOW_GROWTH=true WANDB_MODE=disabled \
  python code/engram/run_engram.py train \
  --config code/engram/configs/engram/engram-e02-s1.json \
  --data-cache outputs/engram-cache/n16/data --out outputs/engram/engram-e02-s1
```

For the historical 100-epoch screening stop, append `--stop-after 100`; the learning-rate schedule remains 1,000 epochs. E00/E01 are controls without memory, E02 is ungated memory, and E03 is gated memory. E04–E07 were not run in this original N16 continuation; separately adapted versions appear in the [N8/N64 exploratory screen](../../docs/current-work/CONSTITUENT_SCREEN_20260923.md). Changes to seeds, configuration, source or dependencies need a separate run/output identity.

Tracking is optional and omitted above. The frozen runner's `--track` mode requires its configured separate private W&B destination and verifies it. Public GitHub publication does not change that project's access. Reproduction without W&B uses the same numerical training code.

## Validation and artifact caveat

[validate_engram_publication.py](../analysis/validate_engram_publication.py) checks source/configuration hashes, snapshot invariants and artifact scope without TensorFlow. `check_engram.py` additionally tests addressing, masking, initialization, gradients, optimizer behavior, serialization and interrupted/resumed synthetic training.

`COMPLETE.json` is written by the inner trainer before the outer selected-checkpoint reload assertion. Require a successful outer validation and an `engram_result.json` matching the current completed epoch before using final predictions or memory tables. An older result file can survive a failed continuation; never select it solely because it exists.

Publication verification: [all 11 synthetic check groups passed](../../results/engram/publication-checks.json) on macOS CPU. Runtime source bytes match production; local NumPy was 2.4.6 versus production 2.5.0. This software check does not resolve the real-data finalization failures.
