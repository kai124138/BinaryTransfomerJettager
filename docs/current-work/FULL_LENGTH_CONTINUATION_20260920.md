# Full-length continuation — 20 September 2026

**September 21 update:** [current continuation results](TRAINING_PROGRESS_20260921.md). The observations and original protocol below retain their stated dates.

All **27 current public training runs** have been submitted to complete their existing **1,000-epoch schedules**. This supersedes selective promotion for the architecture and attention screens. The original seven EBOP ablations are already complete and are not rerun.

Execution snapshot: **2026-09-20T22:08:06.102882+00:00**. [Machine-readable status](continuation-20260920.json) · [Recovered screening metrics](TRAINING_PROGRESS_20260920.md) · [Checkpoint preflight evidence](checkpoint-screen-status-20260920.json).

| Campaign | Runs | Resume checkpoint | Target | Pod phases |
|---|---:|---:|---:|---|
| batch20260917 | 12 | 100 | 1,000 | 11 Running, 1 Pending |
| batch20260918 | 15 | 400 | 1,000 | 15 Pending |

Pending runs are submitted, not missing. The scheduler starts them as compatible GPUs, CPU and memory become available. A Running container may still be installing dependencies; per-index resume and completed-epoch observations are recorded separately in the JSON.

## Checkpoint and schedule continuity

The CPU inspection verified every saved screening checkpoint against its submitted code and canonical configuration hashes. All configurations already specify 1,000 epochs, and none has a full-training COMPLETE marker. Each trainer retains its existing output root, optimizer/controller state and data identity. No model source, training configuration, learning-rate schedule or TF32 setting was changed.

The screening wrapper accepts only 100/200/400-epoch stops. These continuations call the same frozen trainer directly and omit the screening-stop argument, allowing the configured 1,000 epochs and normal completion/artifact path to execute. Three-seed attention comparisons retain seeds 4,5,6.

Each public campaign can run all its indexes concurrently (12 and 15), subject to scheduler capacity. Indexed retries preserve checkpoints, with six retries per index, a 14-day whole-job deadline and a 71-hour process timeout. Jobs run independently of the laptop. These limits do not guarantee a completion time.

## Metrics and remaining interpretation

All six previously unavailable attention summaries were recovered from W&B. All 15 attention last checkpoints are over 350k EBOPs, with 27.6992%–42.0290% internal-validation accuracy. Preflight also confirms that all 27 durable screening states record no feasible checkpoint at their own final targets. Full training is being continued under the existing protocol; no accuracy or budget improvement is assumed.

The earlier 100/400-epoch snapshots remain linked as historical measurements. New resumed measurements retain their source timestamps. Final selected-checkpoint evaluation and hardware validation are separate from a job reaching 1,000 epochs.

[Return to current work](README.md).
