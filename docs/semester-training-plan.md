# Semester training foundation

> Historical planning snapshot: September 22, 2026. Recommendations remain
> useful research questions, but implementation/tracking status below is dated.
> Use the [current workflow](../README.md), [architecture](architecture.md),
> [operations runbook](operations.md) and [experiment registry](../experiments/README.md)
> for current evidence. This notice was added on September 27; the original
> proposal below is preserved, not silently promoted to a completed feature list.

## Recommendation

Use W&B for team experiment comparison, and keep structured local logs and
self-contained run artifacts as the durable record. These solve different problems.
Do the environment correctness work before hyperparameter sweeps or longer runs.
A dashboard cannot make a broken reward meaningful.

The current bounded runner already writes a resolved environment config, source
hashes and Git identity, asset hashes, lockfile, hardware snapshot, JSONL events,
SB3 CSV/TensorBoard metrics, checkpoints, evaluation trajectories, and videos.
It does not upload anything to W&B. Its test fixture is uncalibrated, and the
inherited observation/reward defects are explicitly recorded in the manifest.

## Priority order

| Priority | Work | Acceptance criterion |
| --- | --- | --- |
| P0 | Remove ego from other-car masks | A lone car sees empty other-car input and gets nonzero line reward on the line. |
| P0 | Make observation functions pure | Calling observations twice without stepping produces identical values and does not advance history. |
| P0 | Correct image normalization | An assertion at the actual CNN input verifies the intended range, dtype, channel order and shape. |
| P0 | Reset and timeout semantics | Reset clears wheel/steering state and history, every spawn satisfies clearance, time limits truncate rather than terminate. |
| P0 | Define the driving objective | Metrics and rewards distinguish route progress, stopping, yielding, passing, collisions and driving off-track. |
| P1 | Add deterministic evaluation suite | Start/end use identical initial states; fixed validation seeds and maps are separate from training; final test set remains untouched. |
| P1 | Consolidate launch/config paths | One train entry point, one eval entry point, one validated config schema; no task-specific launcher copies. |
| P1 | W&B + JSONL/TensorBoard | Every plotted metric has a run ID, explicit step/unit, config, source and asset version. Offline training survives network failure. |
| P1 | Checkpoint continuation | Save/reload tests preserve actions for a fixed observation; resumed runs retain optimizer/schedule/normalizer/RNG semantics or clearly declare a restart. |
| P2 | Baselines and seeds | Compare random, straight-driving and simple geometric controllers with PPO; use at least three training seeds for an initial comparison. |
| P2 | Realistic assets and system identification | Match geometry, wheel/steering response, timing, vision errors and battery-dependent performance to physical measurements. |
| P2 | GPU profiles | Same semantics on A6000 and both free desktops, with measured memory/throughput and car count separated from independent arena count. |

## Logging architecture

Use ordinary Python logging plus a JSON formatter or structlog. There is no need
to adopt a distributed logging service for this repository yet. Keep human-readable
console logs, structured lifecycle/error logs, and scalar learning metrics separate.

Suggested JSONL envelope:

```json
{
  "schema_version": 1,
  "run_id": "2026-09-22-baseline-seed42",
  "timestamp_utc": "2026-09-22T00:00:00Z",
  "phase": "train",
  "event": "rollout",
  "agent_timesteps": 4096,
  "control_steps": 1024,
  "sim_seconds_per_arena": 34.1333,
  "wall_seconds": 55.0,
  "metrics": {"off_track_terminations": 12}
}
```

Never mix agent-transitions, vector control steps, simulated seconds, and wall
seconds on one unlabeled axis. In this branch four cars share one arena, so
`num_envs` is not independent environment count. Version the schema and include
units in metric names. Emit events for start, rollout, evaluation, checkpoint,
resume, completion, and exceptions. Log scalar summaries every rollout and
hardware health every few seconds; avoid serializing image tensors every step.
Keep full trajectories for selected evaluations, not every training episode.

Metric groups worth standardizing:

- **Task:** route progress/laps, completion, collisions per vehicle-km,
  off-track events, interventions, closest approach, speed, steering saturation,
  episode duration, and each reward component. Report exposure and censoring;
  a 15-second unfinished episode is not a completed successful lap.
- **PPO:** approximate KL, clip fraction, entropy, explained variance,
  value/policy loss, gradient norm, learning rate, action distribution and saturation.
- **Data:** observation ranges, NaN/Inf counts, mask occupancy, ego leakage,
  history timestamps, pose/velocity noise and age.
- **System:** simulation/rollout/update wall time, aggregate agent steps/s,
  real-time factor, GPU memory/utilization, host RAM, disk use and artifact size.

The inherited residual network uses BatchNorm. Inspect train/eval behavior and
large KL changes before accepting results; compare GroupNorm/LayerNorm or a simpler
small CNN as controlled ablations. Add a PPO KL early-stop guard once the baseline
is recorded. Do not change architecture, reward, history and dynamics together,
or it becomes impossible to identify which change helped.

## W&B specifically

I recommend one team project, with groups for experiment families and tags for
reward/observation version, asset set, seed, and GPU profile. Record the resolved
config at run creation. Name checkpoints as artifacts with lineage to the exact
asset and source version; attach evaluation tables and videos to the run. Use
`latest`, `best-validation`, and `final` aliases with explicit meanings. Select
best checkpoints using held-out driving metrics, not training reward.

The official SB3 integration supports TensorBoard metric synchronization. Prefer
reusing the existing SB3 logger over maintaining two competing metric pipelines.
W&B also supports offline logging followed by later synchronization. Keep local
run outputs even when using the hosted dashboard. Set the team/project and access
settings deliberately before uploading footage, assets, or source snapshots.

Sources checked for this recommendation:

- [SB3 integrations](https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html)
- [W&B artifact versioning and lineage](https://docs.wandb.ai/models/artifacts)
- [W&B offline operation](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues)
- [SB3 2.7 evaluation guidance](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/rl_tips.html)

## Reproducibility and artifacts

Every run should own an immutable directory:

```text
runs/<run-id>/
  manifest.json           # seed, code/asset hashes, versions, GPU, known limitations
  config.json            # requested configuration
  resolved_env.yaml      # actual instantiated environment
  source/                # archive or commit plus complete patch/untracked source
  events.jsonl           # lifecycle and structured diagnostics
  metrics/               # CSV and TensorBoard
  checkpoints/           # periodic, interrupted, best-validation, final
  evaluation/            # seeds, exact initial states, metrics, trajectories, videos
```

Hashing dirty source identifies it but does not preserve its contents. Preserve a
source archive too. Keep USDs, cached maps, camera calibration, textures and asset
manifests versioned together; a cached image must never silently describe a
different track. Regenerate synthetic fixtures from their committed generator.
Store large originals/checkpoints outside Git, through W&B Artifacts or one agreed
team storage location, and retain explicit hashes and retrieval instructions.

A PPO zip contains useful model/optimizer state, but loading it alone is not proof
of exact continuation. Specify reset versus continuous rollout on resume, preserve
schedule progress, RNG state and any normalizers, and test interruption/restart.
Independent Isaac process starts should be the default for evaluation, since this
repo currently has mutable observation history and reset state.

## Practical first two weeks

1. **First few days:** fix the three known observation/reward defects; add small
   CPU regression tests for transforms/history/masks and one short GPU integration
   test. Enable explicit timeout and audit resets. Keep the recorded inherited
   baseline as a reference, with its known defects labeled.
2. **Remainder of week one:** establish one corrected single-car task and one
   small multi-car task, matched evaluation seeds and videos, local structured logs,
   checkpoint reload tests, and the team's W&B project.
3. **Week two:** run a small seed matrix on the free desktops, compare against a
   geometric controller, then increase traffic density and map diversity. Add
   measured latency/noise/dynamics variation only after the basic task is sound.

CPU CI should validate config schemas, transforms, mask ownership, observation
purity, action scaling and asset manifests. A GPU job should check camera frames,
finite stepping/rewards, one policy update, save/reload, and a short evaluation.
No need to run a full Isaac installation on every Mac or every documentation PR.
Keep semester experiment hypotheses and success criteria in the repository; link
results into the team's course logbook rather than relying on screenshots alone.
