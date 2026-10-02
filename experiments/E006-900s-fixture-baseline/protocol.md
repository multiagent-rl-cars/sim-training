# E006 — 900-second frozen-fixture baseline

Status: preregistered, not started. Date: 2026-09-27. Run only if all four E001
integration gates pass; otherwise retain this as an unexecuted protocol.

## One question

Does 900 seconds of the existing Basic PPO recipe reduce off-track termination
events **without reducing reset-excluded travel**, relative to its initial
policy, on the fixed seed-123 circular-fixture evaluation?

Success means strictly fewer `out_of_bounds` events and at least as much total
`nonreset_distance_total_m` in the final evaluation. Any other complete result
fails this hypothesis. Missing/incomplete evidence is inconclusive. Report mean
return and all termination counts too, but do not change the rule after seeing
results. This single-seed diagnostic is not a statistical generalization test or
a definition of convergence. An unexpected result motivates a separate study.

## Frozen method

- Same 153-file foundation source and 14 generated fixture files as E001; exact
  [source identity](../E001-foundation-integration/record.json), not just base
  commit `b407792bcda16dbc1bb39db2d89e821ce8578b6d` on dirty `astra-cleanup`.
- Unchanged `configs/experiments/line_following_15min.json`: 12 separate Basic
  arenas, obstacles disabled, train/eval seeds 42/123. CombinedExtractor and
  64-by-64 actor/critic heads; PPO rollout 1,024 vector steps (12,288 car samples),
  batch 256, 10 epochs, learning rate 0.0003, gamma 0.99, GAE 0.95, clip 0.2,
  entropy 0.01, value coefficient 1, gradient norm 1, gSDE resampled every 4 steps.
- Stop collection at the callback's 900-second budget, or 10 million samples if
  reached first. Optimization can overrun that callback deadline; the last
  partial rollout may be collected but unoptimized. Report actual times/samples,
  epoch counter, complete rollout count and partial-rollout events separately.
- Same single A6000, Isaac Sim 5.1, pinned Lab and installed package receipt as
  E001. Offline W&B; periodic actual training video; initial/final checkpoints.
- Fresh-process deterministic before/after evaluation: 450 control steps at
  1/30 second (15 simulated seconds), seed 123, matched initial-state tolerance
  1e-6, 225 frames each. No cherry-picked checkpoint or start seed.
- Whole-job bound 2,400 seconds plus 30-second kill grace. Serial repository
  lock; current user allocation ceiling still applies. Back up all artifacts,
  independently inspect them, and delete the instance after useful work.

## Interpretation limits fixed before results

This is **not a reproduction of the prior team's 15-minute convergence claim**.
The final historical code/assets/criterion are not associated reliably: this
run uses repeated generated circles, an uncalibrated car, A6000 rather than the
reported RTX 5070, and 1/30-second rather than reported 0.1-second control.
The recipe's old L4 limitation annotation is stale; actual hardware is A6000.
The foundation includes explicitly recorded behavior changes. The inherited
image-range issue (network pixels peak at 1/255) remains unchanged. There is no
route-direction/lap reward, and reward growth alone cannot pass this hypothesis.

All 12 arenas repeat one fixture; car samples are not 12 held-out tracks. The
evaluation is short and fixed-seed. Angular progress accumulated across resets
is not a completed uninterrupted lap. Videos show simulation, not real cars.

## Evidence to retain

Requested/resolved recipes, source/asset/runtime hashes, command and status,
training metrics/events, timing and samples, initial/final checkpoints, matched
initial states, raw per-step trajectories, training/start/end videos and a
side-by-side comparison. Report the full result (including failures), not just
a representative image. W&B recording, portable backup and online upload are
separate outcomes. No sweep is authorized by this protocol; fix measured
pipeline/behavior problems before a tuning study.

## Observed result

Pending. No result or demonstration is claimed yet.
