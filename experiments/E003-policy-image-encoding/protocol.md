# E003 — Policy image amplitude, one isolated correction

Preregistered 2026-09-27, before candidate GPU execution. Candidate source identity
must be frozen after independent review and the full local gate. The running
E006 reference uses the old source and must finish unchanged before deployment.

## Hypothesis

Scaling only policy image terms by 255 makes actual Basic and Multi CNN inputs
span exactly 0..1 (within float32 rounding), while preserving internal binary
maps, shapes, velocity observations, reward helpers and fixed-action motion.
This is an observation-contract test, not a learning-quality hypothesis.

## Change and exclusions

Encode policy images as float32 0/255 at the observation-manager boundary;
retain its 0..255 bounds, the pinned SB3 wrapper's uint8 conversion and SB3's
single /255 normalization. Keep raw helper/history/reward maps at float32 0/1.
Correct diagnostic PNG encoding, record the new observation/behavior contract,
and fail closed when supported checkpoint loading encounters an old/mismatched
encoding. Do not silently convert old checkpoints or resume E006 under new code.

Do not change ego masking, history advancement, reward weights/objective, reset
semantics, assets, simulator timing, velocity/action mapping, architectures or
PPO hyperparameters. These remain separate tickets, including known defects.

## Gates and fixed controls

1. Independent review and local tests: patterned black/white input through both
   real SB3 extractors/preprocessing; exactly-once normalization; no mutation of
   helper/history inputs; correct PNG pixels; old/mismatched checkpoint rejection;
   unchanged nonimage dimensions. Ruff, formatting, ty and the full CPU suite.
2. New source capsule with verified bytes. Reuse only the same live A6000/runtime
   and the same verified 14 fixture files. No second instance. Preserve all E001
   and E006 source identities and outputs.
3. Serial Basic smoke, Basic rehearsal, Multi smoke, Multi rehearsal using the
   unchanged smoke recipes, seeds and existing bounded wrappers. Fresh output
   paths; offline W&B; actual images/videos; matched reload evaluations.
4. Require every existing E001 integration criterion, plus actual image network
   minima 0 and maxima 1 (absolute tolerance 1e-7) for both image keys in both
   smoke tests. Compare reset-excluded fixed-action per-car travel and final
   weighted reward terms against E001 with absolute/relative tolerances 1e-4.
   This checks these measured invariants, not full trajectory equivalence.
   The comparison must be reported even if it fails; do not relax thresholds.

Stop at a failed invariant; retain evidence and diagnose before another attempt.
The maximum existing session allocation deadline remains 2026-09-28 06:14:09 UTC,
not a target or provider-enforced billing cap. Job process limits do not delete
the GPU. Back up and verify all artifacts, then delete when useful work ends.

## Decision and limits

Accept the encoding correction only if local and GPU gates pass under the new
identity. Otherwise mark failed or incomplete and retain the frozen reference.
No fresh 15-minute training or sweep is part of this hypothesis. A later learning
comparison needs its own protocol and paired sample/time budgets. An amplitude
fix does not establish better driving, fix ego/history leakage, recover original
assets, prove convergence or validate sim-to-real transfer.
