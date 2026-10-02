# E001 — Frozen foundation integration

Status: integration passed; backup verified and instance deleted.
Date: 2026-09-27. Related gate: T05. This is a private local research record,
not a shared deliverable.

## Hypothesis and decision rule

**H1:** the frozen `astra-foundation-v2` source and generated fixtures can complete
the declared Basic/Multi software-integration checks, including actual camera
data and parameter updates, without violating their recorded invariants.

The [original protocol](../../outputs/cloud-validation/20260927-run02/protocol.md)
was recorded before simulator execution. Its SHA256 and immutable source/fixture
identities are in [record.json](record.json). This report adds documentation; it
does not replace the protocol or become part of the 153-file frozen runtime.

Passing requires all four gates under the frozen identity. Require finite
state, observations, rewards and parameters; every smoke car travels more than
0.001 m excluding reset transitions, and resets are fewer than cars × control
steps. Require positive PPO epoch count and finite trainable-parameter changes,
not merely different ZIP bytes or BatchNorm buffers. Camera footage must decode
to its expected frame count and include nonblank sampled frames. Rehearsals also
require checkpoint identity, matched seeds/control duration/initial states
(1e-6 absolute and relative tolerance), and positive recorded per-car travel.
Inspect actual footage and policy-input images as well as the saved-evidence
report. Exit code zero or reward growth alone cannot pass this protocol.

Stop at a failed gate. An invariant violation is a failure; missing or interrupted
evidence leaves the relevant claim unestablished. Preserve both. Corrections need
a new source identity and retained original failures, not rewritten thresholds.

## Method, controls and budget

Use one shared RTX A6000 instance, serial gates, offline W&B, pinned Isaac Lab
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`, and Isaac Sim 5.1. The base Git commit
alone does not identify this dirty worktree: the prepared capsule fixes 153 source
files and 14 uncalibrated fixture files. Exact manifests, locks, requested recipes,
hardware receipts and paths are recorded in the machine-readable companion.

| Gate | Fixed conditions and planned sample budget |
| --- | --- |
| Basic smoke | 12 separate arenas; obstacles enabled; seed 42; 32 fixed-action control steps/32 video frames; PPO requests 64 samples and rounds to one 384-sample rollout |
| Basic rehearsal | Obstacles disabled; unchanged Basic smoke recipe; train/eval seeds 42/123; 384 requested samples; 48 evaluation steps/24 frames per policy |
| Multi smoke | Four interacting cars; seed 42; 32 fixed-action steps/32 frames; PPO requests 64 samples and rounds to one 128-sample rollout |
| Multi rehearsal | Unchanged Multi smoke recipe; train/eval seeds 42/123; 256 requested samples; 64 evaluation steps/32 frames per policy |

Shared-world car samples are not independent arenas. Smoke and rehearsal differ
in policy/settings and are not a learning-quality comparison. Each rehearsal has
a 60-second training callback budget; actual collected/optimized samples and time
must be measured. Whole-job wrappers default to 900 seconds plus 30-second kill
grace. Setup, startup, evaluation, copying and billing are separate quantities.

The user-authorized **ten-hour allocation limit is a ceiling, not a target**:
allocation requested at 20:14:09 UTC on September 27; latest deletion deadline
06:14:09 UTC on September 28. One instance maximum. Stage work efficiently and
delete earlier after verified backup. No independently enforced provider billing
limit exists; process timeouts do not stop charges. A later baseline requires its
own hypothesis/report and does not extend this integration claim.

## Observed results and demonstrations

All four jobs exited zero. Their saved-evidence checks were independently rerun
locally: 7/7 for each smoke and 11/11 for each rehearsal. Finite policy tensors
changed in every gate; matched-state and checkpoint checks passed in both
rehearsals. Per-car measurements and checkpoint hashes are in `record.json` and
the linked raw summaries, not inferred from ZIP-byte differences.

| Gate | Collected samples / PPO epochs | Whole job / training-only seconds | Motion or evaluation evidence | Raw result and inspected clip |
| --- | --- | --- | --- | --- |
| Basic smoke | 384 / 1 | 177 / not separately recorded | All 12 cars: 0.095121–0.464708 m; zero resets; 32 frames | [summary](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-01/smoke/result.json), [checks](../../outputs/cloud-validation/20260927-run02/basic-01-evidence.json), [fixed-action video](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-01/smoke/smoke.mp4) |
| Basic rehearsal | 384 / 2 | 96 / 3.562604327 | Matched 48-step/1.6-second evaluations; zero resets; 24 frames each | [summary](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-rehearsal-01/experiment/training_summary.json), [checks](../../outputs/cloud-validation/20260927-run02/basic-rehearsal-01-evidence.json), [initial/final video](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-rehearsal-01/experiment/start-vs-end.mp4) |
| Multi smoke | 128 / 1 | 29 / not separately recorded | All four cars: 0.251054–0.553059 m; zero resets; 32 frames | [summary](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/multi-01/smoke/result.json), [checks](../../outputs/cloud-validation/20260927-run02/multi-01-evidence.json), [fixed-action video](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/multi-01/smoke/smoke.mp4) |
| Multi rehearsal | 256 / 2 | 94 / 5.169705610 | Matched 64-step/2.133-second evaluations; one reset each; 32 frames each | [summary](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/multi-rehearsal-01/experiment/training_summary.json), [checks](../../outputs/cloud-validation/20260927-run02/multi-rehearsal-01-evidence.json), [initial/final video](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/multi-rehearsal-01/experiment/start-vs-end.mp4) |

Whole-job time includes startup/evaluation, not allocation or installation. Basic
smoke includes the first cold shader startup. Smoke wrappers used 600 seconds;
rehearsals used 900 seconds. Every inspected video had eight nonblank sampled
frames. Saved [camera](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-01/smoke/camera.png)
and [policy-input](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/basic-01/smoke/policy_inputs.png)
images and representative actual-video frames were visually reviewed. The linked
clips contain the complete short checks, not cherry-picked driving successes.
They are **generated-fixture simulation**, never real-hardware footage. Smoke
uses fixed actions; rehearsal uses initial/final policies.

Counterexamples matter: Multi training had two out-of-bounds terminations, and
each 2.133-second Multi evaluation reset once. Initial/final behavior appeared
similar in the reviewed comparison frames; this is not improvement evidence.
Both smoke reports confirm inherited image inputs of 0–1/255, not that this
scaling is desirable. Recorded distance does not prove route following.

Multi also retains a confirmed ego-observation defect: its
[policy-input image](../../outputs/cloud-validation/20260927-run02/remote/outputs/validation/multi-01/smoke/policy_inputs.png)
shows the observing car centered in all four obstacle-history channels. The
[shared-map code](../../source/car_training/car_training/tasks/manager_based/car_training/mdp/observation.py)
paints every car without ego exclusion; the
[reward gate](../../source/car_training/car_training/tasks/manager_based/car_training/mdp/rewards.py)
tests for any pixel below 0.12 across those channels and can suppress line and
steering rewards. The saved final-step line terms are zero for all four cars,
with nonzero velocity/alive terms: **consistent with ego gating**, not a same-step
causal attribution of every zero. Identical initial history channels do not prove
correct temporal history, and PNG contrast is not the post-wrapper 0–1/255 input
scale. The Basic obstacle-disabled router does not use this Multi ego gate.

Both rehearsal W&B identities are recorded locally and offline; no online upload
is claimed. All 240 session files (200,059,483 bytes) match the remote checksum
inventory, with independent exact-coverage review. Deletion was observed in Brev
at 21:07:53 UTC; see the [session record](../../outputs/cloud-validation/20260927-run02/session.json)
and [empty inventory receipt](../../outputs/cloud-validation/20260927-run02/deletion-inventory.json).

## Interpretation and closure

Decision: H1 is supported for these four frozen integration gates. This protocol
cannot establish useful/safe driving, 15-minute
convergence, calibrated physics or sim-to-real transfer. Recorded distance is not
route progress, and state-tensor changes do not reconstruct optimizer execution.

The next learning-quality test must use a separate protocol, such as
[E006](../E006-900s-fixture-baseline/README.md); passing E001 is not its outcome.
Reopen this decision if integrity review finds missing/mismatched evidence or an
unmet original invariant. No runtime correction was applied during these gates.

Preserve logs, statuses, runtime/source/fixture identities, checkpoints, metrics,
images/videos and the whole offline W&B job, including failed attempts. Verify
the copied bytes against remote checksums independently; inspection failure must
not prevent backup. Record deviations, then delete the exact instance and verify
its absence. Backup, deletion and online W&B synchronization remain separate
claims; no upload or billing audit is established here.
