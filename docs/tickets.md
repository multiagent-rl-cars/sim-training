# Local execution tickets

Updated 2026-09-27. Personal repo backlog; not published to Linear or team spaces.
“Local complete” never means GPU, driving or physical hardware acceptance.
See [work plan](work-plan.md) for sequencing and [refactor](refactor.md) for scope.

| ID | Priority / state | Work and definition of done | Dependencies |
| --- | --- | --- | --- |
| T01 | P0 / foundation validated | Portable package, strict recipes, ty/Ruff/CPU gate and independent reviews; current YAML/tracking/cleanup tree passes 681 local tests (frozen image-only candidate passed 453); frozen reference passes four GPU integration modes in E001. Type-check exclusions for Isaac adapters remain explicit | Existing foundation; E001 |
| T02 | P0 / live reference path passed | E001 setup/four modes and E006 baseline completed; 240 files independently checksum/coverage verified and exact instance deletion confirmed. Synthetic timeout tests are not a provider billing safeguard. New source snapshots still need their own gates | T01; E001/E006 |
| T03 | P0 / image fix local pass; other defects open | E003 image-only correction has a 453-test capsule and independent review; upload approval/GPU gates pending. Audit/fix history advancement/reset isolation and ego masking separately; no bundled reward redesign | T01; E003 |
| T04 | P0 / open | Audit terminations/truncation, per-agent/world reset ownership, time units and synchronized trace; prove one-step observation/action/reward/termination alignment | T03 |
| T05 | P0 / inherited integration closed; candidate pending | [E001](../experiments/E001-foundation-integration/README.md): all four modes passed, backup independently verified, instance deleted. E003 source is not GPU-validated; known history/ego defects are not accepted as correct behavior | T02; T03–04 for corrected-behavior acceptance |
| T06 | P0 / partial audit | Associate final historical code/config/assets/checkpoints and convergence criterion with report claims; validate recovered older USDs without overwriting fixtures; publish explicit differences or “not comparable” | Source recovery, T05 for asset runtime inspection |
| T07 | P1 / open | Freeze held-out tracks/starts/seeds; implement lap/route/collision/off-track/oscillation metrics, invariant tests and aligned video/input/reward overlays; approve criteria before results | T04, T06 |
| T08 | P1 / fixture diagnostic complete; reproduction pending | E006 unchanged 900-second fixture baseline improved its fixed 15-second evaluation; raw videos/trajectories/time/sample accounting preserved. Historical identity/criterion, full-lap and held-out/paired-seed evidence remain missing | T05–07 |
| T09 | P1 / online recovery verified; sweeps pending | Synthetic online scalar/video/artifact gate passed. All three September 27 runs synced from verified copies; original histories and 10 training clips checked against server metadata, checkpoints/evaluation artifacts repaired and exact manifests verified. Live tracking is selected in global YAML; new launcher remains GPU-unvalidated. Rotate the previously exposed credential. See outputs/wandb-upload-20260927/summary.json | T02; sweep integration remains unimplemented |
| T10 | P1 / post-refactor | Preregister single-hyperparameter pilot and paired confirmation; fixed sample budgets, held-out metrics, failures included, total compute budget and serial execution | T07–09 |
| T11 | P1 / canonical workflow implemented; legacy audit open | One config.yaml and no-argument workflow now own supported configuration. Obsolete supplied-media tracking/alternate job wrapper and Flake8 config retired; 19 root debug/action-log files recoverably archived. Internal Isaac argument adapters and checkpoint-compatibility trainers remain deliberate dependencies; further retirement requires inventory and tests | T03–05 |
| T12 | P1 / open | Independent expiry/billing safeguards: test control-plane cleanup after client loss using the smallest approved resource; document actual guarantees and failure modes, no untested unattended launch | Future explicit cloud authority |
| T13 | P2 / later | Supported Isaac runtime migration in isolation, pinned dependency review and matched integration/physics checks; retain historical environment | T05–08 |
| T14 | P2 / later | Select one extension from work-plan table; calibrate real control/latency and define safety acceptance before transfer/real training | T07–08, physical supervision |

## Acceptance discipline

- A reviewer other than the author checks substantive changes and their tests.
  Record concrete findings/fixes, not “best practices” as an unsupported label.
- Closing a ticket requires the artifact/log/test path and its source identity.
- No new framework/platform, hyperparameter sweep or reward objective is a
  prerequisite to cleaning the repository. Reopen that choice when a measured
  research requirement justifies it.
- Failed GPU runs are data worth preserving, not permission to keep trying
  indefinitely. Stop, back up, delete and diagnose locally when possible.
