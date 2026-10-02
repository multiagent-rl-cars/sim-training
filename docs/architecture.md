# Architecture and executable contracts

Updated 2026-09-27. This describes the working branch, not a completed eight-month
platform. The frozen `astra-foundation-v2` snapshot passed the GPU integration
gates and one fixture behavior test. The newer image correction and tracking
changes have separate evidence boundaries; the whole current branch has not
been validated on the GPU. Start with the [current workflow](../README.md).

## Dependency direction

```text
CLI / tests / analysis          inherited Isaac adapters
          \                         /
           rc_car: contracts, kernels, models, telemetry

rc_car.experiments.config -> Pydantic (no Torch, Isaac, or W&B import)
rc_car.core / models / telemetry -> optional learning dependencies
rc_car.experiments.tracking -> W&B only on an explicit tracking call
```

`import rc_car` has no registration or simulator side effects. The legacy
`car_training` extension still eagerly imports Isaac tasks/UI; consumers must
use `rc_car` for CPU work. Do not move simulator imports into the portable tree.

The root `rc-car-research` distribution packages both `src/rc_car` and the
inherited `car_training` package without importing the simulator at build time.
There is one `pyproject.toml`, one `uv.lock` and one `.venv` per checkout. Default
CPU dependencies use PyTorch's CPU index on Linux; the explicit `isaac` extra
selects CUDA 12.8 and NVIDIA's complete `isaaclab==2.3.0` bundle. CPU and Isaac
selections conflict intentionally. Use `uv run rc-car` locally, or
`uv run --locked --no-group cpu --extra isaac rc-car` on a prepared Linux RTX host.
Selecting CPU again can replace CUDA Torch; never change dependencies during a
job. The bundled packaging path still requires a fresh GPU integration gate.

## Numerical contracts actually extracted

| Function | Meaning retained |
| --- | --- |
| `convert_coordinates` | Centered world metres ↔ top-left image pixels; y flips; width/height boundaries, not final pixel centres; optional aspect-preserving padding |
| `world_velocity_to_image` | `(vx, -vy)` remains m/s, not pixels/s; output does not alias input |
| `heading_to_image_vector` | `(cos(yaw), -sin(yaw))` |
| `legacy_aligned_velocity` | Historical `R(-yaw - pi/2)` convention, optionally divided by 0.8 m/s before rotation; no new forward/lateral convention |
| `legacy_action_targets` | `tanh` then wheel-speed/steering scaling; zero action is not a stop command |
| `legacy_line_proximity` | `1 - distance_px/40`, gated near perceived obstacles; contour distance, not established centerline error |
| `legacy_steering_penalty` | Negative steering magnitude divided by steering limit; not steering rate or jerk |

Task managers still own reward weights and dt multiplication. Kernels must not
apply either implicitly. Values beyond the contour scale remain negative. Cheap
shape checks run on calls; opt-in finite-value checks avoid synchronizing CUDA
on every step in a previously validated loop.

For the inherited driving limits 10..30 wheel rad/s and ±0.6 steering rad, the
adapter applies `tanh` to the raw action. The manager environment's action space
is unbounded; the pinned Isaac Lab SB3 wrapper clips it to **[-100, 100]**, not
[-1, 1]. An action of ±1 therefore reaches about 12.384..27.616 rad/s and
±0.456956 rad, but larger raw actions approach the physical limits. Raw zero
commands 20 rad/s forward motion. This is not a calibrated real-car mapping.

## Models and checkpoint boundaries

The extracted residual CNN + velocity encoder preserves layer names,
state-dict layout, RNG initialization order, and historical BatchNorm behavior.
The constructor's dummy forward changes BatchNorm statistics once; tests record
that fact. It is not silently corrected during an architecture move.

Explicit input validation checks channel-first float images and velocity shape,
dtype, finite values, and normalized image range. The model forward does not
insert new normalization. Moving the network alone did not fix image scaling;
the separate correction below changes the policy observation terms. See
`models/inputs.py` and `observation_contract.py` for the accepted contracts.

New model objects use portable `rc_car.models` module paths. The old extractor
module re-exports the same classes for Isaac callers. CPU tests isolate the old
module namespace to verify serialized-class compatibility; this does **not**
prove that arbitrary old checkpoints with eager Isaac import paths can load on
a Mac unmodified. Keep their original environment and provenance for migration.
Only load trusted checkpoints: SB3 archives can contain pickled Python objects.

The versioned experiment evaluator requires a matching image-contract marker and
SB3 image normalization. Inherited car-training launchers reject checkpoint loads
that bypass that contract, including old automatic pretrained-policy selection.
This is an intentional compatibility restriction: use the matching frozen
environment for historical weights, or perform an explicit, reviewed migration.
An arbitrary external `PPO.load` call is outside these guards.

Tests include exact fixed-seed old/new state, train/eval outputs and gradients,
state-dict and SB3 checkpoint round trips, and a real PPO update on a tiny
synthetic environment. These establish model/plumbing behavior, not driving.

## Configuration and tracking

`load_config` / `validate_config` return a strict, frozen recipe. `resolve_config`
returns JSON-compatible effective defaults, omitting absent optional budgets and
W&B settings so legacy `dict.get` behavior remains meaningful. Unknown fields,
boolean numeric impostors, nonfinite values, invalid seeds/budgets, and batches
larger than the rollout fail locally. Partial PPO minibatches remain permitted.

The training runner writes `requested_config.json` and resolved `config.json`.
`resolved_env.yaml` captures Isaac scene details. These are distinct records:
recipe defaults are not a complete simulator configuration.

W&B starts only on an explicit call. Recipe mode may be overridden by
`WANDB_MODE`; the effective mode is recorded. Fresh run IDs and no-resume rules
prevent accidental cross-experiment reuse. Public run identity is published
atomically without overwriting an existing identity. SDK setup/storage failures
attempt to close the new run. Credentials never belong in the recipe schema.

Ordinary tests use fake SDKs. The internal `rc_car.ops.offline_check` gate
uses the real pinned W&B SDK with synthetic-only data, isolated SDK directories,
offline mode and a loopback service destination. It verified persisted scalars,
a decodable synthetic video, artifact bytes and successful exit on 2026-09-26.
This is local persistence evidence, not online upload or training evidence. Its
private offline-record reader is SDK-version-pinned. Separate saved-stream
verification compares original scalar history, final summary, training-video
metadata and finished server state; artifact verification compares exact server
manifest entries. Neither a successful sync command nor a `.synced` marker
establishes that all artifacts arrived.

Online experiment launch checks authentication and project access before Kit
startup. During an online run, the training host sends metrics and selected video
clips directly to W&B. The completed-output uploader resumes that exact run and
attaches evaluation media and checkpoint/trajectory artifacts from a verified
copy. It does not rewrite the original backup. Offline mode remains explicit;
authentication failure must not silently disable live monitoring. See the
[test boundaries](testing-and-brev.md) and [operations runbook](operations.md).
The three September 27 saved training runs have been uploaded and server-verified;
the [recovery receipt](../outputs/wandb-upload-20260927/summary.json) records the
individual history/media/artifact checks. This does not prove the newer launcher
has run online on a GPU host.

## Basic wiring and source identity follow-up

`astra-foundation-v2` adds deliberate Basic-task corrections: steering accepts
its configured obstacle list; both gated reward terms use static-obstacle data
(or no gate when disabled), not shared-agent history; track crops and off-track
checks use each arena's local coordinates/dimensions. Multi defaults and reward
formulas/weights/thresholds remain unchanged. CPU tests execute actual adapter
bodies and inspect config bindings. The frozen v2 source then passed the four
[E001 GPU integration gates](../experiments/E001-foundation-integration/README.md).
Those checks do not establish complete scheduling or reset correctness.

`astra-foundation-v3-image-amplitude` is a separate correction:

- Inherited path: raw binary **0/1** → uint8 **0/1** → SB3 divides by 255 →
  CNN values **0 or 1/255**.
- Candidate path: policy terms scale to **0/255** → uint8 → SB3 divides by 255 →
  CNN values **0 or 1**.
- Raw image helpers, history buffers and reward inputs remain **0/1**. No reward,
  action, velocity, architecture or PPO hyperparameter change is bundled with it.

The frozen candidate passed 453 local tests, Ruff and ty; its four preregistered
GPU gates are still pending. Later tracking/workflow edits are not part of that
168-file candidate snapshot. See [E003](../experiments/E003-policy-image-encoding/README.md).

Before constructing AppLauncher, the runner checks manifest-listed source bytes,
rejects symlink/escaping paths and checks the installed Isaac Lab bundle against
the locked version and its installed file hashes. Start/end evaluation validates
against the **training** run's source hashes, recipe and bundle content identity. A fresh
source capture cannot silently re-label changed evaluation code as equivalent.
No remote URL, Git configuration, or credential value is collected.

This is not a full environment attestation: unlisted assets, extra files and
installed-package state are outside this source gate. Old manifests without the
new fields require an explicit migration, not silent compatibility assumptions.
The runtime launcher now uses its own installed Python for recipe/provenance
commands; it does not create or synchronize a second CPU environment on the GPU.

## Telemetry and failure paths

Pinned Isaac Lab 2.3 stores weighted reward terms **before dt** in `_step_reward`.
`RewardRollup` accumulates every step and records both pre-dt means and actual
dt-scaled contributions per agent. It uses detached accumulators and reads no
simulator state itself. Snapshotting is read-only; partial budget-stopped rollouts
are labeled unoptimized. This is not a replacement for future synchronized
observation/action/state traces.

Termination causes are not generally an exclusive partition of episodes. The
current pinned Isaac manager additionally retains only the last configured
matching cause for an agent. Reported cause counts therefore cannot establish
all simultaneous physical failure causes. Episode totals come from dones.

On training exceptions the runner attempts an interrupted checkpoint, partial
telemetry, and video closure independently, retaining the original exception.
Checkpoint restoration does not recreate simulator/RNG/history/rollout state.

## Checks and review scope

Set `operation: check` in `config.yaml`, then run `uv run rc-car`.
Ruff/ty scope portable code and tests; selected support scripts are linted too.
Frozen historical model fixtures are deliberately excluded from formatting and
type checking. The inherited Isaac tree is **not** certified type-clean.
CPU CI is configured for Linux and macOS, with pinned action SHAs and read-only
permissions. A local pass is not a claim the remote CI workflow has executed.

Model/core authors cross-reviewed each other's implementation; the numerical
author independently reviewed configuration/tracking; the model author reviewed
root telemetry/lifecycle integration. Review findings and test evidence are
recorded in [refactor progress](refactor.md).

## Remaining issues

- Observation history still advances through inherited observation/reward call
  paths. Its redesign needs explicit once-per-control-step lifecycle integration.
- Multi-agent obstacle images still include the observing car; their reward gate
  may therefore suppress line and steering terms. Do not interpret a zero term
  as causal proof without synchronized traces.
- Image scaling is corrected in the local v3 candidate, not yet GPU-validated.
- Basic v2 wiring has GPU integration evidence; full task/reset semantics remain
  unaudited. [E006](../experiments/E006-900s-fixture-baseline/README.md) showed fewer
  resets after 900 seconds on one generated fixture, not historical convergence,
  a complete lap, held-out performance or safe multi-agent coordination.
- Independent worlds versus interacting agents need an explicit backend contract.
- Full sensor/observation/action timeline recording and a calibrated planar
  simulator are planned, not implemented in this first stage.
- skrl/MAPPO versus RLlib needs a scoped experiment, not an untested migration.
- Supported Isaac runtime migration, validation of new behavior versions, hardware
  calibration and safety/network experiments remain separate gates.

Do not start an expensive training run merely because CPU checks are green.
