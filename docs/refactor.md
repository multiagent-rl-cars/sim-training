# Simulation training refactoring plan

Original plan: 2026-09-24. Current status updated 2026-09-27.

The portable foundation is implemented on `astra-cleanup`; the broader migration
and scientific/hardware gates are not complete. Use the [current entry point](../README.md)
and [architecture](architecture.md), not commands or cloud state from dated review
sections below. This document preserves the implementation/review history.

Historical command records and proposed layouts below are unsupported. The
current interface is `uv run rc-car`, with one root dependency manifest/lock and
optional Isaac dependencies; see the README for that selection.

### September 27 evidence boundary

- Frozen `astra-foundation-v2` passed all four [E001 integration gates](../experiments/E001-foundation-integration/README.md).
- [E006](../experiments/E006-900s-fixture-baseline/README.md) completed a 900-second
  fixture baseline. In the matched 15-second evaluation, off-track resets fell
  from 66 to 0 and reset-excluded travel increased. No full lap, held-out result
  or strict reproduction of the earlier team's claim was demonstrated.
- All 240 result files were backed up and verified; the GPU instance was deleted.
  Runtime binaries were not backed up and can be rebuilt from the pinned setup.
- [E003](../experiments/E003-policy-image-encoding/README.md) freezes an isolated
  v3 image-amplitude correction. Its 453-test snapshot passed local gates but
  still needs four GPU gates. The existing videos are v2 evidence, not v3 proof.
- The current working tree additionally consolidates the workflow into
  `config.yaml` and adds explicit online tracking/recovery verification. Those
  changes are outside both frozen snapshots. Online recovery status is recorded
  in [server verification receipts](../outputs/wandb-upload-20260927/summary.json),
  not inferred from a successful sync command.

Full-branch acceptance requires a frozen reviewed diff, a fresh local quality
gate, independent review of the later workflow/tracking changes and resolution
of the v3 GPU boundary. Either validate that correction or separate it into a
follow-up; do not transfer v2 runtime claims to newer source.

The [work plan](work-plan.md) and [tickets](tickets.md) distinguish remaining
refactor work from later reproduction, demonstrations, tuning and extensions.

## First-stage implementation and review

- Preserved the pre-refactor source and both locks in a checksummed archive; see
  [ADR 0001](decisions/0001-portable-foundation.md). Existing outputs/handoffs and
  unrelated dirty-worktree changes were preserved.
- Added an installable `src/rc_car` package: tensor geometry/action/reward kernels,
  portable model definitions, strict experiment recipes, optional W&B integration,
  rollout telemetry, and failure-recovery helpers.
- Kept legacy model imports as aliases and connected existing Isaac adapters to
  the extracted kernels. Kept the inherited objective and model architecture.
- Deliberately fixed velocity-read mutation; manifests distinguish the resulting
  behavior as `astra-foundation-v1`, not historical trajectory parity.
- Added `rc-car validate-config` and `show-config`, resolved configuration records,
  full-rollout reward decomposition, and protected interrupted-checkpoint cleanup.
- Installed Ruff, ty, pytest/Hypothesis; added explicit portable-code checking
  scope and CPU CI for Linux/macOS. No Pyright gate. GPU imports remain a separate
  integration boundary, not hidden behind blanket type ignores.
- Preserved the original README in `docs/history/` and replaced the entry page
  with current workflow and architecture links.

### Independent review record

| Author scope | Independent reviewer | Findings and disposition |
| --- | --- | --- |
| Portable models | Tooling/config agent | Historical fixture is independent; model/gradient/checkpoint comparisons pass; no blocking model findings |
| Numerical core + Isaac call sites | Model agent | Bit-exact coordinate comparisons and adapter-body tests pass; intentional non-mutating velocity fix explicitly classified |
| Recipe/tracking extraction | Numerical-core agent | Fixed SDK active-run reuse, atomic identity publication, seed bounds, unnecessary minibatch divisibility restriction, and duplicate JSON keys; re-reviewed |
| Telemetry/failure integration | Model agent | Verified pre-dt reward semantics; added partial-rollout/error-path tests; fixed checkpoint prevention and outer cleanup error masking; documented pinned manager's last-cause limitation |
| Runtime packaging | Numerical-core agent | Both locks validate; all pre-existing Isaac package versions/sources unchanged; only local portable package added |
| Basic task wiring (September 26) | Model agent | 35 core/adapter tests pass; static-obstacle routing, steering arguments and local arena frames corrected; Multi defaults preserved |
| Source/evaluation identity (September 26) | Model agent | 18 provenance tests pass; evaluation now binds to training source/recipe/upstream revision, not a recaptured manifest |
| Actual offline W&B gate (September 26) | Model agent | Independently verified persisted scalars/video/artifact/exit; 5 boundary/failure tests pass after cleanup fixes |

These are source/CPU reviews, not proof of physics or remote runtime behavior.
The model compatibility test isolates legacy module paths; it does not establish
that every historical checkpoint loads on a Mac without a migration step.

### September 26 continuation

- Full local gate resumed at 162 passing tests before new work; the final
  simultaneous video/telemetry failure case received independent re-review.
- Basic wiring corrections are recorded under `astra-foundation-v2`; they are
  intentional behavior changes, not a claim of historical trajectory parity.
- Added source-integrity and matched-evaluation preflight before Kit startup,
  actual clean upstream checkout identity, and direct CPU regression tests.
- Verified the real W&B SDK offline with three synthetic scalars, a two-second
  synthetic MP4, an artifact and successful exit. No upload or training occurred.
- Added `basic_smoke.json` and `multiagent_smoke.json`: tiny PPO rehearsals with
  offline recording and 60-second training budgets. These are validated recipes,
  not yet validated GPU runs. Startup/evaluation/billing need separate bounds.
- Final combined local gate: **193 tests passed**, Ruff lint/format and ty clean.
  Both locks validate (66 root / 226 Isaac packages); installed development
  dependencies pass compatibility checks. This does not mean remote CI ran.
- Current restart state and exact next gates:
  [September 26 handoff](https://github.com/multiagent-rl-cars/sim-training/blob/0e1199da8f230226e02071226283f5a8c51c3489/docs/handoffs/2026-09-26-resume.md).

### September 26 preparation record and remaining stages

The next September 26 local gate passed 366 tests with Ruff lint/format and ty clean. Local
operations now include verified source/fixture/result bundles, source-drift
guards, bounded shell supervisors, offline job isolation, and an artifact-based
integration inspector. Independent reviews fixed nested-timeout group escape and
status-write error masking. See [ops/README](../ops/README.md). At that date these
were locally tested preparations, not real Linux/Isaac or billing-safety
validation. The [September 26 operations handoff](https://github.com/multiagent-rl-cars/sim-training/blob/0e1199da8f230226e02071226283f5a8c51c3489/docs/handoffs/2026-09-26-local-only-ops.md)
preserves that state; the September 27 evidence boundary above supersedes it.

The migration sequence below remains the roadmap. In particular, history
lifecycle, ego masking, image normalization, remaining task/reset semantics,
world/agent separation, full synchronized traces, calibrated planar simulation,
backend comparison, and GPU validation of newer behavior are not complete. Do not interpret the
first-stage passing suite as approval for a long training run.

## Purpose

Make this repository understandable, testable, and reproducible for the next eight
months of RC-car research. A contributor should be able to identify the active
task, policy, reward, experiment configuration, and evaluation criteria without
reading several alternate training scripts.

Retain Isaac Lab and SB3 as historical baselines, not immutable eight-month
choices. Use uv, Ruff, ty, pytest/Hypothesis and local artifacts with optional W&B.
Evaluate one successor learner when the research question needs it (skrl/MAPPO
for cooperative MARL; RLlib if opponent populations lead). Preserve existing Isaac
extension packaging initially. Separate structural changes from changes to the
learning problem so that their effects can be evaluated independently.

## What Omniverse does here

Omniverse is NVIDIA's platform of technologies for constructing and simulating
3D worlds. Relevant pieces include OpenUSD scene descriptions, the Kit application
and extension framework, and rendering infrastructure. Isaac Sim builds robotics
simulation on that foundation, including physics and sensors. Isaac Lab adds
robot-learning environments and interfaces to learning libraries. In this project,
SB3 PPO trains the policy, while W&B records experiment information.

The dependency chain is:

```text
Our tasks and experiment code
  -> Isaac Lab: environments, observations, rewards, actions, resets
  -> Isaac Sim: robot physics and sensor simulation
  -> Omniverse technologies: scene/application/rendering infrastructure

SB3/PyTorch: policy and optimization     W&B: tracking and artifacts
```

The project does not need the example Omniverse UI extension just to train a
policy. Removing that example must not remove the Isaac runtime dependencies.

## Findings motivating the work

These are pre-refactor source-review findings from 2026-09-24, not a current
inventory or claims that all paths have been executed. See first-stage progress
above for changes since that review.

| Area | Pre-refactor evidence (2026-09-24) | Consequence |
| --- | --- | --- |
| Packaging | `source/car_training/car_training/` follows extension/package nesting | Preserve initially; folder repetition is not the main defect |
| Template tasks | `tasks/direct/car_training*` still implement cart-pole/pendulum behavior | Car-related names and registrations misrepresent their purpose |
| Training | Four trainers in `scripts/sb3/`, plus `scripts/training_experiment.py` | Policy construction and configuration can diverge between entry points |
| Task composition | Basic driving, reset, and multi-car tasks share broad `mdp` modules | Task-specific assumptions leak across environments |
| Reward wiring | Basic steering configuration passes `multi_obstacle_cfgs` to a function that does not accept it | Configuration/function compatibility needs validation |
| Reward semantics | `distance_from_center_reward` uses nearest extracted outer-contour points | Its name suggests a centerline quantity that is not explicitly computed |
| State ownership | Reward helpers call observation helpers that can update history buffers | Results can depend on call order and repeated reads |
| Imports | Package initialization imports tasks and a template UI extension | Lightweight research functions are difficult to import without Isaac |
| Outputs | Debug images exist at the root; helpers create output directories inside the source tree | Source and generated artifacts are mixed |
| Assets | Original USD assets are incomplete; generated fixtures and cached images/tensors coexist | Asset provenance and intended use are unclear |
| Documentation | README combines current setup, historical instructions, and template content | No concise authoritative workflow |
| Checks | Five local tracking/configuration tests; no `.github` CI directory; pre-commit references missing license-header files | Numerical and simulator behavior are not adequately protected |

Review starting points:

- [Task registration](../source/car_training/car_training/tasks/manager_based/car_training/__init__.py)
- [Basic environment configuration](../source/car_training/car_training/tasks/manager_based/car_training/car_training_env_cfg.py)
- [Reward functions](../source/car_training/car_training/tasks/manager_based/car_training/mdp/rewards.py)
- [Observation functions](../source/car_training/car_training/tasks/manager_based/car_training/mdp/observation.py)
- [Geometry/image helpers](../source/car_training/car_training/tasks/manager_based/car_training/mdp/utils/helpers.py)
- [Current experiment runner](../scripts/training_experiment.py)
- [Existing training analysis](training-story.md)

## Target structure

This original extension-centric target is superseded for portable code by
`src/rc_car/{core,models,experiments}` plus telemetry/lifecycle/CLI modules. The
legacy extension remains under `source/car_training` as an adapter. Other
directories in the diagram are future migration targets, not completed modules.

```text
sim-training/
├── README.md
├── pyproject.toml / uv.lock       # Mac/CPU development and tests
├── environments/isaac/           # Locked Linux simulator runtime
├── config.yaml                  # Sole editable workflow and named profiles
├── source/car_training/
│   ├── config/extension.toml
│   └── car_training/
│       ├── core/                 # Simulator-independent math and transforms
│       ├── sim/                  # Isaac adapters, assets, scene construction
│       │   └── tasks/
│       │       ├── line_following/
│       │       ├── obstacle_avoidance/
│       │       ├── shared_track/
│       │       └── reset/
│       ├── policies/             # Networks and observation specifications
│       ├── training/             # PPO construction, callbacks, checkpoints
│       ├── evaluation/           # Rollouts, metrics, comparison tools
│       ├── tracking/             # Local logs, W&B, artifact manifests
│       └── cli/                  # Thin train/evaluate/check commands
├── assets/                       # Manifests and generation recipes
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gpu/
├── ops/                          # Setup, preflight, sync, bounded remote jobs
├── docs/
│   ├── architecture.md
│   ├── tasks/
│   ├── runbook.md
│   ├── decisions/
│   └── history/
└── outputs/                      # Ignored generated run artifacts
```

Create task directories when supported implementations exist; this tree is not
a reason to manufacture abstractions or claim that all four tasks already work.
Keep upstream Isaac Lab as an ignored, pinned dependency. Do not edit it to hold
project functionality. Do not combine the Mac and Linux environments into a lock
that requires installing the simulator to run local tests.

## Architectural boundaries

### Simulator-independent research code

`core` receives explicit scalars/arrays/tensors and returns results. It must not
import Isaac, access an environment object, start an application, write files on
import, or implicitly mutate observation history. CPU PyTorch is appropriate for
tensor implementations; distinguish numerical parity tolerances from exact equality.

Isaac adapters extract state and call this code. For example, a future reward
interface could accept lateral error, forward speed, steering, and a typed reward
configuration, returning named terms and their sum. This is an interface proposal,
not permission to replace the inherited contour-distance/speed-magnitude objective
with a centerline/forward-progress objective during a move.

Maintain explicit contracts for units, coordinate frames, tensor shapes, image
ranges, steering/throttle scaling, and whether values are pre- or post-time scaling.
Make time scaling occur exactly once.

### Tasks and state

Each task owns its observation specification, reward composition, terminations,
reset semantics, and success criteria. Share implementations only where their
meaning is actually shared. Distinguish independent parallel single-car environments
from several cars interacting on one track.

History buffers advance once per environment step and reset explicitly per affected
environment. Observation reads and reward calculations must not advance them.
Environment initialization and task registration should be explicit simulator
operations, not side effects of importing `rc_car.core` or a portable policy module.
The inherited `car_training` package still eagerly imports Isaac and is not the
portable entry point.

### Configuration and entry points

Provide one supported training command and one evaluation command. Select network
variants through configuration rather than duplicating the training program.
Keep Isaac scene declarations in Python where required; experiment recipes select
those declarations and specify assets, policy, PPO settings, seeds, limits, and
recording/tracking settings.

Define override precedence, reject unknown fields, validate reward call signatures,
and save the fully resolved configuration before training. Every effective parameter
must be discoverable in that record. Preserve old task IDs/import paths with temporary
compatibility aliases when necessary; do not silently redirect them to changed tasks.

### Assets and results

Version asset manifests and generation recipes. Record hashes, units, geometry
versions, calibration status, and source/provenance. Store large asset bundles in
an explicitly selected artifact store; keep generated fixtures clearly distinct
from recovered original assets. Do not migrate large binaries or rewrite Git history
as an incidental cleanup step.

Each new run gets a unique directory containing resolved configuration, source and
asset identity, runtime/GPU details, structured metrics, checkpoints, videos, and
evaluation results. All debug output belongs there. W&B mirrors selected results;
it must not be the only copy. Credentials never enter configurations or artifacts.

Checkpoint loading must document what is restored: policy/optimizer state is not
automatically equivalent to restoring simulator, RNG, observation history, and
in-progress rollout state.

## Migration sequence and completion gates

### 1. Preserve and inventory

- Preserve the dirty worktree, untracked recovery tooling, existing outputs, and
  lockfiles. Record source hashes and baseline artifact locations before moving code.
- Classify task registrations, trainers, assets, and docs as supported, experimental,
  historical, or template-only. Confirm call sites before retiring anything.
- Establish a historical reference snapshot without treating it as validated driving.

Gate: every existing artifact and task path has a known disposition; no results lost.

### 2. Establish the supported workflow

- Choose the bounded experiment runner as the starting integration path, subject to
  fixing its known wiring problems; it is not yet a validated line-following trainer.
- Factor training, evaluation, tracking, and CLI responsibilities into modules.
- Consolidate configuration resolution and policy selection. Keep temporary wrappers
  for old commands and clearly mark unsupported historical behavior.

Gate: one documented command path resolves and records the intended task and policy;
configuration errors are caught locally wherever possible.

### 3. Extract and characterize research logic

- Extract geometry, reward arithmetic, observation transformations, action scaling,
  and history management behind explicit contracts.
- Add characterization cases for inherited behavior, including known defects.
- Compare old/new outputs on fixed inputs with stated tolerances. Do not silently
  normalize, clip, change units, or change the objective while extracting code.

Gate: simulator-independent imports and behavioral tests run on the Mac; numerical
differences are either eliminated or explicitly classified as subsequent fixes.

### 4. Fix defects as separate changes

- Resolve reward argument mismatches and single-car/multi-agent helper leakage.
- Correct ego masking, image normalization, and history side effects where verified.
- Audit reset state, episode timeouts, and termination versus truncation semantics.
- Decide and document whether the desired objective uses contour distance, centerline
  error, route progress, steering magnitude, or steering-rate penalties.

Gate: each fix has a regression test and a documented behavioral change. Reward
redesign gets a new experiment identity rather than overwriting the inherited baseline.

### 5. Clean structure and documentation

- Move operational scripts into `ops`; leave only thin compatibility entry points
  where useful. Move shared script logic into importable modules.
- Remove template tasks/UI from supported registration after checking dependencies;
  preserve their historical source through the reference snapshot/Git history.
- Consolidate model definitions; organize task-specific modules; route generated
  files out of source. Preserve useful debug images as historical artifacts.
- Make README a short entry point linking architecture, task definitions, testing,
  and runbook. Move dated handoffs into history and label stale claims.
- Repair lint/pre-commit configuration and add CPU CI. Keep GPU checks explicit and
  bounded rather than requiring paid hardware for every small change.

Gate: a new contributor can locate and explain the active training configuration,
observation/action contracts, reward equation, and evaluation procedure from docs.

### 6. Validate the integrated system on GPU

- Verify driver, CUDA, Vulkan, required assets, and runtime versions before training.
- Run a bounded smoke test: finite state/observations/rewards, actual vehicle movement,
  sensible resets, rendered camera frames, and changed PPO weights.
- Run a short instrumented rehearsal before a 900-second experiment.
- Evaluate initial/final policies in fresh processes with matched seeds and initial
  states. Inspect videos and report terminations and reset-excluded route/travel metrics.
- Copy results back, verify integrity, and perform the agreed instance cleanup.

Gate: tests and artifacts substantiate the pipeline and observed behavior. Generated
fixture success must not be described as reproduction of the historical result or
as validation of real-car performance.

## Test strategy

| Layer | Examples | Where |
| --- | --- | --- |
| Unit | Coordinate transforms; reward signs/scales; finite outputs; action bounds; image range; history read idempotence and reset isolation | Mac/CPU CI |
| Integration | Config/function compatibility; observation-to-policy shapes; PPO construction on a small test environment; checkpoint round trip; W&B offline and synthetic upload checks | CPU where dependencies permit |
| GPU smoke | Isaac startup; asset/joint resolution; renderer; movement; resets; real observations; PPO parameter updates; VRAM use | Linux NVIDIA host |
| Evaluation | Matched states; no reset jumps in progress; uninterrupted laps; failure rates; held-out tracks and multiple seeds | GPU runs plus local analysis |

Before the refactor, five tests covered configuration/tracking plumbing only. The
first stage adds numerical/property, model/gradient/checkpoint, strict-config,
tracking-failure, CLI/import, telemetry and lifecycle coverage. None establishes
correct physics or driving. There is no established held-out driving benchmark
yet. Define one before tuning policies against success claims.

## Decisions to retain

- One repository and existing extension packaging during the initial migration.
- uv-managed local tooling and a separate pinned Linux runtime.
- One training/evaluation workflow with policy and task variants selected explicitly.
- Pure research functions behind thin simulator adapters.
- Local structured artifacts plus W&B; incremental remote sync and bounded jobs.
- Preserve historical behavior first, then fix or redesign it with separate evidence.
- No framework rewrite, multi-cloud platform, or distributed sweeps as a prerequisite.

First implementation milestone: preserve the baseline, remove eager simulator imports
from the future lightweight package path, validate task/reward configuration contracts,
and extract one reward/geometry slice with meaningful CPU tests. Demonstrate this
boundary on a small slice before moving the entire codebase.
