# Open issues from the cleanup review

Reviewed 2026-10-02 for [PR #1](https://github.com/multiagent-rl-cars/sim-training/pull/1).
Evidence below refers to `astra-cleanup` at
`0e1199da8f230226e02071226283f5a8c51c3489`, compared with `main` at
`e42649880eba2e698b57687d4b68c172f055a448`. This list records review findings and
acceptance checks; it does not authorize fixes, GPU allocation, or a merge.
The broader research backlog remains in [tickets](tickets.md).

## Severity and evidence

- **P1:** resolve before claiming current-head training acceptance, correct
  multiagent semantics, or compatibility with main. Each entry states which
  claim it blocks; an acceptance gap is not a demonstrated runtime failure.
- **P2:** bounded input, diagnostic, or maintainability defect that should be
  addressed without expanding the current pruning change.
- **Confirmed** means reproduced in CPU code or established by the cited source.
  **Inherited** means present before the final cleanup commit; it is not a newly
  demonstrated regression. **Unvalidated** means the required runtime evidence
  has not been produced.

## P1: acceptance and scientific correctness

### 1. Exact-head Isaac runtime and image boundary remain unvalidated

**Unvalidated acceptance gate.** Frozen E001/E006 GPU results predate the current
image-amplitude correction and consolidated runtime/workflow/tracking paths.
CPU adapter tests exercise isolated methods and synthetic environments, not a
live Isaac runtime with cameras, physics, and manager/reset scheduling. See
[the evidence boundary](refactor.md#september-27-evidence-boundary),
[image contract](architecture.md), and [E003](../experiments/E003-policy-image-encoding/README.md).

**Close with:** fresh pinned-runtime installation and the four Basic/Multi smoke
and rehearsal gates on the exact final source, recording source/asset/runtime
hashes. Require finite observations/rewards, per-car motion excluding reset
jumps, an actual optimized update, checkpoint reload, matched evaluation starts,
decodable videos, and process cleanup after failure/timeout. On real Isaac,
verify helper 0/1 → scaled ObservationManager 0/255 → SB3 uint8 → CNN normalized
0/1 at reset, ordinary step, and terminal observation, plus old-marker rejection and
new-marker reload. Verify live tracking if claiming that path works. Preserve
failures; earlier videos cannot close this gate.

### 2. Ego occupancy contaminates the other-agent signal and reward gate

**Confirmed source defect; inherited, not newly GPU-reproduced.**
`generate_agents_map` paints every car into one shared map, and
`aligned_other_agents_image` crops that map without removing the observing car.
`_reward_obstacle_present` treats dark pixels from this signal as obstacles,
which can suppress line/steering reward terms even without another nearby car.
See [observation helpers](../source/car_training/car_training/tasks/manager_based/car_training/mdp/observation.py)
and [reward routing](../source/car_training/car_training/tasks/manager_based/car_training/mdp/rewards.py).

**Close with:** exclude ego occupancy per observer; test an isolated car, a
nearby other car, and overlapping cars. Show that the reward gate responds only
to intended obstacles. Preserve unrelated reward math and validate the declared
behavior change through the real Multi runtime before claiming correctness.

### 3. Observation reads can advance history more than once per control step

**Confirmed source defect; inherited, not newly GPU-reproduced.**
`aligned_other_agents_image` mutates its stack and counters on each call; reward
helpers also call it. History therefore depends on read count/order rather than
solely on simulator steps. Subset-reset isolation remains a related acceptance
gap, not a separately reproduced current-head failure. See the stack update in
[observation.py](../source/car_training/car_training/tasks/manager_based/car_training/mdp/observation.py).

**Close with:** give history updates one explicit control-step owner; make repeated
observation/reward reads idempotent. Add tests for repeated reads, reversed call
order, sampling cadence, first reset, and subset resets that preserve unaffected
cars. Confirm those invariants under actual manager/reset scheduling.

### 4. Main checkpoints and model behavior are not preserved by branch parity tests

**Confirmed branch-versus-main incompatibility; predates the final cleanup.**
With the same observation space (track `(1,48,48)`, obstacle history `(4,48,48)`,
velocity `(2,)`) and default constructors,
main has 242,816 trainable parameters and 128 output features; reviewed head has
964,864 and 256. There are 62 state-dictionary tensor-shape differences, and
strict loading of main weights into head fails. The frozen model in
[the parity tests](../tests/models/test_feature_extractor.py) matches intermediate
parent `b407792bcda16dbc1bb39db2d89e821ce8578b6d`, not main. Those tests validly
protect that extraction boundary. A legacy import alias cannot repair tensor
shapes; the [current evaluator](architecture.md) also intentionally rejects old
checkpoints without the image-encoding marker.

**Close with:** explicitly define the supported checkpoint/model versions and
test main versus the intermediate parent separately. Preserve the original
runtime for old checkpoints, or design and validate an explicit migration; do
not silently change the architecture to make a test pass. For a main-equivalent
training claim, use matched model/config/assets, seeds, starts, and budgets,
listing intentional architecture/image changes separately.

## P2: configuration, diagnostics, and maintenance

### 5. Validated configurations can disagree with the runtime task contract

**Confirmed CPU validation inconsistency.** The schema and plan accept
`num_envs=1` when `n_steps=64` and `batch_size=64`, but the training worker rejects
fewer than two environments. They also accept unknown task names and the reset
task, while fixture setup handles Multi explicitly and routes every other string
through Basic-style setup. Unknown-task acceptance was reproduced; subsequent
Isaac failure was not. See [schema](../src/rc_car/experiments/config.py),
[worker guard](../scripts/training_experiment.py), and
[fixture selection](../scripts/fixture_config.py).

**Close with:** one shared supported-task/environment-count contract. Either
support one environment end to end or reject it consistently. Reject unsupported
tasks before authentication, output allocation, or runtime startup, with focused
schema/plan/worker regression cases.

### 6. Public CLI errors hide actionable, safe diagnostics

**Confirmed diagnostic defect.** [The public entry point](../src/rc_car/workflow.py)
catches `OSError`, `ValueError`, and `RuntimeError`, prints only the exception
class, and points to local logs even when configuration failure occurs before
logs exist. Avoiding disclosure of arbitrary user data is intentional, but the
current message also hides safe field/path guidance.

**Close with:** emit sanitized, actionable field/reason information and reference
logs only when they exist. Test invalid YAML, missing files, invalid profile/task
settings, and runtime errors without leaking secrets or arbitrary input.

### 7. Two active coordinate implementations can drift

**Confirmed duplication; no numerical defect demonstrated.** Portable
[`convert_coordinates`](../src/rc_car/core/geometry.py) coexists with legacy
[`convert_coords` / `convert_coords_list`](../source/car_training/car_training/tasks/manager_based/car_training/mdp/utils/helpers.py),
and [Multi reset setup](../source/car_training/car_training/tasks/manager_based/car_training/multiagent_training_env.py)
still uses the legacy list conversion. Independent comparison against actual main
passed 160 cases bitwise (two dtypes, aspect modes, and forward/inverse paths).

**Close with:** route supported callers through one implementation with thin
compatibility adapters if needed. Keep the existing portable kernel and parity
tests; cover list/tensor inputs, non-square dimensions, dtype/device, inverse
conversion, and nonmutation before removing duplication.

## Evidence to retain

The reviewed source passed an independent fresh locked CPU install and
`uv run --locked rc-car`: Ruff, formatting, ty, and all **711 tests**. Its
[Linux/macOS CI run](https://github.com/multiagent-rl-cars/sim-training/actions/runs/36954670175)
also passed. These checks support the portable foundation, CPU PPO/checkpoint
round trips, numerical kernels, and tested failure handling. Keep that coverage.
They do not establish GPU integration, whole-repository coverage, convergence,
full laps, held-out driving quality, or transfer to a real car. Re-run the CPU
gate and CI after changes; keep all remaining acceptance boundaries explicit.
