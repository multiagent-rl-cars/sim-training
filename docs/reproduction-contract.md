# Reproduction contract: claims, comparability, and planned gates

Status: **planning and source audit, 2026-09-26**. No experiment is authorized by
this document. GPU work is paused; the next run requires the user's explicit go.
The order is foundation acceptance, historical reproduction, then hypothesis-led
extensions/sweeps. Do not change the benchmark while trying to reproduce it.

## What the historical sources actually claim

References below use the `PDF PAGE` markers and numbered lines in the local
[final-report extraction](../../research/final-report-extracted.txt), not guessed
page numbering. The [2026 brief](../../research/2026-brief-extracted.txt) has no
page markers, so references use line numbers. These are source claims, not results
verified by the current repository.

| Claim | Exact source location | What is and is not established |
| --- | --- | --- |
| Line following converges in approximately 15 minutes on one RTX 5070 | Report p. 10, lines 255–260; p. 19, lines 442–447 | Uses SB3 PPO; p. 19 says 12 parallel environments. No numerical convergence criterion, seed distribution, raw run, sample count, or timer boundary accompanies this claim. Figure 5.2 is a screenshot, not a lap-success measurement. |
| Parallel track diversity | Report p. 9, lines 242–246; p. 10, lines 251–254 | Up to 12 environments with 12 distinct track layouts; cached static images and vectorized terms. Twelve copies of one circle do not reproduce this distribution. |
| Simulator observations/model | Report p. 9, lines 229–238; p. 17, lines 414–421 | Ackermann car, two actions, 512×384 camera, 48×48 aligned image, velocity/orientation readings, three convolutional layers. Exact final architecture/state and preprocessing are not supplied here. |
| Line-following reward | Report p. 18, lines 429–435; p. 19, lines 440–441 | Four qualitative terms. Its “steering jerk” description explicitly uses normalized steering magnitude, not a temporal derivative. No complete weights/normalizers are provided in this passage. |
| Obstacle and multi-agent tasks | Report p. 19, lines 448–456; p. 20, lines 465–480; p. 21, lines 485–487 | Static obstacle input/termination, then shared-policy multi-car training with temporal obstacle images and collision penalty. Emergent avoidance is qualitative; no success-rate table or 15-minute multi-agent claim. |
| Control timing | Report p. 10, lines 256–260 | Reports 100 ms simulation control while also describing 60 Hz and approximately 20 ms real updates. These are not equal: 100 ms = 10 Hz; 60 Hz ≈ 16.7 ms; 20 ms = 50 Hz. Resolve measured rates instead of silently treating this as a timing match. |
| Fidelity and real fine-tuning | Report p. 10, lines 261–267; p. 24, lines 569–582; p. 25, lines 587–588 | Tire slip, motor nonlinearity, voltage sag, and latency deliberately omitted. Single-car line following reportedly transfers with degradation; approximately 3–4 hours / 170,000 real steps of fine-tuning, learning rate 0.0003 → 0.0001. “Steps” and improvement lack a full quantitative definition. |
| Reset-agent convergence | Report p. 28, lines 656–669; p. 29, lines 675–677 | Approximately one minute with 1,000 environments; plotted reward/episode-length/value diagnostics. A separate task, not evidence for the line-following timing claim. Critic explained variance is not a driving-success metric. |
| Year-2 gaps | Brief lines 24–43 | Multi-agent zero-shot transfer degraded; no inter-agent communication; only line following fine-tuned in real; intersections validated only in simulation; oscillation, low-speed mapping, reset safety, and ~20+ car scaling remain directions. |

The brief's broad “reliable infrastructure” summary (lines 11–20) is not a measured
latency/scale guarantee: the same brief explicitly flags the scaling margin and
chassis/control issues (lines 40–44). Keep single-agent and multi-agent transfer
claims separate rather than treating their different outcomes as a contradiction.

## Three different baselines

| Dimension | Reported Year-1 line follower | Recovered `baseline-002` fixture result | Current refactored repository / next validation |
| --- | --- | --- | --- |
| Task/worlds | Single car in each of 12 distinct arenas | Four interacting cars in one circular world | Basic: 12 separate repeated-circle arenas; Multi: four cars sharing one world. Separate checks. |
| Assets | Claimed digital twin/original track library | Generated, uncalibrated six-joint car; radius-3.3 m circle on 10×10 m floor | Same explicitly labeled fixtures until final benchmark assets and calibration are established; older recovered USDs are not yet validated. |
| Compute | RTX 5070 | Recorded RTX A6000; 206.606 s training | No new GPU result established; host allocation is not a benchmark result. |
| Control/camera | Reported 0.1 s; 512×384 frame | 1/30 s control; fixture 512×512 frame | Fixture overrides retain 1/120 s physics, decimation 4, 1/30 s control, 512×512 camera. |
| Policy inputs | One 48×48 image plus described scalar velocity/orientation | Track, four-channel peer history, two velocity components | Basic also has a one-channel obstacle image, even when constant white with obstacles disabled; heading orients crops but is not a separate actor scalar. Multi retains four-channel peer history. |
| Learning | SB3 PPO; exact final run unidentified | 16,384 aggregate agent samples, 16 full rollouts, four optimization epochs each | Tiny smoke recipes use two rollouts, one epoch each, and a 60 s training budget; not a convergence test. |
| Behavior identity | Final code/assets/checkpoint not linked to claim | Preserved old source/config/weights and observed result | `astra-foundation-v2`: nonmutating velocity plus intentional Basic obstacle-routing/local-coordinate fixes. Not trajectory-equivalent to historical code. |
| Evidence | Report narrative/screenshots | Matched fresh-process evaluations and retained media/data | CPU/fake-host gates only until new artifacts are inspected; model equivalence tests do not establish physics fidelity. |

Current fixture identity is in [fixture.json](../generated_assets/fixture.json).
Fifteen older January 2026 USDs were recovered from Git history; see the
[recovery inventory](../../research/historical-simulation/assets-recovered/README.md).
Their dependencies/scale/Isaac compatibility and association with the final
reported benchmark remain unverified; newer variable-thickness tracks, Nano
model and shared MARL track were not found in that recovery. “Missing benchmark
assets” does not mean that no original asset bytes exist locally.
Source behavior and unresolved history/ego/image-scaling issues are documented in
[architecture.md](architecture.md). Preserve the September 25 pre-refactor archive
and September 26 frozen source archive as historical records; never overwrite them
with the new implementation and call them the same baseline.

The recovered result's [training summary](../outputs/experiments/baseline-002/training_summary.json)
and [comparison](../outputs/experiments/baseline-002/comparison.json) report mean
return **16.18 → 30.65**, reset-excluded distance **27.69 → 57.02 m**, but resets
**16 → 34** (off-track **14 → 30**, proximity/collision **2 → 4**) over matched
450-control-step evaluations. That establishes faster movement under its objective,
not improved driving or reproduction of the single-agent 15-minute claim. This is
a reading of retained measurements, not a new analysis run.

## Hyperparameter provenance

The inherited [SB3 YAML](../source/car_training/car_training/tasks/manager_based/car_training/agents/sb3_ppo_cfg.yaml)
is code evidence, **not a verified manifest of the reported 15-minute run**:
seed 42; `n_timesteps=1e6`; `MultiInputPolicy`; `n_steps=1024`; batch 256;
10 epochs; learning rate 0.0003; gamma 0.99; GAE lambda 0.95; clip 0.2;
entropy coefficient 0.01; value coefficient 1.0; max gradient norm 1.0;
gSDE enabled, resampled every 4 steps; separate `[64,64]` actor/critic heads,
Tanh activations, `squash_output=False`, CUDA device. It does not identify the
exact assets, completed policy, package versions, or convergence criterion.

The `line-following-15min` profile in [config.yaml](../config.yaml) copies
those PPO values but requests 10,000,000 agent samples with a 900 s callback budget;
its train/evaluation seeds are 42/123. It is a new timed fixture attempt, not a
historical manifest. Use each run's actual GPU/manifest, not historical limitation
notes, to establish hardware identity.
The [recovered Multi recipe](../outputs/experiments/baseline-002/config.json)
instead uses 256 steps per car/rollout and four epochs. The `basic-smoke` and
`multiagent-smoke` profiles in the same YAML intentionally
use small rollouts and schema-resolved defaults; do not describe their effective
PPO configuration as the inherited YAML. Save both requested and resolved recipes.
The old JSON recipe bytes remain only under `tests/fixtures/legacy_recipes` as
immutable numerical-comparison inputs, never an alternative runtime configuration.

Current Basic reward, per control transition, is
`dt × [3g(1 − d_px/40) + 1.5||v||/0.8 + 0.2 − 0.5g|steer_rad|/0.6]`,
where `g=0` when its static-obstacle crop contains a qualifying black pixel and
`g=1` otherwise (always 1 when obstacles are disabled). Distance is to an extracted
contour, not an established centerline; speed magnitude does not distinguish
forward progress or reverse motion. Multi changes steering weight to 0.75, adds
its collision term, and uses the inherited peer-history gate. Sources:
[reward adapter](../source/car_training/car_training/tasks/manager_based/car_training/mdp/rewards.py),
[Basic configuration](../source/car_training/car_training/tasks/manager_based/car_training/car_training_env_cfg.py),
[portable kernels](../src/rc_car/core/rewards.py). Similar term names do not make
different gates, pixel scales, dt values, or dynamics equivalent objectives.

## Counting and timer contract

- A **physics step** advances the simulator by `sim.dt`. One **vector control
  step** applies one action per car and advances `decimation` physics steps.
  Render frames, policy calls, and history reads are not interchangeable steps.
- One **agent sample** is one car's control transition. With N cars,
  `agent_timesteps = N × vector_control_steps` for synchronous collection.
  Multi cars interact; N correlated samples are not N independent worlds.
- SB3 `n_steps` means control transitions **per car per rollout**. Full rollout
  samples are `N × n_steps`: Basic smoke 12×16=192, Multi smoke 4×32=128.
  Their requested totals 384/256 mean two rollouts if the budget permits.
- Record collected samples and optimized rollout samples separately. Callback
  termination can leave a partial, unoptimized rollout. `ppo_updates` currently
  stores SB3's `_n_updates` **epoch counter**, not optimizer minibatch steps or
  rollout count. In the recovered run: 16 rollouts × 4 epochs = 64; nominal
  minibatch optimizer steps are 16×4×(1024/256)=256. Changed ZIP hashes alone do
  not prove changed parameters.
- **Training wall time:** current summary timer surrounds `model.learn`; the
  callback budget starts at callback construction just beforehand. Includes
  collection, optimization, callback telemetry and video work; excludes install,
  simulator/model startup, subsequent evaluation and transfer. The callback
  checks only at control-step boundaries; optimization/video work can overrun.
- **Whole-job time:** outer GNU timeout covers provenance/fixture checks, host
  preflight, training and evaluations, with explicit kill grace. It is distinct
  from the callback budget and from setup time. Neither timeout nor process
  exit bounds cloud billing or proves instance deletion.
- **Evaluation time:** report control steps and simulated seconds explicitly.
  Current 450-step fixture evaluation is 15 s; 450 historical 0.1 s transitions
  would be 45 s. Compare matched simulated durations and also disclose sample
  counts; neither alone creates equivalence across different dynamics.

## Planned pass / fail / not-comparable decisions

1. **Foundation integration — pass only with artifacts.** Correct source/assets
   and pinned runtime, finite observations/rewards/weights, expected input shapes,
   positive reset-excluded movement for every car, actual parameter updates,
   checkpoint reload, nonempty decodable simulator videos, and matching fresh
   initial/final evaluation states. A violated invariant is **fail**. Missing
   required evidence is **not established**, not an inferred pass. Basic enabled
   obstacles, Basic disabled obstacles, and Multi are distinct checks.
2. **Historical comparability — establish before timing.** Obtain original USDs,
   final code/config/runtime/checkpoint association, track assignments, seed and
   evaluation protocol, exact “converged” criterion, and timer start/stop rules.
   Missing originals or changed task/assets/objective/timing makes strict
   reproduction **not comparable**, even if a fixture reward curve looks good.
   Hardware changes permit a clearly labeled cross-hardware result, not a matched
   RTX-5070 wall-time replication.
3. **15-minute claim — currently not adjudicable.** No numerical historical
   convergence threshold was found in these sources. Do not invent one or use
   reward growth as its substitute. Once a defensible criterion is recovered and
   frozen, report time/sample count to it; failing to reach it within the
   preregistered 900 s measurement window is **fail for that run/protocol**, not
   proof the published result was false. Report all seeds, including failures.
4. **New behavioral acceptance — proposal, not a historical threshold.** Before
   any behavior comparison, approve a held-out track/start-state set, directions,
   seeds, evaluation duration, minimum nonreset route progress, full uninterrupted
   lap definition, collision/off-track rates, and steering-oscillation metric.
   One uninterrupted valid lap is a useful proposed first milestone, not proof
   of convergence or a substitute for multi-seed generalization. Do not select
   numerical pass cutoffs after seeing test outcomes.
5. **Real transfer — separate later gate.** Use physical camera/control traces,
   real videos/photos, vehicle/track/calibration identity, battery condition,
   latency distribution and intervention/reset log. Label zero-shot versus
   real fine-tuning and account for their samples separately. Simulator clips
   and synthetic W&B test videos are not physical-world evidence.

The minimum comparison packet should pair representative **and failing** clips
with trajectories, actual policy-input panels, terminations, weighted reward
components before/after dt, parameter-change evidence, source/asset hashes,
resolved configuration, and a written differences table. Show full episode/reset
context rather than selecting a favorable few seconds. Cached-map simulator
inputs do not constitute a real vision-pipeline test.

After foundation acceptance, run one frozen baseline and then one explicitly
stated hypothesis per change. Keep hypothesis, fixed controls, expected
falsification, decision rule, result and next decision together; register work in
the root-owned tickets/experiment registry. Recovery of original benchmark
materials comes before sweeps, reward redesign, learner migration or claims of
real-world improvement.
