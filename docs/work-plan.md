# Work plan: foundation first, research second

Updated 2026-09-27. Branch: `astra-cleanup`.
This is a personal technical work plan, not a shared team commitment or a launch
authorization. The [ticket list](tickets.md) is the current execution order.
The [canonical workflow](../README.md) owns current commands; this page keeps
research sequencing separate from implementation instructions. Frozen v2 passed
[E001](../experiments/E001-foundation-integration/README.md) and the narrow
[E006](../experiments/E006-900s-fixture-baseline/README.md) fixture rule. The GPU was
deleted after backup. The newer v3 image correction still needs its own GPU gates.

## What counts as finishing the refactor?

The first portable slice exists; the whole refactor is not finished. Its purpose
is to let us explain and test the system before spending weeks training it.

1. **Local foundation:** one documented recipe/runner path, pure model/reward
   modules, resolved configurations, ty/Ruff/CPU tests, source and asset identity,
   recoverable logs/checkpoints, and independently reviewed failure handling.
2. **Semantic audit:** explicit observation/action units, coordinate frames,
   image ranges, history ownership, reset/termination versus truncation behavior,
   and reward decomposition. Preserve characterization tests; each correction
   needs a regression test and a new declared behavior identity. Inherited
   defects are not approved merely because a parity test preserves them.
3. **Integration acceptance:** each frozen behavior version must pass Basic with/without
   obstacles and Multi on one shared RTX host. Require actual motion, changed
   parameters, reload, matched evaluations and readable recordings. Test cleanup
   and backup on failures too. CPU tests alone cannot close this milestone.
4. **Navigation/introspection:** a contributor can trace camera/state → policy
   inputs → actions → dynamics → reward/termination for the same control step.
   Document supported versus archival entry points, then retire duplication.

SB3 and the pinned Isaac runtime remain comparison baselines. Do not combine a
learner migration, runtime upgrade, new assets, and reward redesign in one change.
There is no claim that the code is bug-free; tests and independent review reduce
specific risks while the remaining boundaries stay explicit.

## After foundation acceptance

| Stage | Question | Required evidence / stopping rule |
| --- | --- | --- |
| Recover benchmark | What exactly did Year 1 measure? | Final code/config/asset/checkpoint association, track set, seeds, convergence definition and timer boundary. Stop calling it strict reproduction if these cannot be recovered. |
| Frozen baseline | Can the new repository reproduce that protocol? | Same protocol and declared differences, all seeds, time/sample counts, matched fresh-process evaluations, representative and failing videos. One run first; no sweep to hide a failed baseline. |
| Corrected baseline | Do individually audited fixes improve driving? | One behavior change per comparison; held-out starts/tracks; reward plus laps, collisions, off-track, resets and steering oscillation. |
| Efficiency | What buys useful driving per sample and per minute? | Matched budgets, optimized versus collected samples, throughput, startup/evaluation/video overhead, GPU and cost record. A fast renderer alone is not a better policy. |
| Research direction | Which extension resolves the most important remaining failure? | One preregistered hypothesis, unchanged controls, an ablation and a decision. Choose from the candidates below only after baseline evidence. |

The [reproduction contract](reproduction-contract.md) separates report claims,
recovered measurements and proposals. Notably, the reported 15-minute result is
single-agent line following on an RTX 5070 with 12 distinct tracks. Our retained
`baseline-002` is four interacting cars on an uncalibrated circular fixture.
Higher reward accompanied more resets; it is not proof of better driving.
Fifteen older USDs have been recovered, but the final benchmark assets and their
compatibility/calibration are not established. Do not substitute them silently.
The newer E006 Basic fixture baseline improved its specified reset/travel metrics,
but does not fill the historical identity or generalization gaps.

## Demonstrations should let a skeptical viewer disprove our claims

For each eventual driving comparison, preserve initial, intermediate and final
checkpoints; raw full-episode clips; fixed-time still frames; and at least one
failure. Show top-down motion and the policy's actual input panels, with aligned
control step/simulated time, actions, speed, reward components, and reset causes.
Keep a trajectory plot and reset markers alongside the video. Label simulator,
real hardware and synthetic logging tests unmistakably. Never generate a pretty
video and call it policy evidence.

Matched before/after evaluation uses fresh processes, identical held-out starts
and seeds, deterministic evaluation, and equal durations. Evaluation must not
silently include reset teleports in distance. A lap requires continuous route
progress through ordered checkpoints without reset; distance alone does not
establish a lap. The new integration inspector deliberately always reports
`driving_proven: false`: human and behavioral benchmark evidence is still needed.

## Small, serial hyperparameter study—not an open-ended sweep

This is a **post-refactor proposal**, not a configured or running W&B Sweep.

1. Freeze task/assets/behavior, optimizer defaults, evaluation split and metrics.
   First reproduce the unchanged baseline at a fixed **sample budget**.
2. Pilot learning rate only: baseline `3e-4`, then `1e-4` and `1e-3`, using one
   paired training seed. These are candidate values, not claims of optimality.
   Use one shared GPU serially; abort on invariant failure. Do not rank a run by
   a lucky maximum reward or its training track alone.
3. Compare driving metrics at the same collected and optimized sample counts;
   show wall-time curves separately. A run that times out before its sample
   budget is censored/failed-budget, not a shorter successful trial.
4. Confirm baseline and selected candidate with at least three paired training
   seeds and the frozen validation set. Report each seed and spread; three
   seeds are exploratory evidence, not strong statistical certainty. Touch the
   final test set only after selection; report extra tuning attempts too.
5. Only then test a second variable, such as entropy coefficient. Preserve rollout
   size while studying other variables; changing `num_envs × n_steps` changes the
   data/optimization regime. Multi-car samples are correlated, not extra worlds.

Use separate W&B runs with resolved configs and local artifacts initially.
Native Sweep agents are **not compatible with our fresh-run identity contract
yet**: inherited run IDs/resume controls are intentionally rejected. Ticket T09
adds/tests an explicit integration before publishing any executable sweep YAML.
This proposal does not authorize a sweep or any additional training allocation.

## Efficiency accounting

Record hardware/driver/runtime, physics dt, decimation, car/world counts,
render resolution/cadence, video cadence, train/eval seeds, collected samples,
optimized rollout samples, control steps and simulated seconds. PPO's existing
`ppo_updates` is an **epoch counter**, not minibatch optimizer steps.

Keep timers for setup, simulator startup, collection, optimization, training
callbacks/video, evaluation, transfer, and total allocated instance lifetime.
Report samples/s, simulated seconds/s, time and samples to the frozen success
criterion, and allocated GPU-hours. Price × allocated hours is an estimate, not
an invoice; include storage/other provider charges if present. No criterion met
within budget means “not reached by budget,” not infinite/zero efficiency.
Reward-scale/dt changes require a new comparison identity.

## Candidate extensions and prerequisites

| Direction | First testable hypothesis | Prerequisites / important confounder |
| --- | --- | --- |
| Multi-agent transfer and communication | A specified communication channel improves held-out collision-free completion over the same policy without messages | Calibrated single-car transfer, synchronized logs, latency/loss ablation, no privileged simulator-only information |
| Real-world training | Equal-budget real fine-tuning improves obstacle avoidance over zero-shot transfer | Emergency stop, bounded speed/area, human supervision, reset/intervention accounting, track permissions; no autonomous real training now |
| Self-play | An opponent pool improves robustness to unseen partners/opponents | Stable multi-agent stepping/reset ownership, opponent versioning, held-out opponents, explicit cooperative/competitive objective; revisit learner only then |
| Tracks and traffic rules | A direction cue reduces wrong-way intersection exits | Route-ground-truth metric, unambiguous yielding/stop semantics, controlled track-color ablation and real validation; separate terrain study |
| Scaling | The required control deadline holds at increasing car counts | Measure latency percentiles, jitter/loss, synchronization and control failures independently of learning quality; simulator FPS is not radio capacity |
| Robustness/control | A calibrated low-speed action map reduces dead-zone stalls without extra oscillation | Measured actuator/velocity/battery curves and step responses; compare to the existing threshold under identical conditions |

Reset-agent awareness belongs in safety/coordination acceptance, not just another
reward term. CAD attachment repairs need mechanical verification separately;
passing Python tests cannot validate the chassis. Prioritize the currently
observed bottleneck rather than beginning all six research directions at once.

## Compute policy and current stop state

The temporary `astra-validate-0926` instance was deleted; Brev returned no
workspaces at **2026-09-27 03:37:44 UTC** (September 26 locally). Only a host probe
ran; no training or source upload occurred. The closure receipt is local under
`outputs/cloud-validation/20260926/session.json`. This is not an invoice audit.

All current work is local. See [operations runbook](operations.md) for prepared scripts.
A future go starts with one attended, bounded validation session—not sweeps.
Recheck price/inventory, stop on the first failed gate, preserve and verify all
outputs including failures, delete the exact instance, and confirm disappearance.
An initial two-hour session target would include setup and backup; it is not an
enforced spending cap and is not authorization to launch now.

Unattended allocation stays blocked until an independent, tested control-plane
expiry can stop/delete the instance even if this agent, laptop or SSH dies.
Guest shutdown, a detached shell and `timeout` do not establish that guarantee.
NVIDIA explicitly warns that an instance can remain running after a desktop or
SSH interruption: [AI Workbench/Brev lifecycle guidance](https://docs.nvidia.com/ai-workbench/user-guide/latest/locations/brev.html).
Stopping can retain storage charges; verify provider lifecycle and final state:
[Brev instance concepts](https://docs.nvidia.com/brev/concepts/gpu-instances).

The W&B key previously pasted into chat should be rotated before a future online
session; do not copy it into this repository, commands, logs or experiment records.
