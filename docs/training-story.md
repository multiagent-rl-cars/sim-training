# What this project actually trains

> Historical code analysis: September 22, 2026, before the `astra-cleanup`
> behavior corrections. It is useful for understanding the inherited system,
> not a specification of the current working tree. See [current architecture](architecture.md),
> [current workflow](../README.md) and the [experiment registry](../experiments/README.md).
> This notice was added on September 27; the original analysis below is preserved.

This describes the `multiagent-training-env` branch, recovered from upstream
commit `7e60abb`, and the small generated-asset bring-up. It describes executable
code, not claims about historical trained policies. The original USD model and
validated historical checkpoints are still unavailable.

## One shared policy, several cars

`Multiagent-Car-Training-v0` registers `MultiAgentEnv`, a `ManagerBasedRLEnv`.
Each Isaac Lab environment slot contains one car, but these cars use the same
world-space track and observe one another. Here `--num_envs 4` means four cars in
one arena, not four independent race simulations. SB3 sees four vector slots;
one PPO policy controls all four and pools their experience into one rollout.
Each slot has its own observation, action, reward, and done flag.

This is parameter-sharing PPO with local observations. It is not MAPPO, a
centralized joint-state critic, learned communication, one policy per car, or
distributed multi-GPU training. The value network sees the same local observation
as the actor. The other cars' behavior changes whenever the shared weights change,
so the learning environment is coupled and experience is correlated.

The unused-looking `tasks/direct/car_training_marl` template is a separate code
path. The operational task examined here is the manager-based registration.

## What an individual car sees

The observation dictionary contains three entries:

| Key | Per-car shape | Meaning and source |
| --- | --- | --- |
| `track_image` | 1 x 48 x 48 | Binary, heading-aligned crop around the car from a static overhead track image. |
| `obstacle_image` | 4 x 48 x 48 | Four historical heading-aligned crops of a synthetic map of the cars. A 96 x 96-pixel neighborhood is downsampled to 48 x 48. |
| `robot_vel` | 2 | Simulator XY center-of-mass velocity, rotated by `-heading - pi/2` and divided by 0.8 m/s. |

Black is track/car occupancy; white is free background. The values returned by
the environment are floats 0 or 1. Despite the names, the obstacle input is not a
live camera segmentation. It is drawn from exact simulator car positions and
headings as oriented trapezoids. It currently includes the observing car too.

The track image is captured once with the cars hidden, then reused. The ordinary
inherited workflow can instead load `multiagent_static_track_image_isolated.pt`.
Our test captures the generated geometry directly and does not reuse a map from
an unrelated original USD. Runtime positions and headings choose and rotate each
crop; the policy does not directly receive its world XY position, global map,
heading angle, car ID, destination, lap progress, battery state, steering feedback,
or wheel encoder readings. It has no recurrent hidden state. Its short memory is
the four image channels, intended to reveal relative motion.

The nominal original map is 2048 x 2048 over 40 x 40 m: 51.2 pixels/m. That makes
the track crop about 0.94 m across and the other-car crop about 1.88 m across.
The fixture uses 512 x 512 over 10 x 10 m, preserving approximately the same pixel
scale while reducing memory. These scales depend on correct camera calibration;
they are assumptions, not automatically inferred from the USD.

**At the network boundary**, the observation specification says images are in
0..255. Isaac Lab's SB3 wrapper casts them to uint8, then SB3's normal image
preprocessing divides by 255. Consequently the binary white value becomes 1/255,
not 1. The final smoke-test result records the measured network input ranges.
The saved `policy_inputs.png` displays the environment's binary masks at visible
contrast: leftmost is track, followed by the four car-history channels. It is not
an RGB image fed to the actor, and its display brightness is not the CNN's numeric
input scale.

Source: `multiagent_training_env_cfg.py:ObservationsCfg`,
`mdp/observation.py:aligned_track_neighborhood_image`, `generate_agents_map`,
`aligned_other_agents_image`, `aligned_robot_vel`; upstream
`.isaaclab/source/isaaclab_rl/isaaclab_rl/sb3.py:_process_spaces`.

## The network

The `train_combined_res.py` launcher explicitly selects `CustomMultiInputExtractor`:

1. Concatenate the track channel and four car-history channels into a 5 x 48 x 48 tensor.
2. Convolution blocks use 32, then 64, then 64 channels, with downsampling,
   BatchNorm, ReLU, and a residual block at each stage.
3. Flatten the resulting 64 x 6 x 6 tensor (2304 values), then project to 256 image features.
4. Project the two velocity values to 64 features.
5. Fuse the 256 image and 64 velocity features into a 256-dimensional feature vector.
6. Separate actor and value heads use the YAML's two 64-unit Tanh layers.
   The actor parameterizes a continuous two-action distribution; the critic
   predicts a scalar expected return. Training also learns the exploration parameters.

`train.py` and `train_mlp.py` do not explicitly select this residual extractor.
They use SB3's default `MultiInputPolicy` feature extraction. Their filenames do
not prove that images are processed by an MLP; the wrapper marks these inputs as
images, allowing SB3's default per-image CNNs. Always record the launcher as well
as the YAML. The bring-up test uses the residual extractor but smaller 32/32 heads.

## What the two actions do

For policy outputs `(a_throttle, a_steer)`, the action term applies:

```
wheel_speed = 10 + 20 * (tanh(a_throttle) + 1) / 2   # rear wheel rad/s
steering    = 0.6 * tanh(a_steer)                    # radians
```

Both rear wheel joints receive the same speed target; both front steering joints
receive the same angle target. The file defines an Ackermann conversion helper,
but `apply_actions` does not use it. These are physics joint targets, not radio
PWM commands or a calibrated mapping to a physical transmitter. The driving task
cannot command stop or reverse: its asymptotic throttle range is 10..30 rad/s.
A zero policy action means 20 rad/s forward, not zero throttle. The distinct
`Reset-Agent` task uses a different throttle range; it is not automatically part
of multi-agent driving.

Default inherited physics: dt = 0.05 s, decimation = 2, so decisions arrive every
0.1 s (10 Hz), with each action held across two physics steps. The fixture uses
120 Hz physics and decimation 4 (30 Hz decisions) for its uncalibrated small car.
The test therefore does not reproduce the original dynamics or policy timing.

## Rewards and resets

Before Isaac Lab multiplies by control-step duration, the configured reward is:

```
3.0  * line-following
+1.5 * speed
+0.2 * alive
+0.75 * steering-term
+1.0 * collision-term
```

- Line-following: `1 - distance_to_nearest_track_contour_pixel / 40`, without a
  lower clamp. It is set to zero if any black pixel appears in the car-history input.
  This measures distance to an extracted line contour, not ordered lap progress.
- Speed: magnitude of the two velocity components divided by 0.8 m/s. It rewards
  speed in either direction and is not clipped at one.
- Alive: constant one before weighting.
- Steering: `-abs(steering)/0.6`, also set to zero if the car-history input has a
  black pixel. Despite a comment saying it becomes a reward, the code sets it to zero.
- Collision: minus one if any other car center is within 20 image pixels. At the
  nominal scale that is about 0.39 m. It uses center distances, not chassis contact forces.

A slot terminates if its central 24 x 24 track crop has no black track pixel or
another car center is within 20 pixels. The config declares a 150-second episode
length but does not enable `mdp.time_out`; that number alone does not enforce a
150-second cutoff. Steering-jerk and contact-force termination terms are commented out.

Reset samples a contour point, points the car along a nearby contour tangent,
and randomly reverses direction with 50% probability. Cars retain a track-group
assignment based on slot index. Spawn rejection is grouped by track ID rather
than a global all-track clearance check. The inherited reset writes root pose and
root velocity, but does not explicitly restore wheel/steering joint states.
Our bring-up fixes initial/reset observations, reuses one identical rendered map
for all cars, uses external contours so a painted loop's inner and outer edges
aren't assigned as separate tracks, and uses the same reset settings at startup
and subsequent resets. Fixture resets use z=0.11 m and 1.5 m spawn clearance.

## How PPO learns

For each control step, the shared actor samples an action per car. The simulator
advances, returns observations and rewards, and resets terminated slots. PPO stores
those transitions and the actor's log probabilities and critic values. It computes
bootstrapped returns and generalized advantage estimates, then updates the policy
with PPO's clipped objective and the critic with a value loss. An entropy term
encourages exploration. No human action labels or prerecorded expert trajectories
are required by this path.

The inherited YAML specifies seed 42, 1,000,000 target agent-timesteps, 1024 rollout
steps per slot, batch size 256, 10 optimization epochs, learning rate 3e-4, gamma
0.99, GAE lambda 0.95, clip range 0.2, entropy coefficient 0.01, value coefficient
1.0, and max gradient norm 1.0. Generalized state-dependent exploration is enabled
and resampled every four rollout steps.

With four cars, one inherited PPO rollout contains 4096 agent-transitions.
`--max_iterations 2` requests 2 x 1024 x 4 = 8192 agent-timesteps. This is not two
physics steps or two episodes. SB3 finishes whole rollout batches and may exceed
a requested timestep target. Saving a checkpoint proves serialization, not
convergence or a validated continuation protocol.

## What to fix before serious training

1. Exclude the ego car from the other-car observation. Currently its own black
   trapezoid can always trigger obstacle proximity, switching off line-following
   reward and steering penalty. The smoke test intentionally retains this behavior.
2. Update image history once per simulator control step. Both reward functions
   call the stateful observation function, as does observation collection. Their
   counters measure calls rather than time; the nominal 0.3-second spacing can be
   wrong. Floating-point truncation and the counter's update order add ambiguity.
3. Align image bounds and normalization (0/1 versus 0/255), then add a test at the
   actual network input. Do not change this silently when loading old checkpoints.
4. Define the intended objective: follow a particular route/direction, yield or
   pass safely, make progress, and optionally stop. The current reward mainly
   encourages speed and survival; it has no route, right-of-way, or lap-progress goal.
5. Audit collision filtering, all-track reset clearance, joint-state reset,
   timeouts, sensor timing, and action calibration. Fixture runs explicitly disable
   cross-slot collision filtering; the original config defaults require review.
6. Establish held-out evaluation: unseen layouts, traffic density, seeds and
   dynamics, reporting completion/progress, collisions per distance, off-track
   rate and intervention rate. Then validate the real camera/pose/velocity pipeline.

Physical transfer would require replacing exact simulator poses and velocities
with measured vision estimates, matching the image preprocessing and timing,
calibrating actuation, and testing delays, tracking errors, battery variation,
friction, and geometry. None of that is established by this bring-up.

## Verified four-car bring-up

On the Brev A6000, the final bounded run completed 32 control steps with zero
resets, then 128 PPO agent-timesteps using the custom residual extractor. The
policy weights changed and remained finite. Non-reset path lengths for the four
cars were 0.252, 0.251, 0.553, and 0.389 m. These are short fixture runs, not trained
traffic behavior or a convergence result. The 128 transitions come from one
32-step rollout across four cars.

Measured observation shapes were `[4,1,48,48]`, `[4,4,48,48]`, and `[4,2]`.
Both image inputs reached the network in `[0, 0.0039215689]`, confirming the scaling
issue. At the final checked control step, line-following reward was zero for all
four cars. The saved input panel shows ego occupancy in the other-car channels;
the disabled line-following reward is consistent with the source-level bug.

Fixture overrides: 10m arena, 512px images, 120Hz physics, 30Hz control,
conservative uncalibrated actuators, cross-slot collisions enabled, and a circular
line. PPO uses 32-step rollouts, batch size 32, one epoch, 32/32 heads, and default
SB3 exploration rather than the inherited YAML's full training configuration.

To reproduce after setup and asset generation:

```bash
export CAR_TRAINING_ASSET_ROOT="$PWD/generated_assets"
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/isaac.sh scripts/smoke_test.py \
  --task Multiagent-Car-Training-v0 --num_envs 4 \
  --headless --enable_cameras --fixture --train --output outputs/multiagent
```

Artifacts are in `outputs/multiagent/` locally and under
`/home/shadeform/sim-training/outputs/multiagent/` on Brev. `camera.png` is an actual
render; `policy_inputs.png` is the actual reset observation for one car displayed
at visible contrast; `result.json` records the checks and input ranges;
`fixture_ppo.zip` is the resulting short-run checkpoint.
