# 15-minute line-following attempt

> Historical attempt plan: September 23, 2026. This L4 bring-up did not produce
> the later A6000 baseline. Host/driver repair notes and launch commands below
> are archival, not current instructions. See the [current workflow](../README.md),
> [operations runbook](../ops/README.md) and the completed
> [E006 fixture experiment](../experiments/E006-900s-fixture-baseline/README.md).
> This notice was added on September 27; the original plan below is preserved.

The inherited final report (local extracted text, PDF pages 10 and 19) reports
approximately 15-minute convergence for line following with 12 environments on
an RTX 5070. It does not provide a numerical convergence threshold or a complete
historical run manifest. This attempt uses the current Basic task, inherited PPO
hyperparameters, 12 environments, and a 900-second training budget.

The available machine is Brev `my-instance`, NVIDIA L4 (24 GB), four vCPUs,
16 GB system RAM. Original car/track USD assets remain unavailable locally.
The generated circular-track fixture is repeated across environments; physics
overrides and 1/30-second control period differ from the historical experiment.
This is a timed fixture attempt, not an exact reproduction of that claim.

Configuration: `configs/experiments/line_following_15min.json`.
W&B destination: `2710-multiagent-rl/sim-training`. Credentials are supplied
outside source/configuration; never add API keys to run manifests or commits.

The timer starts immediately before `model.learn`, includes PPO optimization,
video encoding and tracking overhead, and excludes runtime setup, simulator
startup, model creation, and subsequent evaluation. The callback stops at the
next control step after 900 seconds; an in-progress optimization can overrun
the deadline, and the final partial rollout is not optimized. Both elapsed time
and actual agent timesteps are recorded. The large timestep cap is a safety cap,
not a promised amount of training.

Training captures environment 0 every two minutes, saves periodic checkpoints,
and sends scalar metrics and clips to W&B. Initial and final policies are then
evaluated in separate simulator processes with seed 123, matched initial states,
and 450 control steps. Full trajectories and reset-excluded distance are retained.
These short evaluations measure behavior changes, not generalization or lap
completion. Reward increase without reduced failures is not convergence evidence.

On the provisioned Linux host, with W&B login configured:

```sh
export PATH=/home/ubuntu/.local/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=YES
cd /home/ubuntu/sim-training
./scripts/run_experiment.sh outputs/experiments/line15-001 configs/experiments/line_following_15min.json
```

The run directory must be new. `wandb_run.json` records the live run URL.
The orchestration script uploads matched evaluation videos, charts, trajectories,
and metrics after training. No automatic instance shutdown is configured.

## Instance bring-up

The fresh Brev image had compute-only NVIDIA 595.91.07 packages. Initial checks
found no NVIDIA Vulkan ICD; installing matching graphics libraries and `libegl1`
restored Vulkan device enumeration. Isaac Sim 5.1 then crashed during RTX scene
initialization, resembling [upstream issue 619](https://github.com/isaac-sim/IsaacSim/issues/619).
The next repair installs Ubuntu's 580.178.04 server userspace and the matching
precompiled open kernel module for `6.8.0-1069-gcp`, followed by a reboot.
Other installed system dependencies include `libglu1-mesa`, `libxt6`, X11 client
libraries, Vulkan tools, and ffmpeg. The original failure logs are retained under
`/home/ubuntu/smoke-line12*.log`; installation logs remain in `/home/ubuntu`.
