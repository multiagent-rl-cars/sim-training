# Recovery bring-up

> Historical snapshot: September 21–22, 2026. Host names, cloud state, commands
> and behavior descriptions below are evidence from that session, not current
> instructions. Do not resume or allocate from this page. Use the [current
> workflow](../README.md), [operations runbook](../ops/README.md) and
> [experiment registry](../experiments/README.md). This notice was added on
> September 27; the historical record below is preserved.

## Environment

- Tested host: Brev `surprised-emerald-hawk`, Ubuntu 22.04.5 x86_64.
- GPU: NVIDIA RTX A6000, 48 GB; driver 570.195.03 (CUDA driver API 12.8).
- Workspace: `/home/shadeform/sim-training`.
- Python 3.11.16, uv 0.12.17; Isaac Sim 5.1.0.0.
- Isaac Lab v2.3.0, commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`.
- Torch 2.7.0+cu128 / torchvision 0.22.0+cu128 / torchaudio 2.7.0+cu128.
- Stable-Baselines3 2.7.0.

The development and simulator environments have separate uv lockfiles. Run
`scripts/setup_isaac.sh` on Linux before `scripts/isaac.sh`. Upstream Isaac Lab
requires its repository layout and config/apps data, so installation uses a
verified, ignored editable `.isaaclab/` checkout rather than upstream wheels.
The dependency lock includes the editable packages; the setup script verifies
their exact source commit and refuses a dirty upstream checkout.

`uv pip check` passes on the host. Three explicit dependency overrides match the
actual Isaac Sim Linux wheel because NVIDIA's index metadata reports different
versions of filelock, fsspec, and networkx. Setuptools is constrained below 81
because the upstream flatdict package requires pkg_resources. `libglu1-mesa`
was installed from Ubuntu's apt repository for the rendering runtime.

## Scope of the fixture

The car and tracks are generated from `scripts/generate_test_assets.py` using
OpenUSD, without downloaded meshes or original USDs. The car preserves the
six joint names and chassis prim expected by the inherited task. The 12 tracks
are deliberately identical circles. The fixture is not a digital twin and
must not be used to claim inherited-policy recovery or sim-to-real quality.

The smoke test applies conservative fixture-only physics and camera settings,
including a chassis spawn height above the floor and a conventional focal length
with a 10m-wide field of view.
The ordinary task defaults remain available for the original assets. Both
manager-based task configurations now support `CAR_TRAINING_ASSET_ROOT` and
resolve their default paths relative to the package rather than a fragile
repository parent count.

## Recovery fixes

- The standard SB3 launcher now passes the task name to the car environment;
  the action controller rejects its old default `Unknown`.
- The car environment preserves a caller-provided overhead camera configuration.
- Reset returns observations recomputed after track and obstacle initialization.
- Extension packaging discovers subpackages and includes agent YAML files.
- Test failures print their traceback and return a failing process status instead
  of allowing Kit shutdown to hide the exception behind an exit status of zero.

## Verified GPU run

On 2026-09-21 Pacific (2026-09-22 UTC), the one-environment fixture test passed:

- 32 control steps, finite observations/rewards/poses.
- Maximum XY displacement: 0.367177 m; zero resets during the motion check.
- 64 PPO timesteps completed; policy weights changed and stayed finite.
- A saved SB3 checkpoint, actual camera renders, PPO CSV metrics, package inventory,
  lockfile snapshot, and full launch log are in `outputs/smoke/` on both machines.
- The 64-step run reported about 19 environment steps/s after startup. This is a
  tiny smoke test, not a throughput benchmark or evidence of policy convergence.

Successful command (after setup and asset generation):

```bash
export CAR_TRAINING_ASSET_ROOT="$PWD/generated_assets"
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/isaac.sh scripts/smoke_test.py --headless --enable_cameras --fixture --train
```

Brev's API token became invalid during bring-up; its existing SSH alias continued
to work. `ssh surprised-emerald-hawk` accesses the same instance. Re-authenticate
with `brev login` if Brev commands report a malformed refresh token. No shell
startup files or account identities were changed. The instance remains running;
the bounded simulator/training process exits when finished.

## Local checks

- Python compilation and shell syntax checks pass.
- Pre-commit configuration validates.
- All 13 generated USD files reopen successfully.
- All six revolute joints reference existing bodies; chassis is an articulation root.
- No repository-local Git user identity was set. Removed only the three named
  identity-switching shell scripts; preserved copyright and historical attribution.

## Portability and remaining validation

Run the same setup and one-environment test on each free-access desktop before
increasing parallelism. These desktops have not been tested. Verify the reported
5070 model and VRAM with `nvidia-smi`. A simulator recovery smoke test does not
validate calibrated handling, useful policy learning, performance at scale,
reset-agent training, checkpoint resumption, or real-car deployment.
