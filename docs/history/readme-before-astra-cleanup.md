# Archived README — before astra-cleanup

Snapshot of the inherited README before the 2026-09-25 refactor. Commands, branch
names, and relative links below describe that old root-level document; they are
not current instructions. See [the current README](../../README.md).

Historical commands below are unsupported. The public workflow is now
`uv run rc-car`, configured by the one root `config.yaml`.

# RC car simulation training

This checkout is on `multiagent-training-env`. The Brev A6000 test passed with
four cars, 32 control steps, and 128 PPO agent-timesteps using generated assets.
See [the full training story](docs/training-story.md) for policy inputs, actions,
rewards, architecture, and learning issues; [bring-up evidence](docs/bringup.md)
also preserves the earlier single-car result on `main`.

## Overview

Isaac Lab environments and Stable-Baselines3 training for Ackermann-steered RC
cars. Project code stays separate from the pinned upstream Isaac Lab checkout.

**Key Features:**

- `Isolation` Works outside the core Isaac Lab repository, ensuring development efforts remain self-contained.
- `Flexibility` This repo is set up to allow for the code to be run as an extension in Omniverse.


## Installation

Use **uv** for this repository. It is already supported by Isaac Lab's installation
workflow, and this project is a Python extension with no existing Conda dependency
specification. Pixi is useful when a project needs Conda-managed native libraries;
that does not remove Isaac Sim's separate GPU, driver, and operating-system requirements.

### Local development (including Apple Silicon Macs)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then
run these commands from this repository:

```bash
uv sync --locked
uv run --locked pre-commit validate-config
uv run --locked python -m compileall -q scripts source/car_training
```

The root `pyproject.toml` and `uv.lock` manage Python 3.11 development tools in
`.venv`. The actual extension remains in `source/car_training`. The local environment
does not install Isaac Sim, CUDA, PyTorch, or the extension: importing the extension
starts importing Isaac modules. Syntax checks and editing work locally; simulation,
task listing, and training need the server environment below. Syntax checks do not
validate imports, assets, or simulator behavior.

### NVIDIA server / workstation

This code uses the full Isaac Sim runtime, including camera rendering. Apple Silicon
and macOS cannot run that runtime; `--headless` does not remove its NVIDIA RTX
requirement. Use a compatible Linux NVIDIA RTX server/workstation (Windows is also
supported upstream). Check the [Isaac Sim hardware and driver requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
including the compatibility checker, before provisioning a machine. A CUDA-capable
GPU alone is insufficient.

### Reproducible Linux runtime (Brev or a local RTX desktop)

`environments/isaac/pyproject.toml` and its separate `uv.lock` pin Python 3.11,
Isaac Sim 5.1.0, Isaac Lab v2.3.0 at a resolved Git commit, CUDA 12.8 PyTorch
2.7.0, SB3 2.7.0 and transitive dependencies. This recovery baseline follows the
[versioned upstream guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html).
The runtime environment targets **Linux x86_64**; use Ubuntu on the desktops for
the same setup. The Mac development environment is deliberately separate.

```bash
# Ubuntu system prerequisite (in addition to a working NVIDIA driver):
sudo apt-get install libglu1-mesa
./scripts/setup_isaac.sh
# Isaac Sim requires acceptance of NVIDIA's EULA on first launch.
# After reviewing and accepting it, unattended runs can use:
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/isaac.sh scripts/list_envs.py
```

The wrapper always runs from the repository root and selects the locked Linux
environment. No global Python environment, personal Git identity, shell startup
files are required. The setup script creates an ignored `.isaaclab/` checkout
at the verified upstream commit (upstream wheels omit required config files). The old `git-itai.sh`,
`git-sasan.sh`, and `git-will.sh` identity-switching scripts have been removed.
Use your normal Git identity configuration.

### Running without the original USD files

Generate a self-contained six-joint car and 12 identical circular-track USDs:

```bash
uv run --locked --group assets python scripts/generate_test_assets.py
export CAR_TRAINING_ASSET_ROOT="$PWD/generated_assets"
./scripts/isaac.sh scripts/smoke_test.py --task Multiagent-Car-Training-v0 --num_envs 4 --headless --enable_cameras --fixture --train --output outputs/multiagent
```

The generator works on the Mac as well as Linux. It refuses to overwrite a
nonempty output directory. The fixtures need no external meshes or textures;
`fixture.json` records their dimensions and limitations. They match the inherited
wheel joint names and chassis sensor path. They are **uncalibrated test assets**,
not a replacement for the physical car model or evidence of useful driving skill.

The bounded smoke test runs the actual `Multiagent-Car-Training-v0` task, checks
finite observations/rewards/poses and movement excluding reset jumps, then runs
at least 64 PPO agent-timesteps (rounded to a full rollout: 128 with four cars).
It saves a checkpoint, `result.json`, camera render, and a policy-input panel in
`outputs/multiagent/`. The panel shows track then four car-history channels.
Fixture-specific camera, timestep, and actuator overrides stay in the smoke test.
Use `--num_envs 2` for the initial check on a 16 GB desktop. The four-car test
is verified only on the A6000.

For the original model, set `CAR_TRAINING_ASSET_ROOT` to the recovered asset bundle
(or unset it to use the package's `assets/` directory). It must contain
`robot_assets/leatherback_simple_better.usd` and
`scene_assets/tracks_thickness_random/track_0.usd` through `track_11.usd`, with all
referenced meshes/textures. The alternate TTR configuration references
`robot_assets/ttr_nano_v1.usd`. The generator only supplies the active Leatherback
interface. Do not run the original training configuration on fixture geometry
without the fixture overrides.

**Historical single-car commands from `main`; not revalidated on this branch:**

```bash
./scripts/isaac.sh scripts/zero_agent.py --task Basic-Car-Training-v0 --num_envs 1 --headless --enable_cameras
./scripts/isaac.sh scripts/sb3/train.py --task Basic-Car-Training-v0 --num_envs 1 --headless --enable_cameras --max_iterations 2
```

### Hardware and moving between machines

The same runtime lock is intended for the Brev A6000 and Linux RTX desktops.
For the 4070 Ti Super 16 GB / Ryzen 7900X / 32 GB RAM and the Intel i7-14700F /
32 GB RAM machine, begin with **one environment, headless, cameras enabled**.
Check the second machine's exact GPU and VRAM with `nvidia-smi`; do not rely on
the reported “RTX 5070 16 GB” name. Compatibility and usable environment counts
must be verified on each machine; 16 GB does not imply the A6000's capacity.
Budget tens of GB of disk for simulator packages, uv cache, shaders, and runs.

Copy source, both lockfiles, and run artifacts between machines. Recreate the
environment with `./scripts/setup_isaac.sh`; do not copy
`.venv` directories. Keep `CAR_TRAINING_ASSET_ROOT` machine-specific. Save
`nvidia-smi`, `uv pip freeze --python environments/isaac/.venv/bin/python`, and
both lockfiles with substantive training results. See `docs/bringup.md` for the
actual tested machine and remaining limitations.

### Set up IDE

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code. This is related to IsaacSim and Omniverse syntax highlighting.

- To enable syntax highlighting for IsaacLab related modules, make sure to manually add the full path to `settings.json` file.


### Setup as Omniverse Extension

An example UI extension is provided that will load upon enabling your extension defined in `source/car_training/car_training/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

There is a pre-commit template to automatically format your code.
To install pre-commit:

```bash
uv sync --locked
```

Then you can run pre-commit with:

```bash
uv run --locked pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/car_training"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

## Bounded training with before/after video

The inherited baseline intentionally retains the learning issues documented in
[the training story](docs/training-story.md). It is a comparison point, not a
recommended long-running training objective. The implementation plan is in
[semester-training-plan.md](docs/semester-training-plan.md).

On a configured Linux GPU host, with generated assets available and the Isaac
EULA accepted, use a fresh output directory:

```bash
./scripts/run_experiment.sh outputs/experiments/my-baseline
```

The default [experiment config](configs/experiments/multiagent_baseline.json)
requests 16,384 agent-timesteps across four cars. The runner saves initial/final
checkpoints, structured JSONL events, CSV/TensorBoard metrics, config/source/asset
provenance, and 15-second deterministic start/end videos. Evaluation runs in
separate processes with the same seed, then verifies the initial states before
creating `start-vs-end.mp4` and a metrics figure. It does not upload to W&B.

To inspect logs on the GPU host:

```bash
uv run --locked --project environments/isaac tensorboard \
  --logdir outputs/experiments --host 127.0.0.1 --port 6006
```

Use SSH port forwarding to view TensorBoard locally. A source-only deployment
without Git must include `outputs/source-provenance.json` created by
`scripts/capture_provenance.py` in the real Git checkout; preserve its source
archive alongside the run. Model checkpoint continuation is not yet validated.
