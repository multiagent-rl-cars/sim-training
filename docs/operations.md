# Operations: local preparation and bounded GPU validation

The supported front door is now **one editable [config.yaml](../config.yaml)** and
`uv run rc-car`, with no command-line options. Its default `operation:
check` runs the local CPU gate; `plan` prints the selected profile without side
effects. `train` uses the existing bounded supervisor on an already prepared
Linux host and never allocates cloud hardware. `tracking-check` only checks
authentication/readable project access; it does not prove upload permission.
`inspect` and `upload` require the saved run's explicit `job.input_run`. Upload
uses that run's identity, never the newly selected training profile.

The commands below are lower-level operational interfaces retained for source
capsules, backups, and compatibility. They are not additional editable training
configurations. Historical JSON recipes and frozen bundles remain intact as
regression/reference evidence. New workflow jobs materialize their selected profile
under a fresh output path; never edit that generated receipt to tune a run.
Select the operation/profile/output in `config.yaml` **before** preparing a
remote source capsule: that file is source-hashed, so editing it after deployment
invalidates the recorded source identity. Preserve the four named baseline
profiles; add a distinct profile for tuning. `job.timeout_seconds` bounds the
training supervisor, not local inspection or W&B upload completion.

**Local-first scripts; this document does not authorize allocation.** The frozen
foundation completed all four real Linux/Isaac integration modes on 2026-09-27:
see [E001](../experiments/E001-foundation-integration/README.md). That validates
the recorded setup, not every future source snapshot, good driving or billing
expiry. Use one explicitly authorized instance and one job at a time. See the
[work plan](../docs/work-plan.md) and [tickets](../docs/tickets.md).

## Commands and boundaries

| Command | Does | Does not |
| --- | --- | --- |
| `prepare_session.py` | Runs local gate, checks source stability, builds/verifies source + fixture bundles | Access cloud, include Git credentials or existing training outputs |
| `bundle.py` | Verify/extract fresh bundles, verify fixture tree, package one result directory | Authenticate the sender, prove training success, upload or delete |
| `setup_runtime.sh` | Linux-only bounded preflight/locked install; optional explicit apt dependencies | Create instances, accept EULA for you, change NVIDIA drivers |
| `run_gpu_checks.sh` | One Linux job, shared repository lock, whole-job timeout, explicit offline/online W&B, logs/status | Stop/delete the instance, guarantee cleanup after supervisor SIGKILL or escaped sessions |
| `inspect_results.py` | Inspect saved tensors/motion/evaluation/media locally | Launch Isaac, load SB3 cloudpickle metadata, establish driving success |

Run from the repo root. Python commands below use the already-synced local
development environment; a fresh checkout can run `uv sync --locked` first.
The primary `uv run rc-car` command synchronizes automatically. On a prepared
GPU host, use `uv run --locked --no-group cpu --extra isaac rc-car` to select
Isaac/CUDA from the same root manifest and lock. Plain `uv run rc-car` selects
CPU dependencies and can replace CUDA Torch. Do not switch dependencies during
a job or run multiple environment synchronizations concurrently.

## 1. Prepare without paying for a GPU

Freeze edits first. The script refuses an existing output directory and rejects
source changes during the gate or between capture and bundling.

```sh
.venv/bin/python scripts/prepare_session.py --output outputs/gpu-sessions/next-session --plan
.venv/bin/python scripts/prepare_session.py --output outputs/gpu-sessions/next-session
(cd outputs/gpu-sessions/next-session && shasum -a 256 -c SHA256SUMS)
bash scripts/setup_runtime.sh --plan
bash scripts/run_gpu_checks.sh --mode smoke-basic --output outputs/validation/basic-01 \
  --fixture-manifest outputs/fixture-bundle-manifest.json --plan
```

Prices/types and account inventory must be checked again after a new go. The
prepared session contains the root lockfile, dirty source identity, local test
log and generated fixture bytes; it does not include NVIDIA runtime wheels,
secrets, old results or a provider lifecycle timer. One-off notebooks, publisher
scripts and provider checklists belong in ignored `scratch/`, not this workflow.
Fixture generation, if needed, remains an explicit separate operation; never
replace recovered original assets with generated ones under the same identity.

Inspect/extract locally into fresh destinations:

```sh
.venv/bin/python scripts/bundle.py extract \
  --archive outputs/gpu-sessions/next-session/source/bundle.tar.gz \
  --manifest outputs/gpu-sessions/next-session/source/manifest.json \
  --destination outputs/gpu-sessions/next-source
.venv/bin/python scripts/bundle.py extract \
  --archive outputs/gpu-sessions/next-session/fixtures/bundle.tar.gz \
  --manifest outputs/gpu-sessions/next-session/fixtures/manifest.json \
  --destination outputs/gpu-sessions/next-fixtures
```

The second destination contains `generated_assets/`. Extraction is no-overwrite,
checks archive hash/member allowlist, and rejects links/traversal/special members.
Checksums prove byte consistency, not provenance from a trusted sender. Only
inspect checkpoints/media from trusted research runs; parsers are not a hostile
file sandbox.

## 2. Future host handoff — only after a new go

Require a present operator who can delete if this agent/laptop fails, or an
independently tested control-plane expiry. No such expiry is currently verified.
Do not allocate unattended. Recheck inventory, exact type/price/disk, cleanup
authority and a session deadline; record returned instance ID. Only root/operator
allocates; workers share that single environment.

After the separate attended allocation, `brev refresh` supplies an SSH alias.
The following illustrates a fresh home-directory deployment for the name
`astra-next`; change the explicit name/path consistently for the actual session.
Do not copy into another Git checkout: transferred source identity must remain
the frozen local capture, not an unrelated parent repository's identity.

```sh
brev refresh
ssh astra-next 'mkdir "$HOME/rc-car-validation-next"'
rsync -a -- outputs/gpu-sessions/next-source/ astra-next:rc-car-validation-next/
rsync -a -- outputs/gpu-sessions/next-fixtures/ astra-next:rc-car-validation-next/
ssh astra-next 'mkdir "$HOME/rc-car-validation-next/outputs"'
rsync -a -- outputs/gpu-sessions/next-session/source-provenance.json \
  astra-next:rc-car-validation-next/outputs/source-provenance.json
rsync -a -- outputs/gpu-sessions/next-session/source/manifest.json \
  astra-next:rc-car-validation-next/outputs/source-bundle-manifest.json
rsync -a -- outputs/gpu-sessions/next-session/fixtures/manifest.json \
  astra-next:rc-car-validation-next/outputs/fixture-bundle-manifest.json
```

No `--delete`, home-directory copy, credentials, or remote Git identity required.
These are staged source/assets only. Before expensive installation inspect
`nvidia-smi`, disk space and Vulkan/driver capability. A6000 compatibility is not
established by the GPU name alone. Ensure `uv`, Git, GNU `timeout` and `flock`
are installed by a reviewed method; the setup script fails if they are absent.

## 3. Future Linux setup and serial checks

On that host, `cd "$HOME/rc-car-validation-next"`. Review NVIDIA's EULA before
explicitly supplying `OMNI_KIT_ACCEPT_EULA=YES`; the script will not accept it for
you. Setup allows 3,600 seconds by default; use a shorter explicit budget when
the whole session requires it. `--install-system-deps` allows only the listed
apt userspace packages (including Vulkan tools/ffmpeg), never driver replacement.
Keep setup stdout/stderr in a fresh `outputs/setup-01.log` even on failure. The
example enables shell noclobber; use `setup-02.log` for a retry, never overwrite
the preceding attempt's evidence.

```sh
# Only with the reviewed EULA accepted in your environment:
(set -C; bash scripts/setup_runtime.sh --install-system-deps --timeout-seconds 2700 > outputs/setup-01.log 2>&1)
.venv/bin/python scripts/bundle.py verify-tree --root . \
  --manifest outputs/source-bundle-manifest.json
bash scripts/run_gpu_checks.sh --mode smoke-basic --output outputs/validation/basic-01 \
  --fixture-manifest outputs/fixture-bundle-manifest.json --timeout-seconds 900
```

Inspect and back up after **each** gate. Do not blindly chain all four modes:

| Mode | Leaf output | Purpose |
| --- | --- | --- |
| `smoke-basic` | `smoke/` | 12 Basic fixture arenas, static obstacles enabled, real camera frames, finite signals, all-car movement, tiny PPO update |
| `rehearsal-basic` | `experiment/` | Basic obstacles disabled; resolved smoke recipe, initial/final weights and matched fresh-process evaluations |
| `smoke-multi` | `smoke/` | Four interacting fixture cars, motion/camera/optimizer integration |
| `rehearsal-multi` | `experiment/` | Multi smoke recipe, saved training/media/telemetry and matched evaluation |
| `experiment` (internal) | `experiment/` | Canonical YAML workflow's generated recipe; same lock, source/fixture checks and timeout |

Use a fresh job path for every mode/retry. Default whole-job limit is 900 seconds
plus 30-second kill grace, including preflight and evaluations, but excluding
installation and backup. It may terminate a cold-starting run: a timeout is a
failed/incomplete gate, not evidence that the simulator is broken. The 60-second
recipe budget bounds only training callbacks and can overrun until checked.

The outer GNU timeout owns the normal process group. Rehearsals use
`run_experiment.sh ... --externally-supervised` to avoid nested timeout groups
leaving Isaac alive after the outer deadline. Do not pass that flag without a
real external supervisor. Repository `flock` prevents concurrent setup/jobs in
this one checkout, not work launched from another checkout. Abrupt host failure,
SIGKILL and deliberately detached children are outside graceful cleanup guarantees.

Validation jobs default to offline W&B, ignoring an inherited `WANDB_MODE=online`
and removing inherited API keys. For live monitoring, explicitly opt a rehearsal
into online tracking on the host where the job will run:

```sh
bash scripts/run_gpu_checks.sh --mode rehearsal-basic \
  --tracking-mode online --output outputs/validation/basic-online-01 \
  --fixture-manifest outputs/fixture-bundle-manifest.json --timeout-seconds 900
```

Online tracking sends metrics and logged media/artifacts directly from that host
to the recipe's W&B entity/project. It needs a valid credential already supplied
securely on that host, for example through its secret environment; never put a key
in a recipe, source bundle, command-line argument, or checked-in file. Rotate the
previously exposed key before using online tracking. The launcher does not copy
credentials from the Mac or perform an interactive login. Smoke modes have no
W&B run and reject `--tracking-mode online`; use a rehearsal or the ordinary
experiment runner to test live tracking.

The launcher isolates all W&B data/cache/config/artifact directories under the
job and clears inherited run/resume controls in either mode. An online request
checks authentication and readable project access before GPU preflight or Kit;
successful initialization is still needed to prove run-write access. The check is
inside the whole-job timeout and never silently falls back to offline. On failure,
fix host authentication/access and use a fresh output path, or explicitly retry
with `--tracking-mode offline` and a fresh output path. `--plan` remains entirely
local and does not authenticate. Native W&B sweeps are not implemented.

These options apply to newly prepared source snapshots. Frozen historical bundles
and their offline results are unchanged; online launch is not a repair/sync command
for a previous offline run. Preserve local backups even when live tracking works.

## 4. Save failures as well as successes, then verify locally

On the host, inspect the leaf; a failed inspection must **not prevent backup**:

```sh
.venv/bin/python scripts/inspect_results.py \
  --run-dir outputs/validation/basic-01/smoke \
  --report outputs/validation/basic-01/evidence.json
.venv/bin/python scripts/bundle.py pack-results \
  --root outputs/validation/basic-01 --output outputs/backups/basic-01
```

Run these as separately inspected steps, not `inspect && backup`. `pack-results`
does not require training success. It excludes and records symlinks, rejects
credential-like filenames, and packages all selected regular job files. It is
not a general secret scanner. Copy setup logs/provenance/session notes too;
they live outside the selected job and need their own explicit backup.

Back on the Mac, choose fresh local locations:

```sh
mkdir -p outputs/cloud-backups
mkdir outputs/cloud-backups/next-basic-01
rsync -a -- astra-next:rc-car-validation-next/outputs/backups/basic-01/ \
  outputs/cloud-backups/next-basic-01/
.venv/bin/python scripts/bundle.py verify \
  --archive outputs/cloud-backups/next-basic-01/bundle.tar.gz \
  --manifest outputs/cloud-backups/next-basic-01/manifest.json
.venv/bin/python scripts/bundle.py extract \
  --archive outputs/cloud-backups/next-basic-01/bundle.tar.gz \
  --manifest outputs/cloud-backups/next-basic-01/manifest.json \
  --destination outputs/recovered/next-basic-01
.venv/bin/python scripts/inspect_results.py --run-dir outputs/recovered/next-basic-01/smoke \
  --report outputs/recovered/next-basic-01/local-evidence.json
```

Local inspection needs PyTorch (default CPU group), ffmpeg and ffprobe. Smoke checks
require decoded frame counts matching control steps; rehearsal checks require
matched initial states/seeds, per-car motion, finite changed tensors, updates and
videos. A preserved result can be a valid backup and a failed experiment.
Inspect actual clips/stills and failure traces before making driving claims.

Keep the entire job, including W&B staging/cache directories. Offline artifact
records may contain absolute paths: copied bytes do **not** prove cross-machine
`wandb sync` works. The portable sync test is T09; raw metrics/media/checkpoints
remain independently usable and must not depend on a successful upload.

For a copied rehearsal/baseline job, prepare an offline-only handoff inventory:

```sh
.venv/bin/python scripts/prepare_wandb_handoff.py \
  --backup-root outputs/recovered/next-rehearsal-01 \
  --manifest outputs/cloud-backups/next-rehearsal-01/manifest.json \
  --run-directory experiment
```

This verifies the supplied regular-file inventory, validates the public run
destination and stream filename, and lists model/evaluation files with hashes.
It does not parse or rewrite W&B's binary stream, read credentials, upload, or
prove media completeness. Partial/empty artifact groups are marked incomplete.
Keep the verified backup immutable; any later sync operates on a disposable copy.

The pinned SDK 0.25.1 has no artifact-path remapping option. A later synthetic
online test must exercise bounded explicit-run sync plus explicit re-upload of
verified local model/evaluation artifacts, with `mode="online"` and
`resume="must"`, then compare server manifest names, sizes and digests. A zero
sync exit or `.synced` marker alone is not proof that artifacts uploaded. Rotate
the previously exposed credential before that online step. No sweep/online
transport has been validated by the offline helper.

Finally recheck the exact saved instance name/ID, apply the approved deletion,
poll inventory until absent, and save that receipt. Do not claim deletion from
a finished process or a timeout. If backup is incomplete, report which bytes are
missing; an urgent user shutdown instruction takes precedence over retaining
compute indefinitely. No script here automatically creates or deletes instances.
