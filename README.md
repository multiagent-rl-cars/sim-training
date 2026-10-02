# RC-car research and simulation

An inspectable SB3 + Isaac research workspace on `astra-cleanup`. Keep one
configuration, attributable experiments and visible driving evidence. **Tests
and a higher reward curve are not proof of correct physics or useful driving.**

## One workflow

Edit [config.yaml](config.yaml), then run the same command from this checkout:

```bash
uv run rc-car
```

The installed entry point is `rc-car`; it takes no arguments. `uv run` installs
and uses the project's locked environment, including development and tracking
dependencies by default. No separate sync or dependency flags are needed.
`config.yaml` selects the operation, named profile, output directory, budgets and
W&B destination. Its default `operation: check` runs Ruff, format verification,
ty and CPU tests. It does not authenticate, allocate cloud compute or start Isaac.
Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

| `operation` | Purpose |
| --- | --- |
| `check` | Run the local quality suite; default and safe on the Mac |
| `plan` | Print the merged recipe and paths without running it |
| `tracking-check` | Check configured W&B authentication/project access; not upload proof |
| `train` | Run bounded training and matched evaluations on an already prepared Linux RTX host |
| `inspect` | Inspect a saved `job.input_run`; write evidence to a fresh `job.output` |
| `upload` | Attach completed outputs to the saved W&B run; historical offline metrics must already be synced |

Each profile contains its own training settings; the shared `tracking` section
chooses the destination and mode. Leave `job.output` unset for an automatically
allocated directory, or specify a fresh directory for a particular investigation.
W&B generates its own display name; the local record retains the name, run ID
and URL. Existing directories are never overwritten. Historical numerical JSON
recipes live only under `tests/fixtures/legacy_recipes` for comparison.
Limitations and interpretation belong in experiment reports, not the YAML.
No credentials belong in YAML, source, logs or artifacts.

`train` is fixture-only. Follow the [operations runbook](docs/operations.md) for the
locked runtime, approved source/assets, explicit NVIDIA EULA acceptance, host
credentials, backup and deletion. The workflow does **not** allocate or delete
instances. A process timeout is **not** a provider billing cap.
On the prepared GPU host, select the simulator dependencies with
`uv run --locked --no-group cpu --extra isaac rc-car`. Both modes use the same
root `pyproject.toml`, `uv.lock` and `.venv`; there is no second environment file.
CPU and Isaac options are mutually exclusive. Select Isaac on every GPU command:
plain `uv run rc-car` selects CPU dependencies and can switch Torch back to CPU.
Do not switch dependencies while another job uses this checkout.
`job.timeout_seconds` bounds the training supervisor; it is not a global deadline
for local checks, inspection or W&B operations.

## What has actually been demonstrated?

| Evidence | Result and boundary |
| --- | --- |
| [E001: foundation integration](experiments/E001-foundation-integration/README.md) | Four Basic/Multi GPU gates passed on the frozen 153-file `astra-foundation-v2` source |
| [E006: 900-second fixture baseline](experiments/E006-900s-fixture-baseline/README.md) | Matched 15-second evaluation: 66 → 0 off-track resets; combined reset-excluded travel 98.773 → 197.148 m |
| [E003: image-input correction](experiments/E003-policy-image-encoding/README.md) | Separate 168-file v3 candidate passed 453 local tests; its four GPU gates remain pending |

[Watch actual before/after driving](outputs/cloud-validation/20260927-run02/remote/outputs/baselines/basic-900s-01/experiment/start-vs-end.mp4).
The baseline used generated, uncalibrated circles: no full lap, held-out
generalization, historical convergence reproduction or real-car transfer was
demonstrated. Its result does not validate the newer image or tracking changes.
Later working-tree edits are not the frozen v2 or v3 snapshots.

The September 27 GPU session was deleted after a checksum-verified local backup
of 240 files. See the [session receipt](outputs/cloud-validation/20260927-run02/session.json).
Local paths under `outputs/` are ignored by Git; keep the verified backup.

## Live tracking and durable results

Online tracking sends metrics and selected training videos **from the training
host directly to W&B**, so the laptop is not an upload relay. Authentication and
project access are checked before simulator startup. The online workflow also
uploads final checkpoints, matched evaluation data and media. Offline mode is
an explicit choice, not a silent fallback when online authentication fails.

[W&B project](https://wandb.ai/2710-multiagent-rl/sim-training) ·
[15-minute baseline](https://wandb.ai/2710-multiagent-rl/sim-training/runs/434aa9828c404f999d1863242058f783) ·
[Basic rehearsal](https://wandb.ai/2710-multiagent-rl/sim-training/runs/2f170b9a2a1044829ef01ac686466951) ·
[Multi rehearsal](https://wandb.ai/2710-multiagent-rl/sim-training/runs/0261f7aa1729410ea072d7b31b3561f1).
All three saved runs have now been uploaded and server-verified against their
original history, final summary, media metadata and exact checkpoint/evaluation
artifact manifests. See the [upload receipt](outputs/wandb-upload-20260927/summary.json).
This CPU-side recovery is not a GPU test of the newer online launcher. Keep local
backups even after upload; W&B is not the only copy.

## Where to work

| Path | Responsibility |
| --- | --- |
| `config.yaml` | The editable workflow, profiles and tracking destination |
| `src/rc_car/` | Portable kernels/models/contracts, workflow, experiments and telemetry |
| `source/car_training/` | Inherited Isaac adapters, packaged with `rc_car`; not CPU-importable |
| `pyproject.toml` / `uv.lock` | One dependency definition/lock, with optional Isaac support |
| `tests/` | CPU contracts, compatibility and failure-path tests |
| `experiments/` | Single-hypothesis protocols/reports; [registry](experiments/README.md) |
| `scripts/` | Reproducible setup, internal workers, supervised jobs and verified handoffs |
| `outputs/` | Ignored raw runs, media, checkpoints and verified handoffs |
| `scratch/` | Ignored one-off local scripts, generated review pages and debugging work; never required by a fresh clone |

Read [architecture and scientific contracts](docs/architecture.md),
[test boundaries](docs/testing-and-brev.md), the
[portable-foundation decision](docs/decisions/0001-portable-foundation.md) and
[tickets](docs/tickets.md). [Refactor history](docs/refactor.md) preserves the
reasoning and review record. Dated bring-up/handoff documents are historical
evidence, not current launch instructions or current cloud state.

## Assets and framework code

`generated_assets/` is procedural test input generated by the versioned fixture
generator, not the original team's calibrated asset library. Generated files
stay outside Git; each run preserves the exact bytes and checksums in its asset
bundle. Generator source alone does not establish byte-for-byte determinism.
Small authored tracks/models belong in Git (prefer text USD where practical).
Large binary assets need a pinned artifact or LFS workflow with provenance and
delivery checks; a Git ignore rule is not a delivery or backup mechanism.

Isaac Lab comes from NVIDIA's pinned `isaaclab==2.3.0` bundle, which includes its
runtime configuration and assets. No `.isaaclab` checkout or editable framework
paths are required. The package consolidation is locally checked, but has not
yet been exercised on a GPU; the older driving videos do not validate it.
