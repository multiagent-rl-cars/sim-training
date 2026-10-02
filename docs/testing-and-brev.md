# Test boundaries

Updated September 27, 2026. Use [config.yaml and the canonical workflow](../README.md)
for everyday work. The [operations runbook](operations.md) owns runtime setup,
host preparation, tracking diagnostics, bounded jobs, backups and cleanup;
this page explains what each check does and does not establish.

## Local software checks

With `operation: check`, the workflow runs Ruff lint/format verification, **ty**
and the CPU pytest suite. Ordinary checks exclude explicitly marked GPU/network
integration. They exercise strict recipes, geometry and reward/action kernels,
adapter bodies, historical model/gradient compatibility, a real CPU PPO update,
checkpoint guards, telemetry, provenance, tracking and failure cleanup.

ty covers portable modules, tests and operations code—not the whole inherited
Isaac extension. Frozen historical fixtures are deliberately excluded from
formatting/type checks. A local pass does not prove remote CI has executed,
camera/physics correctness, complete reset/history semantics or driving quality.
The Mac is suitable for these checks, not full Isaac simulation.

## Tracking checks

| Gate | Establishes | Does not establish |
| --- | --- | --- |
| Mock SDK tests | Identity, configuration, allowlists, failure handling and verification logic | Authentication or service behavior |
| Real offline SDK check | Local scalar/video/artifact persistence, hashes and successful stream closure | Upload, account permissions or cross-machine portability |
| `tracking-check` | Configured online credential/project access, or explicit offline selection | Artifact upload permission or successful server ingestion |
| Synthetic online transport check | Server scalar/media/artifact evidence for clearly labeled test data | Simulator training or policy quality |
| Saved-run recovery verification | Original history, independently checked final summary, finished state, video metadata and exact artifact manifests | Remote byte redownload, exact training continuation or useful driving |

The binary-stream adapter targets pinned W&B 0.25.1 and must be revalidated on
upgrade. Remote artifact records may retain inaccessible GPU staging paths after
copying, so stock sync alone is insufficient. Keep originals immutable, verify
the selected copy, repair artifacts from local files and check the server.
A sync exit code or `.synced` marker is not a complete upload receipt.
All three September 27 saved runs now have successful server verification;
see the [recovery receipt](../outputs/wandb-upload-20260927/summary.json). This
was CPU-side recovery of earlier GPU results, not a new GPU run.

## GPU and behavior evidence

[E001](../experiments/E001-foundation-integration/README.md) passed four integration
gates on the frozen v2 snapshot: Basic smoke/rehearsal and Multi smoke/rehearsal.
Checks include motion, finite changed model weights, reload, matched initial
states, readable recordings and retained provenance. Integration is not a test
of safe coordination or all simulator scheduling semantics.

[E006](../experiments/E006-900s-fixture-baseline/README.md) tested a preregistered
behavior rule after 900 seconds of training: fewer off-track resets without less
reset-excluded travel in a fixed matched evaluation. It passed on generated
circular fixtures. It did not demonstrate a complete lap, held-out performance,
the previous team's convergence claim or real-hardware transfer.

[E003](../experiments/E003-policy-image-encoding/README.md) has a separately frozen
image-input correction with local tests; four candidate GPU gates remain pending.
The older videos do not validate newer source. Later tracking/workflow changes
are also outside the frozen snapshots. No full-branch GPU pass is claimed.

There is no validated convergence benchmark set yet. That needs a frozen task,
calibrated assets, lap/direction/collision criteria, held-out starts/tracks and
paired seeds. Keep collected samples separate from fully optimized samples:
the time callback can leave an unoptimized partial rollout.

## Cost and recovery are separate checks

A bounded process can stop training while an allocated instance keeps billing.
Current wrappers do not establish independent provider-enforced expiry. One
shared host, explicit budgets, verified result handoffs and observed deletion
remain required. Check live pricing and cloud state before any allocation;
historical session receipts are not current inventory.
