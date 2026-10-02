# Repository contract

Follow the parent workspace's ownership and sharing rules. This file adds
repository-specific implementation and review requirements.

## Simplicity is a correctness requirement

- One root `pyproject.toml`, one `uv.lock`, one `.venv` per checkout. Select CPU
  and simulator dependencies with uv options; do not create another environment
  manifest to avoid understanding dependency resolution.
- The public workflow is `uv run rc-car`, configured by `config.yaml`. Do not
  recommend `python -m`, direct Python scripts, or legacy argument-heavy training
  launchers as alternative user workflows. Internal worker commands are allowed.
- Keep active documentation consistent with that interface. Label dated command
  transcripts as historical and unsupported; never rewrite frozen run evidence.
- Remove unused template changelogs, placeholder documentation, duplicate
  manifests, and demo scaffolding after checking references. Do not remove
  functional conventions merely because they are small or unfamiliar: for
  example, `py.typed` declares packaged type information.
- Generated fixtures/caches/results are not authored assets. Keep generators
  and specifications in Git and preserve exact run assets with checksums.
  Do not blanket-ignore USD extensions: authored assets need versioning and an
  explicit delivery/provenance path (Git for small files, a pinned artifact/LFS
  workflow for large binaries). Ignoring output is not a backup strategy.
- Keep one editable training configuration. Historical numerical recipes belong
  in test fixtures, not a second live `configs/` tree. Recovery/sync SDK utilities
  belong in operational support; dated report builders belong with the specific
  experiment. Reusable scripts must not hard-code a particular session/date.
- Tests that assert specific settings must construct explicit test inputs;
  editable `config.yaml` values are user choices, not fixed test expectations.
- `scratch/` is ignored, disposable, and never an import/runtime/test dependency
  of committed code. Put one-off local work there. Keep reproducible runtime
  workers in `scripts/` and reusable library support under `src/`; do not restore
  a top-level `ops/` directory. Dated research evidence is not disposable scratch.
- Profiles contain self-contained execution parameters, not report prose or
  redundant run names. Put limitations/interpretation in experiment reports.
  Default to automatic output directories and W&B-generated display names;
  explicit investigation directories remain optional and never overwrite runs.
- Human experiment logs are free-form Markdown, with an optional short template.
  Never require hand-written JSON records or duplicate machine metadata by hand.
  Generated JSON/JSONL stays an internal run artifact for analysis/provenance;
  preserve historical evidence and link it from the notes. Review both the
  manual research workflow and the automated training workflow.

## Review the design, not just the diff

Every review must answer both: “Does this work?” and “Is this the simplest
supported interface?” Existing structure is not evidence that it is necessary.
Reviewers must flag duplicated sources of truth, unexplained path dependencies,
stale commands, and template leftovers even if the targeted tests pass.

For dependency or packaging changes, require evidence for:

- A clean checkout running the public CPU command without pre-existing caches,
  simulator source directories, or GPU packages as hidden prerequisites.
- CPU versus simulator dependency selection, platform guards, lock freshness,
  package resources, and consistent selection in nested worker launches.
- No unintended changes to validated simulator/library versions. If a packaging
  fix requires such a change, report the tradeoff rather than quietly migrating.
- Updated active docs and provenance capture; frozen snapshots remain untouched.

For each user-reported issue: fix the issue, inspect related instances, identify
the concrete review gap, and add a focused regression check when feasible.
Record a reusable rule here only when it prevents a class of mistakes; avoid a
growing incident diary. A review that only validates a requested local edit must
say so and cannot be presented as approval of the entire architecture.

CPU tests, lock resolution, and wheel inspection do not prove Isaac startup,
physics, rendering, training quality, or convergence. Report those boundaries.
