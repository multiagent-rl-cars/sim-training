# Experiment registry

This is a human-written research notebook: ordinary Markdown, edited by hand.
For a new investigation, create a descriptively named folder with a `README.md`.
No JSON record, registration script, fixed schema or W&B run is required.
The optional [starting point](templates/report.md) is just a few headings; add
notes, tables, photos, failed attempts and follow-up questions as you work.

Large raw data, weights, photos and videos stay under ignored `outputs/` and a
verified backup or explicitly authorized artifact store. Link them from your
notes. Local files are not automatically shared with the team. W&B is a mirror,
not the only copy. Throwaway work goes in `scratch/`; useful research notes here
are intended for Git.

The frozen foundation now passes all four GPU integration gates in
[E001](E001-foundation-integration/README.md). The 900-second fixture baseline
[E006](E006-900s-fixture-baseline/README.md) reduced resets from66to0 and increased
travel in its fixed15-second evaluation. This is not a historical convergence,
held-out generalization or sweep result. Most items below remain
post-refactor work. Begin with the [work plan](../docs/work-plan.md) and
[reproduction contract](../docs/reproduction-contract.md).

| ID | Status | Single question | Gate |
| --- | --- | --- | --- |
| [E001](E001-foundation-integration/README.md) | Passed; backed up; instance deleted | Does a frozen foundation meet the declared end-to-end integration invariants? | Four modes passed independent checks; known behavior defects remain |
| E002 | Planned, source recovery blocked | Is the reported line-following protocol reproducible with its stated convergence criterion? | T06–08; “not comparable” if identity/criterion unresolved |
| [E003 image encoding](E003-policy-image-encoding/README.md) | Local pass; candidate upload approval pending | Does one isolated image-scale correction satisfy the explicit observation contract? | 453-test snapshot ready; four candidate GPU gates unexecuted; no learning-quality claim |
| E004 | Planned | Does changing only learning rate improve sample efficiency on the frozen task? | T09–10; pilot then paired-seed confirmation |
| E005 | Planned | How much whole-job overhead does the fixed recording configuration add? | T05, T07; same steps, hardware and settings, recording ablation clearly separated from driving evidence |
| [E006](E006-900s-fixture-baseline/README.md) | Rule passed; backed up; instance deleted | Does a 900-second frozen-fixture run reduce off-track events without less travel in the fixed matched evaluation? | 66→0 resets,98.773→197.148m; explicit deviations prevent a strict historical reproduction claim |

Historical reference, **not E001/E002**:
[baseline-002](../outputs/experiments/baseline-002/) has raw evaluation data,
[start/end video](../outputs/experiments/baseline-002/start-vs-end.mp4),
[comparison still](../outputs/experiments/baseline-002/comparison-frame.png) and
[metrics](../outputs/experiments/baseline-002/comparison.json). It used older code
and generated uncalibrated assets. Reward and distance rose, but failures rose
too. Do not relabel these artifacts as post-refactor or real-world demonstrations.

## What you write versus what the tools save

- **You write Markdown:** the question, what you changed, dated observations,
  links to runs/videos, what the evidence suggests and what to try next.
- **You edit `config.yaml`:** settings for the next training run.
- **The tools save JSON/JSONL:** exact run settings, identity/hash manifests,
  metrics and per-step events under the run output directory. Existing analysis
  and recovery tools expect these formats. Do not hand-edit recorded evidence;
  link it and explain corrections or interpretation in your notes.

One investigation should answer one main question. For a controlled comparison,
write the success criterion, controls and budget before running; for exploratory
work, say it is exploratory. Append dated notes rather than rewriting the
original expectation after seeing results. Include failures and uncertainty.
Link generated metadata instead of copying hashes and configuration into a
second hand-maintained form. For manual or real-hardware work without a generated
record, note the setup and measurements needed to understand the result.

The existing E001/E003/E006 `record.json` files are retained historical snapshots,
not a template or a requirement for new experiments. They are not rewritten to
reflect later changes; use their Markdown reports for subsequent interpretation.
