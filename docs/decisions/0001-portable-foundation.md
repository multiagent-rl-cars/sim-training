# ADR 0001 — Portable foundation before backend replacement

Date: 2026-09-25. Status: accepted for the first refactor stage.

## Decision

Use `ty` as the initial and only type-checking gate, as requested. Use Ruff for
formatting/linting, pytest/Hypothesis for behavioral contracts, and uv lockfiles
for separate local CPU and Linux Isaac environments. New portable code lives in
`src/rc_car`; legacy Isaac entry points remain compatible while migration proceeds.

Keep SB3 and Isaac as historical baselines. Evaluate skrl IPPO/MAPPO for cooperative
MARL or RLlib if opponent populations become the next primary experiment. Do not
implement multiple production backends during foundational extraction.

## Why

Importability, observation/action units, state ownership, and reproducibility
are prerequisites to comparing libraries. A framework migration cannot correct
unmeasured hardware dynamics or a mismatched actor observation contract.
Current model/kernel extraction can be tested against frozen historical code
on a Mac without consuming cloud GPU credits.

`ty` is pinned by uv.lock rather than assumed diagnostic-stable. Public inputs
still need runtime/unit/shape tests: type checking does not prove physical units,
tensor axes, reward meaning, or training correctness.

## Behavior versus structure

Model extraction preserves architecture and serialization. Numerical kernels
preserve legacy formulas. One separately identified fix makes velocity reads
non-mutating. Telemetry now aggregates whole rollouts and failure paths protect
checkpoint attempts. Record these differences as `astra-foundation-v1`; do not
claim trajectory parity with the archived baseline.

## Preservation

Pre-refactor source, recipes, docs, tests and both locks are archived locally at
`outputs/refactor-baseline-20260925/source-before-refactor.tar.gz`.

SHA-256: `4e894fb410d9453c5c94834e3fbb50aa3effb4001657863c151b9d352700db73`.

Its adjacent `source-provenance.json` records file hashes and the dirty worktree.
This is a source snapshot, not a backup of all generated assets or runs. Existing
output/handoff archives remain untouched. Nothing is pushed or shared by this ADR.

## Revisit when

- Local contract tests and bounded GPU integration establish a trustworthy baseline.
- A chosen research question needs centralized critics, opponent populations, or
  measured real-world sample efficiency beyond the current learner.
- Held-out real trajectories demonstrate an identified simulator limitation.

## Sources

- [ty configuration](https://docs.astral.sh/ty/configuration/)
- [Isaac Lab 2.3 learner interfaces](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [RLlib multi-agent environments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [Isaac Sim 5.1 unsupported-release notice](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
