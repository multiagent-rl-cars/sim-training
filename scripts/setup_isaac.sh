#!/usr/bin/env bash
# Install the pinned official Isaac Lab distribution into the one root environment.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Isaac setup requires Linux x86_64; no runtime installation was attempted.' >&2
    exit 2
fi
uv sync --locked --no-group cpu --extra isaac
uv pip check --python .venv/bin/python
.venv/bin/python -c 'from rc_car.provenance import inspect_isaaclab_runtime; inspect_isaaclab_runtime()'
