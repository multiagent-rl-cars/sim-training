#!/usr/bin/env bash
# Always launch from the repo root with the explicit Linux RTX dependency set.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Isaac execution requires Linux x86_64; use uv run rc-car for local checks.' >&2
    exit 2
fi
# Setup owns synchronization. A phase must not mutate the shared environment
# after the supervisor has verified its installed runtime and source identity.
exec uv run --locked --no-sync --no-group cpu --extra isaac python "$@"
