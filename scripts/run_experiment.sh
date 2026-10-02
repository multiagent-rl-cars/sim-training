#!/usr/bin/env bash
# Linux RTX host: a bounded baseline plus matched before/after evaluation.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_DIR="${1:?Internal worker requires RUN_DIR and a generated recipe from config.yaml}"
CONFIG="${2:?Internal worker requires an explicit generated recipe; use uv run rc-car}"
if (( $# > 3 )) || [[ -n "${3:-}" && "${3:-}" != --externally-supervised ]]; then
    echo 'Internal worker: run_experiment.sh RUN_DIR GENERATED_RECIPE [--externally-supervised]' >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Choose a new run directory; existing output will not be overwritten: $RUN_DIR" >&2
    exit 1
fi
# An outer GNU timeout must own the process group. Nested timeout commands
# would create new groups and allow a simulator to outlive the outer deadline.
PHASE_COMMAND=(./scripts/isaac.sh)
if [[ "${3:-}" != --externally-supervised ]]; then
    PHASE_COMMAND=(timeout -k 20 1800 "${PHASE_COMMAND[@]}")
fi
# Reject recipe errors before starting the simulator or creating a run directory.
./scripts/isaac.sh -m rc_car.cli validate-config "$CONFIG"
# No Kit/GPU initialization and no interactive login. Online recipes must have
# reachable tracking before training; offline is a deliberate user choice.
./scripts/isaac.sh -m rc_car.experiments.online_check --config "$CONFIG"
: "${CAR_TRAINING_ASSET_ROOT:=$REPO_ROOT/generated_assets}"
export CAR_TRAINING_ASSET_ROOT PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ./scripts/isaac.sh scripts/capture_provenance.py --archive "$RUN_DIR/source_snapshot.tar.gz"
elif [[ ! -f outputs/source-provenance.json ]]; then
    echo 'Capture source provenance in the Git checkout before copying to this host.' >&2
    exit 1
fi
for phase in train start end; do
    echo "Running $phase; log: $RUN_DIR/$phase.log"
    "${PHASE_COMMAND[@]}" scripts/training_experiment.py \
        --config "$CONFIG" --mode "$phase" --run-dir "$RUN_DIR" \
        --headless --enable_cameras > "$RUN_DIR/$phase.log" 2>&1
done
./scripts/isaac.sh scripts/compare_evaluations.py "$RUN_DIR" > "$RUN_DIR/comparison.log" 2>&1
if [[ -f "$RUN_DIR/wandb_run.json" ]]; then
    if grep -q '"mode": "offline"' "$RUN_DIR/wandb_run.json"; then
        echo 'Offline run retained locally. Sync its W&B directory before uploading evaluation artifacts.'
    else
        ./scripts/isaac.sh scripts/upload_experiment.py "$RUN_DIR" > "$RUN_DIR/upload.log" 2>&1
    fi
fi
echo "Done: $RUN_DIR/start-vs-end.mp4"
