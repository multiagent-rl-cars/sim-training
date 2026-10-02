#!/usr/bin/env bash
# One bounded validation job on an already-provisioned Linux RTX host.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=''
OUTPUT=''
FIXTURE_MANIFEST=''
ASSET_ROOT="$REPO_ROOT/generated_assets"
TIMEOUT_SECONDS=900
TRACKING_MODE=offline
CONFIG=''
PROFILE=''
PLAN=false
usage() {
    echo 'Usage: bash scripts/run_gpu_checks.sh --mode smoke-basic|smoke-multi|rehearsal-basic|rehearsal-multi|experiment --output NEW_PATH --fixture-manifest PATH [--config GENERATED_RECIPE (experiment only)] [--tracking-mode offline|online] [--timeout-seconds 900] [--asset-root REPO/generated_assets] [--plan]'
}
while (($#)); do
    case "$1" in
        --plan|--dry-run) PLAN=true; shift ;;
        --mode|--output|--fixture-manifest|--timeout-seconds|--asset-root|--tracking-mode|--config)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            case "$1" in
                --mode) MODE="$2" ;;
                --output) OUTPUT="$2" ;;
                --fixture-manifest) FIXTURE_MANIFEST="$2" ;;
                --timeout-seconds) TIMEOUT_SECONDS="$2" ;;
                --asset-root) ASSET_ROOT="$2" ;;
                --tracking-mode) TRACKING_MODE="$2" ;;
                --config) CONFIG="$2" ;;
            esac
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$MODE" in smoke-basic|smoke-multi|rehearsal-basic|rehearsal-multi|experiment) ;; *) usage >&2; exit 2 ;; esac
if [[ "$MODE" == experiment ]]; then
    [[ -n "$CONFIG" ]] || { echo 'experiment requires its generated --config recipe.' >&2; exit 2; }
    [[ "$CONFIG" == /* ]] || CONFIG="$REPO_ROOT/$CONFIG"
elif [[ -n "$CONFIG" ]]; then
    echo '--config is supported only for experiment mode; fixed validation modes keep their recorded recipes.' >&2
    exit 2
fi
case "$TRACKING_MODE" in offline|online) ;; *) echo 'Tracking mode must be offline or online.' >&2; exit 2 ;; esac
if [[ "$TRACKING_MODE" == online && "$MODE" == smoke-* ]]; then
    echo 'Smoke modes do not create W&B runs. Use rehearsal-basic or rehearsal-multi with --tracking-mode online.' >&2
    exit 2
fi
[[ -n "$OUTPUT" && -n "$FIXTURE_MANIFEST" ]] || { usage >&2; exit 2; }
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]{0,5}$ ]] || { echo 'Timeout must be a positive integer, at most six digits.' >&2; exit 2; }
[[ "$OUTPUT" == /* ]] || OUTPUT="$REPO_ROOT/$OUTPUT"
[[ "$FIXTURE_MANIFEST" == /* ]] || FIXTURE_MANIFEST="$REPO_ROOT/$FIXTURE_MANIFEST"
[[ "$ASSET_ROOT" == /* ]] || ASSET_ROOT="$REPO_ROOT/$ASSET_ROOT"
[[ "$ASSET_ROOT" == "$REPO_ROOT/generated_assets" ]] || { echo 'Only the bundled REPO/generated_assets root is supported.' >&2; exit 2; }
case "$MODE" in
    smoke-basic|smoke-multi)
        TASK=Basic-Car-Training-v0; NUM_ENVS=12
        if [[ "$MODE" == smoke-multi ]]; then TASK=Multiagent-Car-Training-v0; NUM_ENVS=4; fi
        COMMAND=(bash scripts/isaac.sh scripts/smoke_test.py --task "$TASK" --headless --enable_cameras --fixture --num_envs "$NUM_ENVS" --steps 32 --train --output "$OUTPUT/smoke")
        ;;
    rehearsal-basic) PROFILE=basic-smoke; CONFIG="$OUTPUT/requested-recipe.json" ;;
    rehearsal-multi) PROFILE=multiagent-smoke; CONFIG="$OUTPUT/requested-recipe.json" ;;
esac
if [[ -n "$CONFIG" ]]; then
    COMMAND=(bash scripts/run_experiment.sh "$OUTPUT/experiment" "$CONFIG" --externally-supervised)
fi
if "$PLAN"; then
    printf 'PLAN ONLY: no GPU access, simulator, network calls, or files created.\n'
    printf 'Require Linux x86_64, supplied OMNI_KIT_ACCEPT_EULA=YES, GNU timeout, flock and an installed locked runtime.\n'
    printf 'Acquire %s/.gpu-job.lock; create fresh output %s\n' "$REPO_ROOT" "$OUTPUT"
    printf 'Verify source provenance and pinned installed Isaac Lab runtime; verify-tree fixture bundle manifest: %s\n' "$FIXTURE_MANIFEST"
    if [[ -n "$PROFILE" ]]; then
        printf 'Resolve config.yaml profile %s into the fresh job receipt %s.\n' "$PROFILE" "$CONFIG"
    fi
    printf 'Set WANDB_MODE=%s and isolate W&B data/cache/config/artifact directories under the job output.\n' "$TRACKING_MODE"
    if [[ "$TRACKING_MODE" == online ]]; then
        printf 'Verify W&B authentication and destination before GPU preflight or Kit; fail without falling back to offline.\n'
    fi
    printf 'Whole job deadline: %s seconds (+30 seconds kill grace); run driver/Vulkan preflight.\n' "$TIMEOUT_SECONDS"
    printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
    exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Execution requires Linux x86_64; use --plan on the Mac.' >&2; exit 2; }
[[ "${OMNI_KIT_ACCEPT_EULA:-}" == YES ]] || { echo 'Supply OMNI_KIT_ACCEPT_EULA=YES yourself after reviewing the NVIDIA EULA.' >&2; exit 2; }
for required in timeout flock; do
    command -v "$required" >/dev/null || { echo "Missing required command: $required" >&2; exit 2; }
done
RUNTIME_PYTHON="$REPO_ROOT/.venv/bin/python"
[[ -x "$RUNTIME_PYTHON" ]] || { echo 'Install the pinned runtime first with scripts/setup_runtime.sh.' >&2; exit 2; }
[[ -f "$FIXTURE_MANIFEST" && ! -L "$FIXTURE_MANIFEST" ]] || { echo 'Missing or symlinked fixture bundle manifest.' >&2; exit 2; }
if [[ "$MODE" == experiment ]]; then
    [[ -f "$CONFIG" && ! -L "$CONFIG" ]] || { echo 'Missing or symlinked generated recipe.' >&2; exit 2; }
fi
cd "$REPO_ROOT"
exec 9> .gpu-job.lock
flock -n 9 || { echo 'Another setup or GPU job holds this repository lock.' >&2; exit 75; }
# mkdir without -p for the leaf is the atomic no-overwrite check, including symlinks.
mkdir -p -- "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT" || { echo 'Output must be a fresh unique directory.' >&2; exit 2; }
printf 'Validation job log: %s/job.log\n' "$OUTPUT"
STARTED_EPOCH=$(date +%s)
job_pid=''
job_waited=false
cleanup() {
    local result=$?
    trap - EXIT TERM INT HUP
    if [[ -n "$job_pid" ]]; then
        kill -TERM -- "-$job_pid" 2>/dev/null || true
        kill -KILL -- "-$job_pid" 2>/dev/null || true
        # A signal can arrive before timeout establishes its process group.
        if ! "$job_waited"; then kill -KILL "$job_pid" 2>/dev/null || true; fi
        wait "$job_pid" 2>/dev/null || true
    fi
    # Preserve the primary failure even if a full disk or filesystem error also
    # prevents status persistence. Successful work without its receipt is failure.
    if ! { FINISHED_EPOCH=$(date +%s) &&
        printf '{"schema_version":1,"mode":"%s","tracking_mode":"%s","exit_code":%s,"timeout_seconds":%s,"started_epoch":%s,"finished_epoch":%s}\n' \
            "$MODE" "$TRACKING_MODE" "$result" "$TIMEOUT_SECONDS" "$STARTED_EPOCH" "$FINISHED_EPOCH" > "$OUTPUT/exit_status.json"; }; then
        echo 'Failed to persist exit_status.json; preserve logs and inspect this job manually.' >&2 || true
        if (( result == 0 )); then result=74; fi
    fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
export WANDB_MODE="$TRACKING_MODE"
export WANDB_DIR="$OUTPUT/wandb"
export WANDB_DATA_DIR="$OUTPUT/wandb-data"
export WANDB_CACHE_DIR="$OUTPUT/wandb-cache"
export WANDB_CONFIG_DIR="$OUTPUT/wandb-config"
export WANDB_ARTIFACT_DIR="$OUTPUT/wandb-artifacts"
# Every job is fresh. Offline validation must not inherit online credentials;
# explicitly online jobs use credentials already supplied securely on this host.
unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM WANDB_FORK_FROM WANDB_NAME
if [[ "$TRACKING_MODE" == offline ]]; then unset WANDB_API_KEY; fi
export CAR_TRAINING_ASSET_ROOT="$ASSET_ROOT" PYTHONUNBUFFERED=1
export GPU_JOB_OUTPUT="$OUTPUT"
mkdir -- "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR"
read -r -d '' WORKER <<'WORKER_END' || true
set -euo pipefail
root=$1
fixture_manifest=$2
config=$3
profile=$4
shift 4
runtime_python="$root/.venv/bin/python"
"$runtime_python" - "$root" "$fixture_manifest" <<'PY'
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from rc_car.provenance import inspect_isaaclab_runtime, verify_source_files
root = Path(sys.argv[1])
output = Path(os.environ["GPU_JOB_OUTPUT"])
source = root / "outputs/source-provenance.json"
verify_source_files(root, json.loads(source.read_text())["files_sha256"])
runtime = inspect_isaaclab_runtime()
shutil.copy2(source, output / "source-provenance.json")
shutil.copy2(sys.argv[2], output / "fixture-bundle-manifest.json")
(output / "runtime-identity.json").write_text(json.dumps(asdict(runtime), indent=2) + "\n")
PY
if [[ -n "$profile" ]]; then
    "$runtime_python" - "$root/config.yaml" "$profile" "$config" "$WANDB_MODE" <<'PY'
import json
import sys
from pathlib import Path
from rc_car.experiments.config import validate_config
from rc_car.workflow_config import load_workflow
recipe = load_workflow(sys.argv[1]).requested_recipe(sys.argv[2])
recipe["wandb"]["mode"] = sys.argv[4]
validate_config(recipe)
with Path(sys.argv[3]).open("x") as stream:
    stream.write(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
PY
fi
"$runtime_python" scripts/bundle.py verify-tree --root "$root" --manifest "$fixture_manifest" --require-fixtures
test -f "$root/generated_assets/fixture.json"
if [[ "$WANDB_MODE" == online ]]; then
    tracking_result=0
    "$runtime_python" -m rc_car.experiments.online_check --config "$config" || tracking_result=$?
    if (( tracking_result != 0 )); then
        echo 'Online W&B preflight failed; no GPU workload was started. Fix host authentication/access and use a fresh output path, or explicitly rerun with --tracking-mode offline and a fresh output path.' >&2
        exit "$tracking_result"
    fi
fi
bash scripts/gpu_preflight.sh
exec "$@"
WORKER_END
# GNU timeout owns the process group; never use --foreground, which would leave
# descendants outside timeout's signal broadcast. Escaped sessions are not covered.
timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" bash -c "$WORKER" -- \
    "$REPO_ROOT" "$FIXTURE_MANIFEST" "$CONFIG" "$PROFILE" "${COMMAND[@]}" > "$OUTPUT/job.log" 2>&1 &
job_pid=$!
job_result=0
wait "$job_pid" || job_result=$?
job_waited=true
exit "$job_result"
