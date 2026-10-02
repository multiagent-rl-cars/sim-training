#!/usr/bin/env bash
# Explicit Linux-host setup. No cloud calls, driver changes, or implicit EULA acceptance.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN=false
INSTALL_SYSTEM_DEPS=false
TIMEOUT_SECONDS=3600
usage() {
    echo 'Usage: bash scripts/setup_runtime.sh [--plan] [--install-system-deps] [--timeout-seconds 3600]'
}
while (($#)); do
    case "$1" in
        --plan|--dry-run) PLAN=true; shift ;;
        --install-system-deps) INSTALL_SYSTEM_DEPS=true; shift ;;
        --timeout-seconds) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; TIMEOUT_SECONDS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]{0,5}$ ]] || { echo 'Timeout must be a positive integer, at most six digits.' >&2; exit 2; }
if "$PLAN"; then
    printf 'PLAN ONLY: no installation, GPU access, network calls, or files created.\n'
    printf 'Require Linux x86_64, supplied OMNI_KIT_ACCEPT_EULA=YES, uv, GNU timeout and flock.\n'
    printf 'Acquire one shared lock: %s/.gpu-job.lock\n' "$REPO_ROOT"
    if "$INSTALL_SYSTEM_DEPS"; then
        printf 'Explicit apt system dependencies: vulkan-tools libglu1-mesa libxt6 libxrender1 libxi6 libxrandr2 ffmpeg. No driver packages.\n'
    fi
    printf 'Whole setup deadline: %s seconds (+30 seconds kill grace).\n' "$TIMEOUT_SECONDS"
    printf 'Run scripts/gpu_preflight.sh then scripts/setup_isaac.sh; verify the pinned distribution and locked install.\n'
    exit 0
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Execution requires Linux x86_64; use --plan on the Mac.' >&2; exit 2; }
[[ "${OMNI_KIT_ACCEPT_EULA:-}" == YES ]] || { echo 'Supply OMNI_KIT_ACCEPT_EULA=YES yourself after reviewing the NVIDIA EULA.' >&2; exit 2; }
for required in timeout flock uv; do
    command -v "$required" >/dev/null || { echo "Missing required command: $required" >&2; exit 2; }
done
cd "$REPO_ROOT"
exec 9> .gpu-job.lock
flock -n 9 || { echo 'Another setup or GPU job holds this repository lock.' >&2; exit 75; }
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
    exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
# GNU timeout creates the supervised process group. Children deliberately escaping
# that group and an uncatchable SIGKILL of this supervisor are outside the guarantee.
timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" bash -c '
    set -euo pipefail
    if [[ "$1" == true ]]; then
        command -v apt-get >/dev/null || { echo "--install-system-deps requires apt-get" >&2; exit 2; }
        elevate=()
        if (( EUID != 0 )); then elevate=(sudo -n); fi
        "${elevate[@]}" apt-get update
        "${elevate[@]}" apt-get install -y --no-install-recommends vulkan-tools libglu1-mesa libxt6 libxrender1 libxi6 libxrandr2 ffmpeg
    fi
    bash scripts/gpu_preflight.sh
    bash scripts/setup_isaac.sh
' -- "$INSTALL_SYSTEM_DEPS" &
job_pid=$!
job_result=0
wait "$job_pid" || job_result=$?
job_waited=true
exit "$job_result"
