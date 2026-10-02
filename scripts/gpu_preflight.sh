#!/usr/bin/env bash
# Fast host checks before downloading Isaac or running paid experiments.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'GPU preflight requires the Linux NVIDIA host.' >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if (( DRIVER_MAJOR >= 590 )); then
    echo 'Isaac 5.1 previously crashed with driver 595. Validate a 570/580 host before proceeding.' >&2
    exit 1
fi
command -v vulkaninfo >/dev/null || { echo 'Install vulkan-tools and graphics libraries first.' >&2; exit 1; }
VULKAN_REPORT="$(vulkaninfo --summary 2>&1)"
echo "$VULKAN_REPORT"
if ! grep -q 'driverID.*NVIDIA' <<< "$VULKAN_REPORT"; then
    echo 'Vulkan does not expose the NVIDIA device. Stop before simulator setup.' >&2
    exit 1
fi
free -h
df -h .
echo 'HOST PREFLIGHT PASSED; simulator rendering still requires a smoke test.'
