# Emergency handoff

Historical commands and environment layouts below are unsupported. Use
`uv run rc-car` and the [current README](../README.md).

> Historical handoff: September 23, 2026. The emergency stop, instance state and
> backup notes below describe that session, not current authorization or cloud
> state. Use the [current workflow](../README.md), [operations runbook](../ops/README.md)
> and [experiment registry](../experiments/README.md). This notice was added on
> September 27; the original evidence below is preserved.

User requested immediate backup because laptop battery reached 3%, before they
delete the Brev instance. Stop experiment work.

No timed training run started. No new trained checkpoint or training video exists.
All source changes were authored locally and are already on this laptop:

- `scripts/training_experiment.py`: bounded training, W&B, periodic video, evaluation route metrics.
- `scripts/upload_experiment.py`: evaluation/media upload to the original W&B run.
- `scripts/run_experiment.sh`, `scripts/compare_evaluations.py`: orchestration updates.
- `configs/experiments/line_following_15min.json`: 12 environments, 900 seconds.
- `environments/isaac/pyproject.toml` and `uv.lock`: pinned W&B dependency.
- `docs/line-following-15min-attempt.md`: claim, deviations, setup details.
- `outputs/line15-source.tar.gz`, `outputs/line15-deploy.tar.gz`: local source/deployment snapshots (individual files may be newer).

Remote: Brev `my-instance`, `/home/ubuntu/sim-training`, L4 24 GB, 16 GB RAM.
Isaac runtime installed successfully. W&B login verified for `2710-multiagent-rl`;
credential is only in remote `/home/ubuntu/.netrc` and is intentionally excluded
from the backup. No W&B training run was created.

Initial GPU smoke test failed because graphics libraries were missing. Vulkan
was repaired, but Isaac 5.1 crashed in RTX startup with driver 595.91.07.
580.178.04 driver packages and a matching GCP kernel module were installed.
Reboot was issued; post-reboot simulator validation was not completed before
the emergency stop. Resume with `nvidia-smi`, Vulkan check, then a bounded
12-environment smoke test. Do not assume the new driver solved the crash.

Remote backup target: `outputs/handoff/line15-emergency-20260923/`.
Runtime virtual environments, downloadable packages, caches, and credentials
are excluded. These can be rebuilt from the source and lockfile. Cloud deletion
is being handled by the user; it has not been performed or verified by the agent.

Backup status: local source archive created and hashed successfully:
`bb807f26504c199c0f12955255c2822b00c8629ea6581aa0697df870004750a5`.
The first remote retrieval failed because SSH closed during reboot; its archive
is empty and is NOT a usable backup. Remote installation/failure logs may be lost
if the instance is deleted before reconnection. There are no new training results
to recover. Existing baseline-002 results remain in the earlier local handoff.

FINAL UPDATE: reconnect succeeded. `remote-logs-outputs-retry.tar.gz` downloaded
successfully (SSH/tar exit 0), and its archive listing was verified locally.
It contains all remote project outputs and logs, Isaac installation log, driver
580 installation log, and both failed smoke-test logs. The empty first-attempt
archive is not the backup; use the `-retry` archive.
