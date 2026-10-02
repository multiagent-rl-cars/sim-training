"""Explicitly sync ONE verified offline W&B run from a disposable copy.

This uploads metrics/media, not source archives. Artifact errors from old GPU
paths are expected and must be repaired with scripts/upload_experiment.py.
CLI success is NOT an upload-verification receipt. Originals remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from rc_car.ops.bundles import _digest, _open_regular
from rc_car.ops.wandb_handoff import prepare_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-directory", default="experiment")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be 1..600 seconds")
    backup = args.backup_root.resolve(strict=True)
    work = args.work_dir.resolve()
    if work.is_relative_to(backup) or backup.is_relative_to(work):
        parser.error("work directory must be outside the backup tree")
    plan = prepare_handoff(backup, args.manifest, run_directory=args.run_directory)
    # Parse the immutable stream and identity before sending anything.
    from rc_car.ops.wandb_sync import inspect_stream

    expected = inspect_stream(backup / plan["offline_record"], plan["identity"])
    work.mkdir(parents=True, exist_ok=False)
    for entry in plan["transport_files"]:
        destination = work / "copy" / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_regular(backup, entry["path"]) as source, destination.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        with destination.open("rb") as copied:
            if _digest(copied) != (entry["size"], entry["sha256"]):
                raise ValueError("Transport bytes changed while making the disposable copy")
    if inspect_stream(work / "copy" / plan["offline_record"], plan["identity"]) != expected:
        raise ValueError("Copied stream and media differ from the verified originals")
    (work / "handoff-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (work / "expected-stream.json").write_text(json.dumps(expected, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "online",
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_DISABLE_GIT": "true",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_ERROR_REPORTING": "false",
            "WANDB_CONSOLE": "off",
            "WANDB_X_DISABLE_MACHINE_INFO": "true",
            "WANDB_X_DISABLE_STATS": "true",
            "WANDB_INIT_TIMEOUT": "30",
            "WANDB_X_SERVICE_WAIT": "30",
        }
    )
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM"):
        env.pop(key, None)
    for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        path = work / key.lower()
        path.mkdir()
        env[key] = str(path)
    identity = plan["identity"]
    command = [
        sys.executable,
        "-m",
        "wandb",
        "sync",
        "--no-sync-tensorboard",
        "--entity",
        identity["entity"],
        "--project",
        identity["project"],
        str((work / "copy" / plan["offline_record"]).parent),
    ]
    receipt = {
        "run_id": identity["id"],
        "sync_exit_code": None,
        "server_verified": False,
        "artifact_repair_required": True,
        "outcome": "not-started",
    }
    try:
        with (work / "sync.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["outcome"] = "running"
            try:
                code = process.wait(timeout=args.timeout)
                receipt["sync_exit_code"] = code
                receipt["outcome"] = "completed" if code == 0 else "failed"
            except BaseException as error:
                # Kill the complete process group even if the parent exits first;
                # child upload processes must not outlive the bounded attempt.
                try:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                except BaseException:
                    error.add_note("Could not confirm sync process-group cleanup")
                raise
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
        receipt["outcome"] = (
            "timed-out" if isinstance(error, subprocess.TimeoutExpired) else "interrupted-or-failed"
        )
        raise
    finally:
        # Never include exception strings: a provider error may contain secrets.
        with (work / "sync-status.json").open("x") as status:
            status.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
