"""Explicit local W&B SDK integration gate using only synthetic data.

Internal integration-test worker, not a user-facing training command.
Requires the tracking dependency group plus ffmpeg/ffprobe. No login, upload,
Isaac, or GPU is involved. Ordinary pytest checks do not run this SDK gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from rc_car.experiments.tracking import log_video, start_tracking
from rc_car.lifecycle import best_effort_cleanup

EXPECTED_SDK_VERSION = "0.25.1"


def isolated_environment(directory: Path) -> dict[str, str]:
    """Build an allowlisted child environment without reading credential values."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONUNBUFFERED": "1",
        "NETRC": str(directory / "no-netrc"),
        "WANDB_MODE": "offline",
        "WANDB_BASE_URL": "http://127.0.0.1:9",
        "WANDB_CONFIG_DIR": str(directory / "sdk-config"),
        "WANDB_CACHE_DIR": str(directory / "sdk-cache"),
        "WANDB_DATA_DIR": str(directory / "sdk-data"),
        "WANDB_ARTIFACT_DIR": str(directory / "sdk-artifacts"),
        "WANDB_DISABLE_GIT": "true",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_ERROR_REPORTING": "false",
        "WANDB_X_DISABLE_MACHINE_INFO": "true",
        "WANDB_X_DISABLE_STATS": "true",
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
        "WANDB_INIT_TIMEOUT": "20",
        "WANDB_X_SERVICE_WAIT": "20",
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signal_group(process: subprocess.Popen[Any], signum: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        # The client or whole group may already have exited between operations.
        pass


def _cleanup_worker(process: subprocess.Popen[Any], original_error: BaseException) -> None:
    """Best-effort bounded cleanup, including surviving SDK group members.

    Even if the Python client exits after TERM, KILL is still attempted against
    its process group. Detached processes outside that group are not covered.
    Cleanup failures become notes; the caller re-raises the original error.
    """
    best_effort_cleanup(
        original_error,
        (
            ("terminate worker group", lambda: _signal_group(process, signal.SIGTERM)),
            ("wait for worker", lambda: process.wait(timeout=5)),
            ("kill surviving group members", lambda: _signal_group(process, signal.SIGKILL)),
            ("reap worker", lambda: process.wait(timeout=5)),
        ),
    )


def _generate_video(directory: Path) -> tuple[Path, dict[str, Any]]:
    video = directory / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        timeout=20,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames,duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    _require(stream["codec_name"] == "h264", "Synthetic clip is not H.264")
    _require(int(stream["nb_read_frames"]) == 20, "Synthetic clip must contain 20 frames")
    return video, stream


def _verify_saved_run(directory: Path, video: Path, marker: Path) -> dict[str, Any]:
    # Internal reader is intentionally confined to this SDK-version-pinned gate.
    # Production tracking does not depend on this private file format.
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    identity = json.loads((directory / "wandb_run.json").read_text())
    _require(identity["mode"] == "offline" and identity["url"] is None, "Expected offline identity")
    run_files = list((directory / "wandb").glob("offline-run-*/*.wandb"))
    _require(len(run_files) == 1, "Expected one persisted offline run")
    run_file = run_files[0]
    reader = DataStore()
    reader.open_for_scan(str(run_file))
    kinds: Counter[str] = Counter()
    scalars: list[float] = []
    video_paths: list[Path] = []
    artifact_paths: list[Path] = []
    exit_codes: list[int] = []
    last_kind: str | None = None
    try:
        while (data := reader.scan_data()) is not None:
            # SDK selects its generated protobuf bindings dynamically at import.
            record = wandb_internal_pb2.Record()  # ty: ignore[unresolved-attribute]
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            last_kind = kind
            if kind is not None:
                kinds[kind] += 1
            if kind == "run":
                _require(record.run.run_id == identity["id"], "Saved run identity mismatch")
            elif kind == "history":
                for item in record.history.item:
                    key = tuple(item.nested_key) if item.nested_key else (item.key,)
                    if key == ("training/test_scalar",):
                        scalars.append(json.loads(item.value_json))
                    elif key == ("test/synthetic_video", "path"):
                        video_paths.append(run_file.parent / "files" / json.loads(item.value_json))
                    elif key == ("test/synthetic_video",):
                        media = json.loads(item.value_json)
                        video_paths.append(run_file.parent / "files" / media["path"])
            elif kind == "artifact":
                for entry in record.artifact.manifest.contents:
                    if entry.path == marker.name:
                        artifact_paths.append(Path(entry.local_path))
            elif kind == "exit":
                exit_codes.append(record.exit.exit_code)
    finally:
        reader.close()
    _require(scalars == [0.0, 0.5, 1.0], "Synthetic scalar history was not fully persisted")
    _require(exit_codes == [0], "Offline run lacks a successful persisted exit record")
    # Core 0.25.1 ends the stream with exit, not a separate footer record.
    _require(last_kind == "exit", "Offline record stream does not end with its exit record")
    _require(len(video_paths) == 1, "Expected one persisted video history entry")
    _require(len(artifact_paths) == 1, "Expected one persisted synthetic artifact entry")
    for saved, original in ((video_paths[0], video), (artifact_paths[0], marker)):
        _require(
            saved.resolve().is_relative_to(directory), "Saved data escaped the isolated directory"
        )
        _require(saved.is_file(), f"Missing saved SDK file: {saved.name}")
        _require(_sha256(saved) == _sha256(original), "SDK file differs from the synthetic source")
    return {
        "run_id": identity["id"],
        "offline_record": str(run_file.relative_to(directory)),
        "record_counts": dict(kinds),
        "persisted_scalars": scalars,
        "exit_code": exit_codes[0],
        "sdk_finish_returned": True,
        "record_stream_read_to_eof": True,
        "last_record_type": last_kind,
        "saved_video": str(video_paths[0].relative_to(directory)),
        "saved_artifact": str(artifact_paths[0].relative_to(directory)),
        "sha256": {
            str(path.relative_to(directory)): _sha256(path)
            for path in (run_file, video, marker, video_paths[0], artifact_paths[0])
        },
    }


def _worker(directory: Path) -> None:
    _require(os.environ.get("WANDB_MODE") == "offline", "Worker must run offline")
    _require(
        os.environ.get("WANDB_BASE_URL") == "http://127.0.0.1:9",
        "Worker service URL must stay on loopback",
    )
    import wandb

    _require(
        wandb.__version__ == EXPECTED_SDK_VERSION, "Revalidate the reader for this SDK version"
    )
    video, probe = _generate_video(directory)
    marker = directory / "synthetic-marker.json"
    marker.write_text(
        json.dumps({"synthetic_only": True, "purpose": "offline SDK transport"}) + "\n"
    )
    config = {
        "name": "synthetic-offline-sdk-check",
        "task": "tracking-test-only",
        "num_envs": 1,
        "train_seed": 0,
        "eval_seed": 0,
        "total_timesteps": 64,
        "eval_steps": 1,
        "video_stride": 1,
        "checkpoint_interval": 64,
        "torch_threads": 1,
        "wandb": {"entity": "offline-test", "project": "offline-sdk-test", "mode": "offline"},
        "test_only": True,
        "synthetic_media": True,
        "note": "Synthetic CPU SDK test; no simulator, model, hardware, or upload",
    }
    run = cast(wandb.Run, start_tracking(config, directory, job_type="offline-sdk-test"))
    try:
        for step in range(3):
            run.log({"training/agent_timesteps": step, "training/test_scalar": step * 0.5})
        log_video(run, video, key="test/synthetic_video", timesteps=2)
        artifact = wandb.Artifact("synthetic-offline-check", type="tracking-test")
        artifact.add_file(str(marker), name=marker.name)
        run.log_artifact(artifact)
        run.summary["synthetic_only"] = True
    except BaseException as error:
        best_effort_cleanup(error, (("finish offline SDK run", lambda: run.finish(exit_code=1)),))
        raise
    else:
        run.finish(exit_code=0)
    report = {
        "schema_version": 1,
        "sdk_version": wandb.__version__,
        "synthetic_only": True,
        "mode": "offline",
        "uploaded": False,
        "video_probe": probe,
        **_verify_saved_run(directory, video, marker),
        "limitations": [
            "Offline SDK persistence only; no online upload or account access verified",
            "No training, simulator, hardware, or driving-quality evidence",
            "SDK internal record reader is pinned to 0.25.1",
        ],
    }
    (directory / "offline-check.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if args.worker:
        _worker(directory)
        return 0
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            parser.exit(2, f"This explicit SDK gate requires {executable}\n")
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("sdk-config", "sdk-cache", "sdk-data", "sdk-artifacts"):
        (directory / name).mkdir()
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                __spec__.name if __spec__ else __name__,
                "--worker",
                "--run-dir",
                str(directory),
            ],
            env=isolated_environment(directory),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=90)
        except BaseException as error:
            _cleanup_worker(process, error)
            raise
    if returncode:
        parser.exit(returncode, f"Offline SDK check failed; inspect {directory / 'worker.log'}\n")
    print(f"Offline SDK files verified: {directory / 'offline-check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
