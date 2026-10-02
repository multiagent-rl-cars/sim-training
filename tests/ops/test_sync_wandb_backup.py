"""Verified-copy and bounded sync orchestration, without subprocesses/network."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rc_car.ops import wandb_sync as sync_verification


@pytest.fixture
def script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/sync_wandb_backup.py"
    spec = importlib.util.spec_from_file_location("sync_backup_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path: Path, script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    backup = tmp_path / "backup"
    record = "experiment/wandb/offline-run-20260101_120000-run123/run-run123.wandb"
    media = "experiment/wandb/offline-run-20260101_120000-run123/files/media/videos/test.mp4"
    transport = []
    for name in (record, media):
        path = backup / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ("synthetic " + name).encode()
        path.write_bytes(data)
        transport.append(
            {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    (backup / "source.tar.gz").write_bytes(b"must not upload")
    work = tmp_path / "work"
    identity = {"id": "run123", "entity": "test-team", "project": "test-project"}
    plan = {"offline_record": record, "identity": identity, "transport_files": transport}
    monkeypatch.setattr(script, "prepare_handoff", lambda *_args, **_kwargs: plan)
    inspect_calls = []

    def inspect(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
        assert expected_identity == identity
        inspect_calls.append(path)
        return {"stream_sha256": transport[0]["sha256"], "history": [1], "media": [media]}

    monkeypatch.setattr(sync_verification, "inspect_stream", inspect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync",
            "--backup-root",
            str(backup),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--work-dir",
            str(work),
            "--timeout",
            "10",
        ],
    )
    launches: list[Any] = []
    kills: list[Any] = []
    monkeypatch.setattr(script.os, "killpg", lambda pid, sig: kills.append((pid, sig)))

    class Process:
        pid = 12345

        def __init__(self) -> None:
            self.results: list[Any] = [0]
            self.waits: list[int] = []

        def wait(self, *, timeout: int) -> int:
            self.waits.append(timeout)
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    process = Process()

    def popen(command: list[str], **kwargs: Any) -> Process:
        launches.append((command, kwargs))
        return process

    monkeypatch.setattr(script.subprocess, "Popen", popen)
    return {
        "backup": backup,
        "work": work,
        "record": record,
        "media": media,
        "transport": transport,
        "inspect_calls": inspect_calls,
        "launches": launches,
        "kills": kills,
        "process": process,
    }


def test_one_explicit_copied_stream_no_original_mutation(
    script: ModuleType, setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_RUN_ID", "must-not-leak-into-child")
    monkeypatch.setenv("WANDB_RESUME", "must-not-leak-into-child")
    before = {str(p): p.read_bytes() for p in setup["backup"].rglob("*") if p.is_file()}
    assert script.main() == 0
    assert setup["inspect_calls"] == [
        setup["backup"] / setup["record"],
        setup["work"] / "copy" / setup["record"],
    ]
    command, kwargs = setup["launches"][0]
    assert len(setup["launches"]) == 1
    assert command[:5] == [sys.executable, "-m", "wandb", "sync", "--no-sync-tensorboard"]
    assert "--sync-all" not in command and "--clean" not in command
    assert command[-1] == str((setup["work"] / "copy" / setup["record"]).parent)
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == setup["work"]
    assert kwargs["env"]["WANDB_MODE"] == "online"
    assert "WANDB_RUN_ID" not in kwargs["env"] and "WANDB_RESUME" not in kwargs["env"]
    for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        assert Path(kwargs["env"][key]).is_relative_to(setup["work"])
    assert not (setup["work"] / "copy/source.tar.gz").exists()
    assert {str(p): p.read_bytes() for p in setup["backup"].rglob("*") if p.is_file()} == before
    receipt = json.loads((setup["work"] / "sync-status.json").read_text())
    assert receipt["sync_exit_code"] == 0 and receipt["outcome"] == "completed"
    assert receipt["artifact_repair_required"] and not receipt["server_verified"]


@pytest.mark.parametrize("failure", ["mutated-bytes", "symlink", "copied-inspection"])
def test_changed_or_redirected_copy_never_launches(
    script: ModuleType, setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original_inspect = sync_verification.inspect_stream

    def inspect(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
        result = original_inspect(path, identity)
        if len(setup["inspect_calls"]) == 1:
            target = setup["backup"] / setup["record"]
            if failure == "mutated-bytes":
                target.write_bytes(b"changed")
            elif failure == "symlink":
                target.rename(target.with_suffix(".elsewhere"))
                target.symlink_to(target.with_suffix(".elsewhere"))
        elif failure == "copied-inspection":
            result["history"] = [999]
        return result

    monkeypatch.setattr(sync_verification, "inspect_stream", inspect)
    with pytest.raises((ValueError, OSError)):
        script.main()
    assert setup["launches"] == []


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "cleanup-fails"])
def test_interruption_kills_group_and_retains_unverified_receipt(
    script: ModuleType, setup: dict[str, Any], failure: str
) -> None:
    error = KeyboardInterrupt() if failure == "interrupt" else subprocess.TimeoutExpired("sync", 10)
    last = subprocess.TimeoutExpired("cleanup", 5) if failure == "cleanup-fails" else -9
    setup["process"].results = [error, subprocess.TimeoutExpired("cleanup", 5), last]
    with pytest.raises(type(error)) as caught:
        script.main()
    assert caught.value is error
    assert setup["kills"] == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert setup["process"].waits == [10, 5, 5]
    receipt = json.loads((setup["work"] / "sync-status.json").read_text())
    assert receipt["sync_exit_code"] is None and not receipt["server_verified"]
    assert receipt["error_type"] == type(error).__name__
    assert receipt["outcome"] in {"timed-out", "interrupted-or-failed"}
    if failure == "cleanup-fails":
        assert any("cleanup" in note for note in getattr(error, "__notes__", []))


def test_launch_failure_receipt_omits_error_message(
    script: ModuleType, setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("synthetic potentially sensitive provider response")

    monkeypatch.setattr(script.subprocess, "Popen", fail)
    with pytest.raises(OSError):
        script.main()
    text = (setup["work"] / "sync-status.json").read_text()
    assert "sensitive" not in text
    assert json.loads(text)["error_type"] == "OSError"
    assert setup["kills"] == []


def test_nonzero_exit_is_not_verification(script: ModuleType, setup: dict[str, Any]) -> None:
    setup["process"].results = [7]
    assert script.main() == 7
    receipt = json.loads((setup["work"] / "sync-status.json").read_text())
    assert receipt["outcome"] == "failed" and receipt["sync_exit_code"] == 7
    assert not receipt["server_verified"]


def test_work_inside_backup_rejected_before_launch(
    script: ModuleType, setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = list(sys.argv)
    argv[argv.index("--work-dir") + 1] = str(setup["backup"] / "forbidden")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        script.main()
    assert caught.value.code == 2 and not setup["launches"]


def test_existing_work_never_overwritten(script: ModuleType, setup: dict[str, Any]) -> None:
    setup["work"].mkdir()
    with pytest.raises(FileExistsError):
        script.main()
    assert not setup["launches"]
