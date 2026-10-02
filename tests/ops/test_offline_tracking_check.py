"""Safety boundary checks for the opt-in actual SDK gate; no SDK run here."""

import ast
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rc_car.lifecycle import best_effort_cleanup
from rc_car.ops import offline_check
from rc_car.ops.offline_check import _cleanup_worker, isolated_environment


def test_worker_environment_excludes_inherited_credentials_and_resume_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for setting in ("WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_RESUME", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(setting, "synthetic-inherited-test-value")
    monkeypatch.setenv("WANDB_MODE", "online")
    environment = isolated_environment(tmp_path)
    assert environment["WANDB_MODE"] == "offline"
    assert environment["WANDB_BASE_URL"] == "http://127.0.0.1:9"
    assert environment["WANDB_ERROR_REPORTING"] == "false"
    assert "synthetic-inherited-test-value" not in environment.values()
    assert environment["NETRC"] == str(tmp_path / "no-netrc")
    for setting in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        assert Path(environment[setting]).is_relative_to(tmp_path)


def test_cleanup_kills_remaining_group_even_after_client_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.pid = 12345
    process.wait.return_value = 0
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(offline_check.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    original = subprocess.TimeoutExpired("synthetic", 90)
    _cleanup_worker(process, original)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert process.wait.call_count == 2
    assert not getattr(original, "__notes__", [])


def test_cleanup_tolerates_process_group_exit_race(monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.pid = 12345

    def missing_group(_pid: int, _signum: int) -> None:
        raise ProcessLookupError("already exited")

    monkeypatch.setattr(offline_check.os, "killpg", missing_group)
    original = RuntimeError("original worker failure")
    _cleanup_worker(process, original)
    assert process.wait.call_count == 2
    assert not getattr(original, "__notes__", [])


def test_cleanup_failures_do_not_hide_original_or_skip_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.pid = 12345
    process.wait.side_effect = subprocess.TimeoutExpired("synthetic", 5)
    kill = MagicMock(side_effect=PermissionError("synthetic failure"))
    monkeypatch.setattr(offline_check.os, "killpg", kill)
    original = KeyboardInterrupt()
    _cleanup_worker(process, original)
    assert kill.call_count == 2
    assert process.wait.call_count == 2
    assert len(original.__notes__) == 4


def test_worker_finish_failure_preserves_original_logging_error() -> None:
    # Execute the real worker's log/finalize block with fakes; no SDK import/run.
    tree = ast.parse(Path(offline_check.__file__).read_text())
    worker = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker"
    )
    block = next(node for node in worker.body if isinstance(node, ast.Try))
    original = RuntimeError("logging failure")
    run = SimpleNamespace(
        log=MagicMock(side_effect=original),
        finish=MagicMock(side_effect=RuntimeError("cleanup failure")),
    )
    namespace = {"run": run, "best_effort_cleanup": best_effort_cleanup}
    with pytest.raises(RuntimeError, match="logging failure") as caught:
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), "<worker-log-block>", "exec"),
            namespace,
        )
    assert caught.value is original
    assert "RuntimeError" in original.__notes__[0]
    run.finish.assert_called_once_with(exit_code=1)
