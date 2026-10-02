"""Mocked SDK transport tests: no network, W&B install or simulator required."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from rc_car.experiments import load_config, log_video, start_tracking, tracking

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    run = MagicMock()
    run.id, run.url, run.entity, run.project = "run-id", None, "test-entity", "test-project"
    run.name = "sdk-generated-name"
    fake = SimpleNamespace(init=MagicMock(return_value=run), Video=MagicMock(return_value="media"))
    monkeypatch.setattr(tracking, "_load_wandb", lambda: fake)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    for setting in (
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RESUME_FROM",
        "WANDB_FORK_FROM",
        "WANDB_NAME",
    ):
        monkeypatch.delenv(setting, raising=False)
    return fake


def recipe() -> dict[str, Any]:
    return load_config(ROOT / "tests/fixtures/legacy_recipes/line_following_15min.json").model_dump(
        mode="json", exclude_none=True
    )


def test_offline_override_and_only_public_identity(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_API_KEY", "private-test-value")
    run = start_tracking(recipe(), tmp_path)
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["mode"] == "offline"
    assert kwargs["reinit"] == "create_new"
    assert kwargs["resume"] == "never"
    assert "name" not in kwargs
    assert isinstance(kwargs["id"], str) and len(kwargs["id"]) == 32
    assert kwargs["config"]["wandb"]["mode"] == "offline"
    identity_text = (tmp_path / "wandb_run.json").read_text()
    assert json.loads(identity_text) == {
        "id": "run-id",
        "name": "sdk-generated-name",
        "url": None,
        "entity": "test-entity",
        "project": "test-project",
        "mode": "offline",
    }
    assert "private-test-value" not in identity_text
    assert "private-test-value" not in repr(kwargs)
    assert run is sdk.init.return_value


def test_existing_identity_is_not_overwritten(sdk: SimpleNamespace, tmp_path: Path) -> None:
    identity = tmp_path / "wandb_run.json"
    identity.write_text("existing")
    with pytest.raises(FileExistsError):
        start_tracking(recipe(), tmp_path)
    assert identity.read_text() == "existing"
    sdk.init.assert_not_called()


@pytest.mark.parametrize(
    "setting",
    ["WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM", "WANDB_NAME"],
)
def test_implicit_resume_controls_are_rejected(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, setting: str
) -> None:
    monkeypatch.setenv(setting, "existing-run-setting")
    with pytest.raises(ValueError, match="fresh tracked run"):
        start_tracking(recipe(), tmp_path)
    sdk.init.assert_not_called()


def test_distinct_tracking_starts_request_distinct_runs(
    sdk: SimpleNamespace, tmp_path: Path
) -> None:
    start_tracking(recipe(), tmp_path / "first")
    first_id = sdk.init.call_args.kwargs["id"]
    start_tracking(recipe(), tmp_path / "second")
    assert sdk.init.call_args.kwargs["id"] != first_id


def test_bad_mode_fails_before_sdk(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_MODE", "unexpected")
    with pytest.raises(ValueError, match="mode"):
        start_tracking(recipe(), tmp_path)
    sdk.init.assert_not_called()


def test_missing_destination_fails_before_sdk(sdk: SimpleNamespace, tmp_path: Path) -> None:
    config = recipe()
    config.pop("wandb")
    with pytest.raises(ValueError, match="destination"):
        start_tracking(config, tmp_path)
    sdk.init.assert_not_called()


def test_failed_setup_closes_run(sdk: SimpleNamespace, tmp_path: Path) -> None:
    sdk.init.return_value.define_metric.side_effect = RuntimeError("metric failure")
    with pytest.raises(RuntimeError, match="metric failure"):
        start_tracking(recipe(), tmp_path)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
    assert not (tmp_path / "wandb_run.json").exists()


def test_failed_cleanup_does_not_hide_setup_failure(sdk: SimpleNamespace, tmp_path: Path) -> None:
    sdk.init.return_value.define_metric.side_effect = RuntimeError("metric failure")
    sdk.init.return_value.finish.side_effect = RuntimeError("cleanup failure")
    with pytest.raises(RuntimeError, match="metric failure"):
        start_tracking(recipe(), tmp_path)


def test_identity_write_failure_closes_run(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = recipe()

    def fail_flush(_fd: int) -> None:
        raise OSError("simulated storage failure")

    monkeypatch.setattr(tracking.os, "fsync", fail_flush)
    with pytest.raises(OSError, match="storage failure"):
        start_tracking(config, tmp_path)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
    assert not (tmp_path / "wandb_run.json").exists()
    assert not list(tmp_path.glob(".wandb-identity-*"))


def test_sdk_must_return_a_run(sdk: SimpleNamespace, tmp_path: Path) -> None:
    sdk.init.return_value = None
    with pytest.raises(RuntimeError, match="did not return a run"):
        start_tracking(recipe(), tmp_path)
    assert not (tmp_path / "wandb_run.json").exists()


def test_invalid_sdk_identity_closes_run(sdk: SimpleNamespace, tmp_path: Path) -> None:
    sdk.init.return_value.id = ""
    with pytest.raises(RuntimeError, match="invalid id"):
        start_tracking(recipe(), tmp_path)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("name", ["", "   ", 17, object()])
def test_invalid_sdk_display_name_closes_run(
    sdk: SimpleNamespace, tmp_path: Path, name: object
) -> None:
    sdk.init.return_value.name = name
    with pytest.raises(RuntimeError, match="invalid display name"):
        start_tracking(recipe(), tmp_path)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
    assert not (tmp_path / "wandb_run.json").exists()


def test_missing_online_display_name_is_not_fabricated(
    sdk: SimpleNamespace, tmp_path: Path
) -> None:
    sdk.init.return_value.name = None
    with pytest.raises(RuntimeError, match="did not return an online display name"):
        start_tracking(recipe(), tmp_path)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_sdk_may_omit_display_name_offline(
    sdk: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk.init.return_value.name = None
    monkeypatch.setenv("WANDB_MODE", "offline")
    start_tracking(recipe(), tmp_path)
    identity = json.loads((tmp_path / "wandb_run.json").read_text())
    assert identity["name"] is None
    assert "name" not in sdk.init.call_args.kwargs


def test_unserializable_identity_leaves_no_file(sdk: SimpleNamespace, tmp_path: Path) -> None:
    sdk.init.return_value.url = object()
    with pytest.raises(RuntimeError, match="invalid url"):
        start_tracking(recipe(), tmp_path)
    assert not (tmp_path / "wandb_run.json").exists()
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_competing_identity_writer_is_not_clobbered(
    sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_link = tracking.os.link

    def race(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_text("another writer")
        original_link(source, destination)

    monkeypatch.setattr(tracking.os, "link", race)
    with pytest.raises(FileExistsError):
        start_tracking(recipe(), tmp_path)
    assert (tmp_path / "wandb_run.json").read_text() == "another writer"
    assert not list(tmp_path.glob(".wandb-identity-*"))
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise ModuleNotFoundError("not installed", name=name)

    monkeypatch.setattr(tracking.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="tracking dependency group"):
        tracking._load_wandb()


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_invalid_media_fails_before_sdk(sdk: SimpleNamespace, tmp_path: Path, kind: str) -> None:
    path = tmp_path / "video.mp4"
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    with pytest.raises(ValueError, match="Missing or empty"):
        log_video(None, path)
    sdk.Video.assert_not_called()


def test_video_uses_selected_metric_and_step(sdk: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"mocked media transport, not a decoding test")
    log_video(sdk.init.return_value, path, key="evaluation/video", timesteps=128)
    sdk.Video.assert_called_once_with(str(path), format="mp4")
    sdk.init.return_value.log.assert_called_once_with(
        {"evaluation/video": "media", "training/agent_timesteps": 128}
    )
