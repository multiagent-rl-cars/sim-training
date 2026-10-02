"""Completed-output upload tests with a fake SDK; no network or credentials."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rc_car.experiments import upload
from rc_car.experiments.upload import UPLOAD_FILES, upload_completed_experiment, validate_upload

IDENTITY = {
    "id": "run123",
    "entity": "test-team",
    "project": "test-project",
    "mode": "offline",
    "url": None,
}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    root.mkdir()
    for name in UPLOAD_FILES:
        (root / name).write_bytes(("public fixture: " + name).encode())
    for name, value in {
        "wandb_run.json": IDENTITY,
        "config.json": {"wandb": IDENTITY},
        "start_evaluation.json": {"mean_return": 1.25, "resets": 5},
        "end_evaluation.json": {"mean_return": 2.5, "resets": 0},
        "comparison.json": {"matched_initial_states": True},
    }.items():
        (root / name).write_text(json.dumps(value))
    (root / "unexpected_evaluation.json").write_text("must not upload")
    (root / ".netrc").symlink_to(root / "missing")
    return root


@pytest.mark.parametrize("name", [None, "calm-forest-42"])
def test_generated_name_is_optional_metadata_not_run_identity(
    source: Path, name: str | None
) -> None:
    (source / "wandb_run.json").write_text(json.dumps({**IDENTITY, "name": name}))
    plan = validate_upload(source)
    assert plan["identity"]["name"] == name
    # A display-name change never creates a new run or changes resume identity.
    upload._identity(SimpleNamespace(**{**IDENTITY, "name": "renamed"}), plan["identity"])


@pytest.mark.parametrize("name", ["", "  ", 42, []])
def test_invalid_display_name_is_rejected(source: Path, name: object) -> None:
    (source / "wandb_run.json").write_text(json.dumps({**IDENTITY, "name": name}))
    with pytest.raises(ValueError, match="display name"):
        validate_upload(source)


class FakeArtifact:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.id = name + "-id"
        self.qualified_name = "test-team/test-project/" + name + ":v7"
        self.manifest = SimpleNamespace(entries={})
        self.waited = False

    def add_file(self, path: str, *, name: str) -> None:
        digest = upload._hash(Path(path).parent, Path(path).name)
        self.manifest.entries[name] = SimpleNamespace(
            size=digest["size"], digest=digest["wandb_content_digest"]
        )

    def wait(self, *, timeout: int) -> None:
        assert timeout == 120
        self.waited = True


class FakeRun:
    id = "run123"
    entity = "test-team"
    project = "test-project"
    url = "https://wandb.invalid/test-team/test-project/runs/run123"

    def __init__(self) -> None:
        self.summary: Any = {}
        self.artifacts: list[FakeArtifact] = []
        self.remote_files: dict[str, Any] = {}
        self.finishes: list[int] = []

    def log(self, data: dict[str, Any]) -> None:
        self.summary.update(data)
        for media in data.values():
            self.remote_files[media["path"]] = SimpleNamespace(
                name=media["path"], size=media["size"], md5=media["md5"]
            )

    def log_artifact(self, artifact: FakeArtifact) -> FakeArtifact:
        self.artifacts.append(artifact)
        return artifact

    def finish(self, *, exit_code: int = 0) -> None:
        self.finishes.append(exit_code)

    def logged_artifacts(self) -> list[FakeArtifact]:
        return self.artifacts

    def files(self, *, names: list[str]) -> list[Any]:
        return [self.remote_files[name] for name in names if name in self.remote_files]


class FakeSDK:
    def __init__(self) -> None:
        self.remote = FakeRun()
        self.calls: list[str] = []
        self.init_kwargs: dict[str, Any] = {}
        self.init_directories: dict[str, str | None] = {}
        self.missing = False
        self.bad_init = False
        self.bad_artifact = False
        self.bad_media = False
        self.bad_summary = False
        self.unlinked = False

    def Api(self, **kwargs: Any) -> FakeSDK:  # noqa: N802
        self.calls.append("api")
        return self

    def run(self, path: str) -> FakeRun:
        assert path == "test-team/test-project/run123"
        self.calls.append("get_run")
        if self.missing:
            raise ValueError("synthetic missing run")
        return self.remote

    def init(self, **kwargs: Any) -> FakeRun:
        self.calls.append("init")
        self.init_kwargs = kwargs
        self.init_directories = {
            key: os.environ.get(key) for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR")
        }
        if self.bad_init:
            self.remote.id = "unexpected-run"
        return self.remote

    def Settings(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        return kwargs

    def Artifact(self, name: str, *, type: str) -> FakeArtifact:  # noqa: N802
        self.calls.append("artifact")
        return FakeArtifact(name, type)

    def artifact(self, name: str, *, type: str) -> FakeArtifact:
        artifact = next(a for a in self.remote.artifacts if a.qualified_name == name)
        assert artifact.waited
        assert type == artifact.type
        if self.bad_artifact:
            artifact.manifest.entries["unexpected"] = SimpleNamespace(size=1, digest="bad")
        return artifact

    def Video(self, path: str, *, format: str) -> dict[str, Any]:  # noqa: N802
        assert format == "mp4"
        return self.Image(path) | {"_type": "video-file"}

    def Image(self, path: str) -> dict[str, Any]:  # noqa: N802
        digest = upload._hash(Path(path).parent, Path(path).name)
        return {
            "_type": "image-file",
            "sha256": digest["sha256"],
            "size": digest["size"],
            "md5": digest["wandb_content_digest"],
            "path": "media/" + Path(path).name,
        }

    def flush(self) -> None:
        self.calls.append("flush")
        if self.bad_media:
            next(iter(self.remote.remote_files.values())).md5 = "bad"
        if self.bad_summary:
            self.remote.summary["evaluation/start"] = {"different": True}
        if self.unlinked:
            self.remote.artifacts = []


def test_validation_has_no_sdk_import_and_uses_fixed_allowlist(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    plan = validate_upload(source)
    assert set(plan["files"]) == {*UPLOAD_FILES, "wandb_run.json"}
    assert plan["identity"] == IDENTITY
    assert "unexpected_evaluation.json" not in plan["files"]


def test_verified_upload_preserves_source_and_uses_existing_exact_identity(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {p.name: p.read_bytes() for p in source.iterdir() if not p.is_symlink()}
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DATA_DIR", str(source))
    sdk = FakeSDK()
    work = tmp_path / "upload"
    receipt = upload_completed_experiment(source, work_dir=work, sdk=sdk)
    assert receipt["uploaded"] and receipt["server_verified"]
    assert not receipt["training_history_verified"]
    assert json.loads((work / "receipt.json").read_text()) == receipt
    assert sdk.calls[:3] == ["api", "get_run", "init"]
    assert sdk.init_kwargs["mode"] == "online"
    assert sdk.init_kwargs["resume"] == "must"
    assert sdk.init_kwargs["id"] == IDENTITY["id"]
    assert sdk.init_kwargs["save_code"] is False
    assert all(str(work) in str(path) for path in sdk.init_directories.values())
    assert os.environ["WANDB_DATA_DIR"] == str(source)
    assert sdk.remote.finishes == [0]
    assert len(sdk.remote.artifacts) == 2
    assert {p.name: p.read_bytes() for p in source.iterdir() if not p.is_symlink()} == before
    assert not (source / "wandb").exists()


@pytest.mark.parametrize(
    "option", ["missing", "bad_init", "bad_artifact", "bad_media", "bad_summary", "unlinked"]
)
def test_failures_never_claim_verified(source: Path, tmp_path: Path, option: str) -> None:
    sdk = FakeSDK()
    setattr(sdk, option, True)
    work = tmp_path / "upload"
    with pytest.raises((ValueError, RuntimeError)):
        upload_completed_experiment(source, work_dir=work, sdk=sdk)
    receipt = json.loads((work / "receipt.json").read_text())
    assert not receipt["server_verified"]
    assert "error_type" in receipt
    assert "synthetic missing run" not in json.dumps(receipt)
    if option == "missing":
        assert "init" not in sdk.calls and "artifact" not in sdk.calls
    elif option == "bad_init":
        assert "artifact" not in sdk.calls
        assert sdk.remote.finishes == [1]


@pytest.mark.parametrize("name", ["final.zip", "end.mp4", "events.jsonl"])
def test_empty_required_files_fail_before_sdk(source: Path, tmp_path: Path, name: str) -> None:
    (source / name).write_bytes(b"")
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="empty"):
        upload_completed_experiment(source, work_dir=tmp_path / "work", sdk=sdk)
    assert sdk.calls == []


def test_symlink_member_rejected(source: Path, tmp_path: Path) -> None:
    original = source / "final.zip"
    original.rename(tmp_path / "weights")
    original.symlink_to(tmp_path / "weights")
    with pytest.raises((OSError, ValueError)):
        validate_upload(source)


@pytest.mark.parametrize("key", ["work_dir", "receipt_path"])
def test_writes_inside_source_rejected(source: Path, tmp_path: Path, key: str) -> None:
    kwargs: dict[str, Any] = {"work_dir": tmp_path / "work", key: source / "new-output"}
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="outside"):
        upload_completed_experiment(source, sdk=sdk, **kwargs)
    assert sdk.calls == []


def test_existing_work_directory_never_overwritten(source: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    sdk = FakeSDK()
    with pytest.raises(FileExistsError):
        upload_completed_experiment(source, work_dir=work, sdk=sdk)
    assert sdk.calls == []


def test_mismatched_config_rejected(source: Path) -> None:
    (source / "config.json").write_text(json.dumps({"wandb": IDENTITY | {"project": "other"}}))
    with pytest.raises(ValueError, match="destinations differ"):
        validate_upload(source)


@pytest.mark.parametrize(
    "key", ["WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM"]
)
def test_conflicting_environment_rejected(
    source: Path, tmp_path: Path, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(key, "conflict")
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="conflicts"):
        upload_completed_experiment(source, work_dir=tmp_path / "work", sdk=sdk)
    assert sdk.calls == []


def test_concurrent_source_change_rejected_before_sdk(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = upload.validate_upload

    def mutate_after_validation(path: str | Path) -> dict[str, Any]:
        result = original(path)
        if Path(path) == source:
            (source / "final.zip").write_bytes(b"changed")
        return result

    monkeypatch.setattr(upload, "validate_upload", mutate_after_validation)
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="Source changed"):
        upload_completed_experiment(source, work_dir=tmp_path / "work", sdk=sdk)
    assert sdk.calls == []


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        ({"mean": 59.31546920041243}, {"mean": 59.3154692004}, True),
        ({"values": [1.23456789123, 0.000000001]}, {"values": [1.234567891, 0.0]}, True),
        ({"mean": 59.31546920041243}, {"mean": 59.32}, False),
        ({"mean": 1.0}, {"mean": 1.00001}, False),
        ({"mean": 0.0}, {"mean": 0.00000002}, False),
        ({"count": 66}, {"count": 66.00000001}, False),
        ({"count": 1}, {"count": True}, False),
        ({"matched": True}, {"matched": 1}, False),
        ({"x": 1.0}, {"x": float("nan")}, False),
        ({"x": 1.0}, {"x": float("inf")}, False),
        ({"x": 1.0}, {"x": 1.0, "extra": True}, False),
        ({"nested": {"x": 1.0}}, {"nested": {}}, False),
        ({"values": [1.0, 2.0]}, {"values": [1.0]}, False),
        ({"values": [1.0]}, {"values": ["1.0"]}, False),
        ({"text": "original"}, {"text": "changed"}, False),
        ({"value": None}, {"value": 0}, False),
    ],
)
def test_server_summary_tolerates_only_finite_float_roundoff(
    expected: Any, actual: Any, matches: bool
) -> None:
    assert upload._equal_summary(expected, actual) is matches


def test_sdk_dotted_summary_and_media_wrappers_verify(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSDK()

    def flush() -> None:
        dotted = upload.normalize_summary(sdk.remote.summary)
        dotted["evaluation/start.mean_return"] += 1e-8
        for key in upload.MEDIA_KEYS:
            dotted[key] = SimpleNamespace(_json_dict=dotted[key])
        # Real pinned SDK public summaries do not implement Mapping.
        sdk.remote.summary = SimpleNamespace(_json_dict=dotted)

    monkeypatch.setattr(sdk, "flush", flush)
    receipt = upload_completed_experiment(source, work_dir=tmp_path / "work", sdk=sdk)
    assert receipt["server_verified"]


def test_extra_server_summary_field_is_not_hidden_by_flattening(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeSDK()

    def flush() -> None:
        sdk.remote.summary["evaluation/start.unexpected"] = {}

    monkeypatch.setattr(sdk, "flush", flush)
    with pytest.raises(RuntimeError, match="summary differs"):
        upload_completed_experiment(source, work_dir=tmp_path / "work", sdk=sdk)
