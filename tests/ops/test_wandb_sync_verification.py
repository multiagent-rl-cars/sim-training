"""Saved protobuf stream/server checks using fakes, without network or credentials."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rc_car.ops import wandb_sync as verification

pytest.importorskip("wandb", reason="Saved stream verification needs the optional tracking group")
from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal.datastore import DataStore  # noqa: E402

IDENTITY = {"id": "synthetic123", "entity": "test-team", "project": "test-project"}
Record = cast(Any, pb).Record


def item(container: Any, path: str | list[str], value: Any) -> None:
    record = container.add()
    if isinstance(path, str):
        record.key = path
    else:
        record.nested_key.extend(path)
    record.value_json = json.dumps(value)


def records(video: dict[str, Any]) -> list[Any]:
    run = Record()
    run.run.run_id = IDENTITY["id"]
    run.run.entity = IDENTITY["entity"]
    run.run.project = IDENTITY["project"]
    run.run.host = "ignored-private-host"
    item(run.run.config.update, "ignored-config", "not-in-expectation")
    first = Record()
    item(first.history.item, ["_step"], 0)
    item(first.history.item, ["training/agent_timesteps"], 12)
    item(first.history.item, ["loss"], 0.25)
    item(first.history.item, ["_runtime"], 4.0)
    for key, value in video.items():
        item(first.history.item, ["training/video", key], value)
    second = Record()
    second.history.step.num = 1
    item(second.history.item, "_step", 1)
    item(second.history.item, "training/agent_timesteps", 24)
    item(second.history.item, "loss", 0.125)
    item(second.history.item, "nested", {"value": 3.5})
    summary = Record()
    item(summary.summary.update, "agent_timesteps", 30)
    item(summary.summary.update, ["loss"], 0.125)
    item(summary.summary.update, ["discarded"], 1)
    item(summary.summary.update, ["terminations", "off-track"], 2)
    item(summary.summary.update, ["_wandb", "runtime"], 8)
    remove = Record()
    remove.summary.remove.add(key="discarded")
    exit_record = Record()
    exit_record.exit.exit_code = 0
    return [run, first, second, summary, remove, exit_record]


def write_stream(path: Path, contents: list[Any]) -> None:
    store = DataStore()
    store.open_for_write(str(path))
    try:
        for record in contents:
            store.write(record)
    finally:
        store.close()


@pytest.fixture
def saved(tmp_path: Path) -> tuple[Path, list[Any]]:
    path = tmp_path / "run-synthetic123.wandb"
    video_path = tmp_path / "files/media/videos/training/video.mp4"
    video_path.parent.mkdir(parents=True)
    payload = b"synthetic-media-for-checksum-only-not-a-real-video"
    video_path.write_bytes(payload)
    video = {
        "_type": "video-file",
        "path": "media/videos/training/video.mp4",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    content = records(video)
    write_stream(path, content)
    return path, content


class RemoteRun:
    def __init__(self, expected: dict[str, Any]) -> None:
        self.id = IDENTITY["id"]
        self.entity = IDENTITY["entity"]
        self.project = IDENTITY["project"]
        self.state = "finished"
        self.summary = copy.deepcopy(expected["summary"])
        self.rows = copy.deepcopy(expected["history"])
        self.file_list = [
            SimpleNamespace(name=value["path"], size=value["size"], md5=value["md5"])
            for value in expected["media"]
        ]
        self.scans = 0

    def scan_history(self) -> list[dict[str, Any]]:
        self.scans += 1
        return self.rows

    def files(self) -> list[Any]:
        return self.file_list


def test_reads_real_protobuf_nested_keys_read_only(
    saved: tuple[Path, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = saved
    original = path.read_bytes()
    opened = []
    original_open = Path.open

    def checked_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.is_relative_to(path.parent):
            opened.append(mode)
            assert mode == "rb"
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    result = verification.inspect_stream(path, IDENTITY)
    assert opened and set(opened) == {"rb"}
    assert result["identity"] == IDENTITY
    assert result["source_completed_successfully"]
    assert len(result["history"]) == 2
    assert result["history"][0]["training/video"]["_type"] == "video-file"
    assert result["summary"]["agent_timesteps"] == 30
    assert result["history"][-1]["training/agent_timesteps"] == 24
    assert "discarded" not in result["summary"]
    assert "_wandb" not in result["summary"]
    assert "_runtime" not in result["history"][0]
    assert "ignored-private-host" not in json.dumps(result)
    assert "ignored-config" not in json.dumps(result)
    assert path.read_bytes() == original


def test_valid_server_allows_new_evaluation_rows_and_partial_final_rollout(
    saved: tuple[Path, list[Any]],
) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    remote = RemoteRun(expected)
    remote.rows.append({"_step": 2, "evaluation/score": 10})
    remote.summary["evaluation/score"] = 10
    remote.rows[0]["_runtime"] = 9999
    result = verification.verify_online_run(expected, remote)
    assert result["passed"] and result["server_verified"]
    assert result["expected_history_rows"] == 2
    assert result["server_history_rows"] == 3
    assert result["expected_training_videos"] == 1
    assert not result["errors"]


@pytest.mark.parametrize("field", ["id", "entity", "project"])
def test_wrong_server_identity_fails_without_scanning(
    saved: tuple[Path, list[Any]], field: str
) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    remote = RemoteRun(expected)
    setattr(remote, field, "wrong")
    result = verification.verify_online_run(expected, remote)
    assert not result["passed"]
    assert remote.scans == 0


@pytest.mark.parametrize(
    "change",
    [
        "missing-row",
        "wrong-scalar",
        "duplicate-step",
        "wrong-step",
        "missing-summary",
        "wrong-summary",
        "running",
        "wrong-media-hash",
        "wrong-media-size",
        "wrong-media-path",
        "missing-file",
        "wrong-file-md5",
        "wrong-file-size",
        "integer-as-bool",
        "nonfinite",
    ],
)
def test_server_mismatch_never_claims_verification(
    saved: tuple[Path, list[Any]], change: str
) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    remote = RemoteRun(expected)
    if change == "missing-row":
        remote.rows.pop()
    elif change == "wrong-scalar":
        remote.rows[0]["loss"] = 100
    elif change == "duplicate-step":
        remote.rows.append(remote.rows[0])
    elif change == "wrong-step":
        remote.rows[0]["_step"] = 5
    elif change == "missing-summary":
        del remote.summary["agent_timesteps"]
    elif change == "wrong-summary":
        remote.summary["agent_timesteps"] = 24
    elif change == "running":
        remote.state = "running"
    elif change.startswith("wrong-media"):
        key = {"wrong-media-hash": "sha256", "wrong-media-size": "size", "wrong-media-path": "path"}
        remote.rows[0]["training/video"][key[change]] = "wrong"
    elif change == "missing-file":
        remote.file_list = []
    elif change == "wrong-file-md5":
        remote.file_list[0].md5 = "wrong"
    elif change == "wrong-file-size":
        remote.file_list[0].size = 0
    elif change == "integer-as-bool":
        remote.rows[1]["_step"] = True
    elif change == "nonfinite":
        remote.rows[0]["loss"] = float("nan")
    result = verification.verify_online_run(expected, remote)
    assert not result["passed"] and not result["server_verified"]
    assert result["errors"]


def test_float_tolerance_but_exact_integer_counts(saved: tuple[Path, list[Any]]) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    remote = RemoteRun(expected)
    remote.rows[0]["loss"] += 1e-8
    assert verification.verify_online_run(expected, remote)["passed"]
    remote.summary["agent_timesteps"] += 1e-8
    assert not verification.verify_online_run(expected, remote)["passed"]


def test_server_dotted_scalar_summary_and_sdk_media_wrapper(
    saved: tuple[Path, list[Any]],
) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    expected["summary"]["training/video"] = expected["history"][0]["training/video"]
    remote = RemoteRun(expected)
    scalar_summary = verification.normalize_summary(remote.summary)
    scalar_summary["training/video"] = SimpleNamespace(_json_dict=scalar_summary["training/video"])
    # Real SDK Summary/SubDict need not inherit Mapping or expose a parent key.
    remote.summary = SimpleNamespace(_json_dict=scalar_summary)
    assert verification.verify_online_run(expected, remote)["passed"]
    scalar_summary["terminations.off-track"] = 100
    assert not verification.verify_online_run(expected, remote)["passed"]


def test_summary_normalization_preserves_slashes_media_and_rejects_ambiguity() -> None:
    media = {"_type": "video-file", "path": "media/videos/example.mp4", "size": 12}
    expected = {
        "evaluation/start": {"mean_return": 1, "terminations": {"off-track": 2}},
        "evaluation/video": media,
        "_runtime": 1,
    }
    assert verification.normalize_summary(expected) == {
        "evaluation/start.mean_return": 1,
        "evaluation/start.terminations.off-track": 2,
        "evaluation/video": media,
    }
    with pytest.raises(ValueError, match="Ambiguous"):
        verification.normalize_summary({"a": {"b": 1}, "a.b": 1})
    assert verification.normalize_summary({"empty": {}}) == {"empty": {}}
    assert not verification._compare({}, 1)


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "duplicate-run",
        "duplicate-step",
        "no-exit",
        "failed-exit",
        "bad-step",
        "duplicate-key",
        "nonfinite",
    ],
)
def test_invalid_saved_records_rejected(saved: tuple[Path, list[Any]], change: str) -> None:
    path, content = saved
    if change == "identity":
        content[0].run.entity = "wrong"
    elif change == "duplicate-run":
        content.insert(1, content[0])
    elif change == "duplicate-step":
        content.insert(2, content[1])
    elif change == "no-exit":
        content.pop()
    elif change == "failed-exit":
        content[-1].exit.exit_code = 1
    elif change == "bad-step":
        content[2].history.step.num = 7
    elif change == "duplicate-key":
        item(content[1].history.item, "loss", 0.2)
    elif change == "nonfinite":
        item(content[1].history.item, "invalid", float("nan"))
    changed = path.with_name("changed.wandb")
    write_stream(changed, content)
    with pytest.raises(ValueError):
        verification.inspect_stream(changed, IDENTITY)


@pytest.mark.parametrize("change", ["corrupt", "truncated", "video-corrupt", "video-symlink"])
def test_corruption_or_redirected_media_rejected(
    saved: tuple[Path, list[Any]], change: str
) -> None:
    path, _ = saved
    if change == "corrupt":
        content = bytearray(path.read_bytes())
        content[-1] ^= 0xFF
        path.write_bytes(content)
    elif change == "truncated":
        path.write_bytes(path.read_bytes()[:-3])
    elif change == "video-corrupt":
        (path.parent / "files/media/videos/training/video.mp4").write_bytes(b"changed")
    else:
        video = path.parent / "files/media/videos/training/video.mp4"
        target = video.with_suffix(".target")
        video.rename(target)
        video.symlink_to(target)
    with pytest.raises(ValueError):
        verification.inspect_stream(path, IDENTITY)


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/video.mp4",
        "../video.mp4",
        "media/videos/../video.mp4",
        "media/videos//video.mp4",
        "media/videos/a\\b.mp4",
    ],
)
def test_unsafe_video_path_rejected(saved: tuple[Path, list[Any]], name: str) -> None:
    path, content = saved
    for entry in content[1].history.item:
        if list(entry.nested_key) == ["training/video", "path"]:
            entry.value_json = json.dumps(name)
    changed = path.with_name("unsafe.wandb")
    write_stream(changed, content)
    with pytest.raises(ValueError, match="Unsafe"):
        verification.inspect_stream(changed, IDENTITY)


def test_sdk_version_is_explicitly_pinned(
    saved: tuple[Path, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verification.importlib.metadata, "version", lambda _: "9.9.9")
    with pytest.raises(RuntimeError, match="0.25.1"):
        verification.inspect_stream(saved[0], IDENTITY)


def test_api_error_propagates_without_success_receipt(
    saved: tuple[Path, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = verification.inspect_stream(saved[0], IDENTITY)
    remote = RemoteRun(expected)

    def fail() -> Any:
        raise RuntimeError("synthetic server unavailable")

    monkeypatch.setattr(remote, "scan_history", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        verification.verify_online_run(expected, remote)
