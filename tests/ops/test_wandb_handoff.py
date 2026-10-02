"""Portable handoff-plan tests; no W&B import, credentials, GPU, or network."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rc_car.ops import wandb_handoff as handoff
from rc_car.ops.bundles import BundleManifest, create_bundle, extract_bundle, inventory_files
from rc_car.ops.wandb_handoff import (
    EVALUATION_FILES,
    EVALUATION_MEDIA,
    MODEL_FILES,
    prepare_handoff,
)

IDENTITY = {
    "id": "run123",
    "entity": "test-team",
    "project": "test-project",
    "mode": "offline",
    "url": None,
}
STREAM = "experiment/wandb/offline-run-20260927_120000-run123/run-run123.wandb"
MEDIA = "experiment/wandb/offline-run-20260927_120000-run123/files/media/videos/test.mp4"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "source"
    (root / "experiment").mkdir(parents=True)
    files = ["experiment/" + name for name in (*MODEL_FILES, *EVALUATION_FILES, *EVALUATION_MEDIA)]
    files += ["experiment/wandb_run.json", STREAM, MEDIA]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic fixture: " + name).encode())
    (root / "experiment/wandb_run.json").write_text(json.dumps(IDENTITY))
    (root / "experiment/config.json").write_text(json.dumps({"wandb": IDENTITY}))
    # Unlisted paths must never be inspected or included.
    (root / ".netrc").symlink_to(root / "nonexistent-credential-target")
    return root, files


def manifest(tmp_path: Path, root: Path, files: list[str]) -> Path:
    path = tmp_path / "inventory.json"
    value = BundleManifest(archive_sha256="0" * 64, files=inventory_files(root, files))
    path.write_text(value.model_dump_json())
    return path


@pytest.mark.parametrize("name", [None, "calm-forest-42"])
def test_offline_handoff_accepts_optional_generated_name(
    source: tuple[Path, list[str]], tmp_path: Path, name: str | None
) -> None:
    root, files = source
    identity = {**IDENTITY, "name": name}
    (root / "experiment/wandb_run.json").write_text(json.dumps(identity))
    result = prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")
    assert result["identity"] == identity


def test_plan_preserves_bytes_and_never_imports_wandb(
    source: tuple[Path, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, files = source
    sidecar = manifest(tmp_path, root, files)
    before = {name: (root / name).read_bytes() for name in files}
    monkeypatch.setitem(sys.modules, "wandb", None)
    result = prepare_handoff(root, sidecar, run_directory="experiment")
    assert result["listed_backup_bytes_verified"]
    assert result["identity"] == IDENTITY
    assert result["all_reupload_groups_complete"]
    assert result["reupload_groups"]["model"]["artifact_name"] == "experiment-checkpoints"
    assert result["reupload_groups"]["model"]["artifact_type"] == "model"
    assert result["reupload_groups"]["evaluation"]["artifact_name"] == "experiment-evaluation"
    assert result["reupload_groups"]["evaluation"]["artifact_type"] == "evaluation"
    assert not result["uploaded"] and not result["server_verified"]
    assert {entry["path"] for entry in result["transport_files"]} == {STREAM, MEDIA}
    assert {name: (root / name).read_bytes() for name in files} == before
    assert ".netrc" not in json.dumps(result)
    assert any("not parsed" in limitation for limitation in result["limitations"])
    entry = result["reupload_groups"]["model"]["files"][0]
    content = before["experiment/initial.zip"]
    assert entry == {
        "name": "initial.zip",
        "path": "experiment/initial.zip",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "wandb_content_digest": base64.b64encode(hashlib.md5(content).digest()).decode(),
    }


def test_relocated_verified_bundle_prepares_without_original_paths(
    source: tuple[Path, list[str]], tmp_path: Path
) -> None:
    root, files = source
    paths = create_bundle(root, files, tmp_path / "bundle")
    copied = tmp_path / "different-location"
    extract_bundle(paths.archive, paths.manifest, copied)
    root.rename(tmp_path / "old-source-not-at-original-path")
    result = prepare_handoff(copied, paths.manifest, run_directory="experiment")
    assert result["backup_root"] == str(copied)
    assert result["all_reupload_groups_complete"]
    assert str(root) not in json.dumps(result)


def test_partial_run_keeps_backup_valid_but_does_not_claim_complete_artifacts(
    source: tuple[Path, list[str]], tmp_path: Path
) -> None:
    root, files = source
    files.remove("experiment/final.zip")
    files.remove("experiment/end.mp4")
    result = prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")
    assert result["listed_backup_bytes_verified"]
    assert not result["all_reupload_groups_complete"]
    assert result["reupload_groups"]["model"]["missing"] == ["final.zip"]
    assert result["reupload_groups"]["evaluation_media"]["missing"] == ["end.mp4"]
    assert all(
        entry["name"] != "final.zip" for entry in result["reupload_groups"]["model"]["files"]
    )


@pytest.mark.parametrize(
    "directory", ["", ".", "../experiment", "/experiment", "experiment/", "a//b", "a\\b"]
)
def test_unsafe_run_directory_rejected(
    source: tuple[Path, list[str]], tmp_path: Path, directory: str
) -> None:
    root, files = source
    with pytest.raises(ValueError, match="canonical relative"):
        prepare_handoff(root, manifest(tmp_path, root, files), run_directory=directory)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "online"},
        {"url": "https://example.invalid/run"},
        {"id": "../run123"},
        {"entity": "other-team"},
        {"project": None},
        {"unexpected": "field"},
    ],
)
def test_invalid_or_mismatched_identity_rejected(
    source: tuple[Path, list[str]], tmp_path: Path, change: dict[str, Any]
) -> None:
    root, files = source
    (root / "experiment/wandb_run.json").write_text(json.dumps(IDENTITY | change))
    with pytest.raises(ValueError):
        prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")


def test_duplicate_public_json_rejected(source: tuple[Path, list[str]], tmp_path: Path) -> None:
    root, files = source
    (root / "experiment/config.json").write_text('{"wandb": {}, "wandb": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")


@pytest.mark.parametrize("missing", ["experiment/wandb_run.json", "experiment/config.json", STREAM])
def test_missing_inventory_requirement_rejected(
    source: tuple[Path, list[str]], tmp_path: Path, missing: str
) -> None:
    root, files = source
    files.remove(missing)
    with pytest.raises(ValueError):
        prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")


def test_multiple_streams_rejected(source: tuple[Path, list[str]], tmp_path: Path) -> None:
    root, files = source
    other = STREAM.replace("run-run123.wandb", "run-other.wandb")
    (root / other).write_bytes(b"other")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_handoff(root, manifest(tmp_path, root, [*files, other]), run_directory="experiment")


def test_wrong_stream_identity_rejected(source: tuple[Path, list[str]], tmp_path: Path) -> None:
    root, files = source
    other = STREAM.replace("run-run123.wandb", "run-other.wandb")
    (root / STREAM).rename(root / other)
    files[files.index(STREAM)] = other
    with pytest.raises(ValueError, match="filename/layout"):
        prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")


def test_tamper_and_symlink_rejected(source: tuple[Path, list[str]], tmp_path: Path) -> None:
    root, files = source
    sidecar = manifest(tmp_path, root, files)
    target = root / "experiment/final.zip"
    original = target.read_bytes()
    target.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        prepare_handoff(root, sidecar, run_directory="experiment")
    target.unlink()
    substitute = tmp_path / "outside.zip"
    substitute.write_bytes(original)
    target.symlink_to(substitute)
    with pytest.raises((OSError, ValueError)):
        prepare_handoff(root, sidecar, run_directory="experiment")


@pytest.mark.parametrize("name", ["final.zip", "end.mp4"])
def test_empty_members_do_not_make_reupload_group_complete(
    source: tuple[Path, list[str]], tmp_path: Path, name: str
) -> None:
    root, files = source
    (root / "experiment" / name).write_bytes(b"")
    report = prepare_handoff(root, manifest(tmp_path, root, files), run_directory="experiment")
    assert report["listed_backup_bytes_verified"]
    assert not report["all_reupload_groups_complete"]
    group = "model" if name == "final.zip" else "evaluation_media"
    assert not report["reupload_groups"][group]["complete"]
    assert report["reupload_groups"][group]["empty"] == [name]


@pytest.mark.parametrize(
    "name", [".env.local", "nested/.netrc", "wandb-config/settings", "key.pem"]
)
def test_credential_names_rejected_before_any_member_read(
    source: tuple[Path, list[str]], tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, files = source
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic-not-a-credential")
    sidecar = manifest(tmp_path, root, [*files, name])

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Member read before credential-name guard")

    monkeypatch.setattr(handoff, "_open_regular", forbidden_read)
    with pytest.raises(ValueError, match="Credential-like"):
        prepare_handoff(root, sidecar, run_directory="experiment")


def test_manifest_is_not_reloaded_after_guard(
    source: tuple[Path, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, files = source
    sidecar = manifest(tmp_path, root, files)
    original_load = handoff.load_manifest
    calls = []

    def replace_after_load(path: Path) -> BundleManifest:
        calls.append(path)
        result = original_load(path)
        path.write_text("not the original manifest any more")
        return result

    monkeypatch.setattr(handoff, "load_manifest", replace_after_load)
    report = prepare_handoff(root, sidecar, run_directory="experiment")
    assert report["listed_backup_bytes_verified"]
    assert calls == [sidecar]


def test_cli_prints_plan_only(source: tuple[Path, list[str]], tmp_path: Path) -> None:
    root, files = source
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_wandb_handoff.py"),
            "--backup-root",
            str(root),
            "--manifest",
            str(manifest(tmp_path, root, files)),
            "--run-directory",
            "experiment",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert json.loads(process.stdout)["uploaded"] is False
    assert not process.stderr
