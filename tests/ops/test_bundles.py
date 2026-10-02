"""Local-only archive integrity and safe-publication regression tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
from pathlib import Path
from typing import Any

import pytest

from rc_car.ops import bundles
from rc_car.ops.bundles import (
    BundleManifest,
    BundlePaths,
    FileEntry,
    create_bundle,
    extract_bundle,
    inventory_files,
    load_manifest,
    verify_bundle,
    verify_directory,
)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    (root / "data.txt").write_bytes(b"data\x00\xff")
    (root / "nested/run.sh").write_bytes(b"#!/bin/sh\nexit 0\n")
    (root / "nested/run.sh").chmod(0o755)
    # Unlisted sensitive-looking files must not be opened or included.
    (root / ".env").write_text("synthetic-not-for-backup")
    return root


def test_bundle_round_trip_preserves_allowlisted_bytes_and_executable_mode(
    source: Path, tmp_path: Path
) -> None:
    paths = create_bundle(source, ["nested/run.sh", "data.txt"], tmp_path / "bundle")
    manifest = verify_bundle(paths.archive, paths.manifest)
    assert [entry.path for entry in manifest.files] == ["data.txt", "nested/run.sh"]
    assert manifest == load_manifest(paths.manifest)
    restored = tmp_path / "restored"
    assert extract_bundle(paths.archive, paths.manifest, restored) == manifest
    assert (restored / "data.txt").read_bytes() == (source / "data.txt").read_bytes()
    assert (restored / "nested/run.sh").read_bytes() == (source / "nested/run.sh").read_bytes()
    assert stat.S_IMODE((restored / "nested/run.sh").stat().st_mode) == 0o755
    assert not (restored / ".env").exists()
    assert verify_directory(restored, paths.manifest) == manifest


def test_directory_verification_allows_unlisted_runtime_files(source: Path, tmp_path: Path) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    (source / "generated.bin").write_bytes(b"not part of source identity")
    (source / "unlisted-link").symlink_to(tmp_path / "does-not-exist")
    verify_directory(source, paths.manifest)
    (source / "data.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_directory(source, paths.manifest)


@pytest.mark.parametrize(
    "paths", [[], ["data.txt", "data.txt"], ["."], ["../outside"], ["/absolute"], ["nested"]]
)
def test_inventory_requires_an_explicit_regular_file_allowlist(
    source: Path, paths: list[str]
) -> None:
    with pytest.raises((ValueError, OSError)):
        inventory_files(source, paths)


@pytest.mark.parametrize("intermediate", [False, True])
def test_inventory_rejects_file_and_directory_symlinks(
    source: Path, tmp_path: Path, intermediate: bool
) -> None:
    if intermediate:
        (source / "alias").symlink_to(source / "nested", target_is_directory=True)
        member = "alias/run.sh"
    else:
        (source / "alias").symlink_to(source / "data.txt")
        member = "alias"
    with pytest.raises((ValueError, OSError)):
        create_bundle(source, [member], tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_inventory_rejects_fifo_without_blocking(source: Path) -> None:
    os.mkfifo(source / "pipe")
    with pytest.raises(ValueError, match="regular"):
        inventory_files(source, ["pipe"])


def test_no_existing_outputs_are_overwritten(source: Path, tmp_path: Path) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    before = paths.archive.read_bytes()
    with pytest.raises(FileExistsError):
        create_bundle(source, ["data.txt"], paths.archive.parent)
    assert paths.archive.read_bytes() == before
    destination = tmp_path / "existing-empty-directory"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        extract_bundle(paths.archive, paths.manifest, destination)
    assert list(destination.iterdir()) == []


def test_broken_destination_symlink_is_not_overwritten(source: Path, tmp_path: Path) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    destination = tmp_path / "link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        extract_bundle(paths.archive, paths.manifest, destination)
    assert destination.is_symlink()


def forged_bundle(
    tmp_path: Path,
    members: list[tuple[str, bytes, bytes]],
    *,
    mode: int = 0o644,
) -> BundlePaths:
    """Construct independent hostile archives with a matching outer checksum.

    Matching archive bytes alone must not bypass member-set/type/content checks.
    The manifest always expects one ordinary data.txt containing b'expected'.
    """
    archive = tmp_path / "forged.tar.gz"
    manifest_path = tmp_path / "forged.json"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type, info.mode = kind, mode
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../outside"
            elif kind == tarfile.REGTYPE:
                info.size = len(data)
            stream.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    manifest = BundleManifest(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files=(
            FileEntry(
                path="data.txt", size=8, sha256=hashlib.sha256(b"expected").hexdigest(), mode=0o644
            ),
        ),
    )
    manifest_path.write_text(manifest.model_dump_json())
    return BundlePaths(archive, manifest_path)


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([], "missing"),
        (
            [("data.txt", b"expected", tarfile.REGTYPE), ("extra.txt", b"extra", tarfile.REGTYPE)],
            "Unexpected",
        ),
        ([("data.txt", b"expected", tarfile.REGTYPE)] * 2, "Duplicate"),
        ([("data.txt", b"modified", tarfile.REGTYPE)], "checksum"),
        ([("data.txt", b"short", tarfile.REGTYPE)], "metadata"),
        ([("data.txt", b"", tarfile.SYMTYPE)], "regular"),
        ([("data.txt", b"", tarfile.LNKTYPE)], "regular"),
        ([("data.txt", b"", tarfile.DIRTYPE)], "regular"),
        ([("data.txt", b"", tarfile.FIFOTYPE)], "regular"),
    ],
)
def test_archive_members_checked_even_with_matching_archive_hash(
    tmp_path: Path, members: list[tuple[str, bytes, bytes]], message: str
) -> None:
    paths = forged_bundle(tmp_path, members)
    destination = tmp_path / "never-created"
    with pytest.raises(ValueError, match=message):
        extract_bundle(paths.archive, paths.manifest, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "./data.txt",
        "a/../data.txt",
        "a//data.txt",
        "a\\data.txt",
        "C:/data.txt",
    ],
)
def test_noncanonical_or_traversal_archive_paths_are_rejected(tmp_path: Path, name: str) -> None:
    paths = forged_bundle(tmp_path, [(name, b"expected", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="relative POSIX"):
        verify_bundle(paths.archive, paths.manifest)


def test_mode_tampering_is_rejected(tmp_path: Path) -> None:
    paths = forged_bundle(tmp_path, [("data.txt", b"expected", tarfile.REGTYPE)], mode=0o755)
    with pytest.raises(ValueError, match="metadata"):
        verify_bundle(paths.archive, paths.manifest)


def test_compressed_archive_corruption_is_rejected(source: Path, tmp_path: Path) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    paths.archive.write_bytes(paths.archive.read_bytes()[:-8])
    with pytest.raises(ValueError, match="archive checksum"):
        verify_bundle(paths.archive, paths.manifest)


@pytest.mark.parametrize(
    "change", ["duplicate", "traversal", "unknown", "bool-size", "parent-conflict"]
)
def test_malformed_manifest_rejected_before_archive_access(tmp_path: Path, change: str) -> None:
    entry: dict[str, Any] = {"path": "data.txt", "size": 8, "sha256": "0" * 64, "mode": 0o644}
    data: dict[str, Any] = {"schema_version": 1, "archive_sha256": "0" * 64, "files": [entry]}
    if change == "duplicate":
        data["files"].append(dict(entry))
    elif change == "traversal":
        entry["path"] = "../outside"
    elif change == "unknown":
        data["unrecognized"] = "synthetic-private-value"
    elif change == "bool-size":
        entry["size"] = True
    else:
        data["files"].append({**entry, "path": "data.txt/nested"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_bundle(tmp_path / "not-opened.tar.gz", manifest)


def test_duplicate_manifest_json_keys_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"files": [], "files": []}')
    with pytest.raises(ValueError, match="Duplicate bundle manifest key"):
        load_manifest(manifest)


def test_tree_verification_rejects_replaced_source_symlink(source: Path, tmp_path: Path) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    (source / "renamed.txt").write_bytes((source / "data.txt").read_bytes())
    (source / "data.txt").unlink()
    (source / "data.txt").symlink_to(source / "renamed.txt")
    with pytest.raises((ValueError, OSError)):
        verify_directory(source, paths.manifest)


def test_failed_staged_recheck_leaves_no_destination(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    inspect = bundles._inspect_archive

    def fail_stage(archive: Path, manifest: BundleManifest, stage: Path | None = None) -> None:
        if stage is not None:
            raise ValueError("archive changed before staging")
        inspect(archive, manifest)

    monkeypatch.setattr(bundles, "_inspect_archive", fail_stage)
    destination = tmp_path / "not-published"
    with pytest.raises(ValueError, match="changed"):
        extract_bundle(paths.archive, paths.manifest, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle-extract-*"))


@pytest.mark.parametrize("operation", ["create", "extract"])
def test_cleanup_failure_preserves_original_error(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "valid-bundle")
    original = RuntimeError("original operation failure")
    output = tmp_path / "failed-output"
    original_rmtree = bundles.shutil.rmtree

    def fail_operation(*_args: Any, **_kwargs: Any) -> None:
        raise original

    def fail_cleanup(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path) == output:
            raise PermissionError("synthetic private cleanup details")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(bundles.shutil, "rmtree", fail_cleanup)
    if operation == "create":
        monkeypatch.setattr(bundles.tarfile.TarFile, "addfile", fail_operation)
    else:
        monkeypatch.setattr(bundles.shutil, "copyfileobj", fail_operation)
    with pytest.raises(RuntimeError) as caught:
        if operation == "create":
            create_bundle(source, ["data.txt"], output)
        else:
            extract_bundle(paths.archive, paths.manifest, output)
    assert caught.value is original
    assert "PermissionError" in original.__notes__[0]
    assert "synthetic private cleanup details" not in " ".join(original.__notes__)
    assert not list(tmp_path.glob(".bundle-extract-*"))


def test_staging_cleanup_failure_does_not_mask_original_error(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = create_bundle(source, ["data.txt"], tmp_path / "bundle")
    original = RuntimeError("original copy failure")
    cleanup = bundles.tempfile.TemporaryDirectory.cleanup

    def fail_copy(*_args: Any, **_kwargs: Any) -> None:
        raise original

    def fail_cleanup(temporary: Any) -> None:
        cleanup(temporary)
        raise OSError("staging cleanup failure")

    monkeypatch.setattr(bundles.shutil, "copyfileobj", fail_copy)
    monkeypatch.setattr(bundles.tempfile.TemporaryDirectory, "cleanup", fail_cleanup)
    with pytest.raises(RuntimeError) as caught:
        extract_bundle(paths.archive, paths.manifest, tmp_path / "restored")
    assert caught.value is original
    assert "remove extraction staging" in original.__notes__[0]
    assert not (tmp_path / "restored").exists()
