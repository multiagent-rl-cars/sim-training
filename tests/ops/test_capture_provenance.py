"""Fail-closed source enumeration without reading real credentials or running Git."""

import errno
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def capture() -> ModuleType:
    script = Path(__file__).resolve().parents[2] / "scripts/capture_provenance.py"
    spec = importlib.util.spec_from_file_location("capture_provenance_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path: Path, capture: ModuleType) -> Path:
    for relative in capture.ROOT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic root source\n")
    for relative, content in {
        "src/rc_car/model.py": "# synthetic source\n",
        "src/rc_car/py.typed": "",
        "experiments/local-ops-plan.md": "# Local experiment plan\n",
        ".github/workflows/tests.yml": "name: cpu\n",
        "docs/notes.md": "# Notes\n",
        "source/car_training/task.py": "# Task\n",
        "tests/models/fixtures/upstream/LICENSE": "synthetic license\n",
        "outputs/result.json": "not source",
        "scratch/local_report.py": "not committed source",
        "generated_assets/car.usd": "not source",
        "src/rc_car/weights.pt": "binary excluded",
        "src/rc_car/outputs/result.json": "nested output excluded",
        "src/rc_car/scratch/local_probe.py": "nested scratch excluded",
        "src/rc_car/__pycache__/cached.py": "cache excluded",
        "src/rc_car/package.egg-info/info.json": "metadata excluded",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def test_explicit_source_selection_and_absent_roots(source: Path, capture: ModuleType) -> None:
    paths, absent = capture.select_source_files(source)
    names = {path.as_posix() for path in paths}
    assert names == set(capture.ROOT_FILES) | {
        "src/rc_car/model.py",
        "src/rc_car/py.typed",
        "experiments/local-ops-plan.md",
        ".github/workflows/tests.yml",
        "docs/notes.md",
        "source/car_training/task.py",
        "tests/models/fixtures/upstream/LICENSE",
    }
    assert "scripts" in absent and "experiments" not in absent
    assert paths == sorted(paths)


def test_unreadable_directory_cannot_silently_disappear(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    inaccessible = source / "docs/private-subtree"
    inaccessible.mkdir()
    original = os.scandir

    def scandir(path: Any) -> Any:
        if Path(path) == inaccessible:
            raise PermissionError(errno.EACCES, "synthetic inaccessible directory", str(path))
        return original(path)

    monkeypatch.setattr(capture.os, "scandir", scandir)
    with pytest.raises(PermissionError):
        capture.select_source_files(source)


def test_stat_failure_propagates_even_for_excluded_suffix(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = source / "src/rc_car/weights.pt"
    original = Path.lstat

    def lstat(path: Path) -> Any:
        if path == target:
            raise PermissionError("synthetic stat error")
        return original(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(PermissionError):
        capture.select_source_files(source)


@pytest.mark.parametrize("kind", ["file", "directory", "root", "required", "dangling"])
def test_symlinks_are_rejected_without_reading_targets(
    source: Path, capture: ModuleType, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = source / "outside-source.txt"
    outside.write_text("synthetic do-not-read marker")
    if kind == "required":
        link = source / "README.md"
        link.unlink()
    else:
        link = source / ("scripts" if kind == "root" else f"docs/linked-{kind}.py")
    target = source / "nonexistent" if kind == "dangling" else outside
    if kind in {"directory", "root"}:
        target = source / "src"
    link.symlink_to(target, target_is_directory=kind in {"directory", "root"})
    monkeypatch.setattr(
        Path, "read_bytes", lambda _path: pytest.fail("Enumeration must not read bytes")
    )
    with pytest.raises(ValueError, match="symlinks"):
        capture.select_source_files(source)


def test_special_file_rejected_without_blocking(source: Path, capture: ModuleType) -> None:
    os.mkfifo(source / "docs/stream.json")
    with pytest.raises(ValueError, match="regular source file"):
        capture.select_source_files(source)


@pytest.mark.parametrize("directory", ["source", "docs"])
def test_symlinked_ancestor_of_declared_path_rejected(
    source: Path, capture: ModuleType, directory: str
) -> None:
    original = source / directory
    target = source / f"moved-{directory}"
    original.rename(target)
    original.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        capture.select_source_files(source)


def test_credential_like_directory_not_entered(source: Path, capture: ModuleType) -> None:
    (source / "docs/secrets").mkdir()
    with pytest.raises(ValueError, match="Credential-like directory"):
        capture.select_source_files(source)


def test_missing_required_root_file_fails(source: Path, capture: ModuleType) -> None:
    (source / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        capture.select_source_files(source)


@pytest.mark.parametrize(
    "name", ["credentials.json", "secrets.yaml", ".env", ".env.local", "id.pem"]
)
def test_credential_like_names_fail_without_reading(
    source: Path, capture: ModuleType, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    (source / "docs" / name).write_text("synthetic credential fixture; not a real key")
    monkeypatch.setattr(
        Path, "read_bytes", lambda _path: pytest.fail("Must not read credential content")
    )
    with pytest.raises(ValueError, match="Credential-like"):
        capture.select_source_files(source)


def test_ignored_tree_not_entered(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.scandir

    def scandir(path: Any) -> Any:
        if Path(path).name in {"outputs", "__pycache__", "package.egg-info"}:
            pytest.fail("Excluded output/cache tree was entered")
        return original(path)

    monkeypatch.setattr(capture.os, "scandir", scandir)
    capture.select_source_files(source)


def _cli(source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.chdir(source)
    monkeypatch.setattr(sys, "argv", ["capture_provenance.py", *args])

    def git(command: list[str], **kwargs: Any) -> str:
        assert kwargs["timeout"] == 10
        if command[1:] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[1:] == ["branch", "--show-current"]:
            return "astra-cleanup\n"
        return " M scripts/capture_provenance.py\n"

    monkeypatch.setattr(capture.subprocess, "check_output", git)
    capture.main()


def test_existing_cli_manifest_and_archive_contract(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cli(source, capture, monkeypatch, "--archive", "outputs/source.tar.gz")
    manifest = json.loads((source / "outputs/source-provenance.json").read_text())
    assert manifest["branch"] == "astra-cleanup"
    assert manifest["source_selection"]["absent_optional_roots"]
    with tarfile.open(source / "outputs/source.tar.gz") as archive:
        assert set(archive.getnames()) == set(manifest["files_sha256"]) | {"source-provenance.json"}
        for name, digest in manifest["files_sha256"].items():
            stream = archive.extractfile(name)
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == digest


def test_failed_enumeration_publishes_no_manifest(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    (source / "docs/bad.md").symlink_to(source / "README.md")
    with pytest.raises(ValueError, match="symlinks"):
        _cli(source, capture, monkeypatch)
    assert not (source / "outputs/source-provenance.json").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("--output", "README.md"),
        ("--archive", "README.md"),
        ("--archive", "outputs/source-provenance.json"),
    ],
)
def test_output_cannot_clobber_source_or_manifest(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...]
) -> None:
    original = (source / "README.md").read_bytes()
    with pytest.raises(SystemExit):
        _cli(source, capture, monkeypatch, *args)
    assert (source / "README.md").read_bytes() == original


@pytest.mark.parametrize("option", ["--output", "--archive"])
def test_capture_does_not_overwrite_symlink_target(
    source: Path, capture: ModuleType, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    target = source / "outside-source.txt"
    target.write_text("preserve unrelated file")
    link = source / "outputs/link"
    link.symlink_to(target)
    with pytest.raises(SystemExit):
        _cli(source, capture, monkeypatch, option, str(link))
    assert target.read_text() == "preserve unrelated file"
