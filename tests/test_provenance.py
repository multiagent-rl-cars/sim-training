"""Source identities must survive copying and reject stale or redirected files."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from rc_car.provenance import (
    IsaacLabRuntimeIdentity,
    inspect_checkout,
    inspect_isaaclab_runtime,
    verify_experiment_identity,
    verify_source_files,
)


def test_verify_source_bytes_and_detect_changes(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("original\n")
    expected = {"example.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_source_files(tmp_path, expected)
    # Additional output is explicitly outside this file-integrity contract.
    (tmp_path / "extra.txt").write_text("unlisted\n")
    verify_source_files(tmp_path, expected)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch: example.py"):
        verify_source_files(tmp_path, expected)


@pytest.mark.parametrize(
    "relative", ["", ".", "../outside", "/outside", "a/../b", "./b", "a//b", "a\\b"]
)
def test_noncanonical_manifest_paths_rejected(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        verify_source_files(tmp_path, {relative: "a" * 64})


def test_empty_invalid_and_missing_manifest_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        verify_source_files(tmp_path, {})
    with pytest.raises(ValueError, match="invalid path or SHA-256"):
        verify_source_files(tmp_path, {"x": "not-a-hash"})
    with pytest.raises(ValueError, match="missing or not a regular file"):
        verify_source_files(tmp_path, {"x": "a" * 64})
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError, match="missing or not a regular file"):
        verify_source_files(tmp_path, {"x": "a" * 64})


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlinks_are_not_source_snapshots(tmp_path: Path, directory_link: bool) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "x.py"
    source.write_text("content")
    if directory_link:
        (tmp_path / "alias").symlink_to(real, target_is_directory=True)
        relative = "alias/x.py"
    else:
        (tmp_path / "alias.py").symlink_to(source)
        relative = "alias.py"
    with pytest.raises(ValueError, match="symlink"):
        verify_source_files(tmp_path, {relative: hashlib.sha256(source.read_bytes()).hexdigest()})


def test_actual_checkout_identity_and_parent_guard(tmp_path: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *arguments], text=True, stderr=subprocess.PIPE
        ).strip()

    git("init", "--initial-branch=example")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--allow-empty",
        "-m",
        "Test fixture",
    )
    clean = inspect_checkout(tmp_path)
    assert clean.commit == git("rev-parse", "HEAD")
    assert clean.branch == "example"
    assert clean.status_porcelain == ""
    (tmp_path / "untracked.py").write_text("content")
    assert "untracked.py" in inspect_checkout(tmp_path).status_porcelain
    child = tmp_path / "nested"
    child.mkdir()
    with pytest.raises(ValueError, match="checkout root"):
        inspect_checkout(child)


def test_missing_checkout_is_not_clean(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_checkout(tmp_path / "missing")
    with pytest.raises(subprocess.CalledProcessError):
        inspect_checkout(tmp_path)


@pytest.fixture
def runtime_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny installed wheel RECORD, never an imported Isaac or GPU runtime."""
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "isaaclab"\nversion = "2.3.0"\n')
    metadata = tmp_path / "isaaclab-2.3.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: isaaclab\nVersion: 2.3.0\n")
    relative_files = [
        "isaaclab/__init__.py",
        "isaaclab/apps/isaaclab.python.headless.kit",
        "isaaclab/apps/isaaclab.python.headless.rendering.kit",
        "isaaclab/source/isaaclab/isaaclab/app/app_launcher.py",
        *[
            f"isaaclab/source/{package}/config/extension.toml"
            for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")
        ],
    ]
    rows = []
    for relative in relative_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Never execute this synthetic package.\n")
        content = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append(f"{relative},sha256={digest},{len(content)}\n")
    (metadata / "RECORD").write_text("".join(rows))
    distribution = importlib.metadata.PathDistribution(metadata)
    monkeypatch.setattr("rc_car.provenance.importlib.metadata.distribution", lambda _: distribution)
    monkeypatch.setattr(
        "rc_car.provenance.importlib.util.find_spec",
        lambda _: importlib.util.spec_from_file_location(
            "isaaclab", tmp_path / "isaaclab/__init__.py"
        ),
    )
    return tmp_path


def test_runtime_bundle_bytes_verified_without_import(runtime_bundle: Path) -> None:
    identity = inspect_isaaclab_runtime(runtime_bundle)
    assert identity.distribution == "isaaclab"
    assert identity.version == "2.3.0"
    assert identity.files_verified == 8
    assert identity.package_root == str(runtime_bundle / "isaaclab")
    assert len(identity.content_sha256) == 64
    bytecode = runtime_bundle / "isaaclab/__pycache__/__init__.cpython-311.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"generated-bytecode-is-not-wheel-source")
    assert inspect_isaaclab_runtime(runtime_bundle) == identity


def test_runtime_requires_locked_version(runtime_bundle: Path) -> None:
    (runtime_bundle / "uv.lock").write_text('[[package]]\nname = "isaaclab"\nversion = "2.3.1"\n')
    with pytest.raises(ValueError, match="version differs from the root lock"):
        inspect_isaaclab_runtime(runtime_bundle)
    (runtime_bundle / "uv.lock").write_text("package = []\n")
    with pytest.raises(ValueError, match="exactly one"):
        inspect_isaaclab_runtime(runtime_bundle)


@pytest.mark.parametrize("change", ["modified", "missing", "symlink", "unrecorded"])
def test_runtime_rejects_damaged_or_added_package_files(runtime_bundle: Path, change: str) -> None:
    target = runtime_bundle / "isaaclab/__init__.py"
    if change == "modified":
        target.write_text("changed\n")
    elif change == "missing":
        target.unlink()
    elif change == "symlink":
        moved = target.with_suffix(".saved")
        target.rename(moved)
        target.symlink_to(moved)
    else:
        target.with_name("extra.py").write_text("unexpected importable module")
    with pytest.raises(ValueError, match="hash mismatch|missing or redirected|unrecorded|shadowed"):
        inspect_isaaclab_runtime(runtime_bundle)


@pytest.mark.parametrize(
    "record_change", ["missing", "empty", "no-hash", "duplicate", "incomplete"]
)
def test_runtime_requires_complete_record(runtime_bundle: Path, record_change: str) -> None:
    record = runtime_bundle / "isaaclab-2.3.0.dist-info/RECORD"
    rows = record.read_text().splitlines(keepends=True)
    if record_change == "missing":
        record.unlink()
    elif record_change == "empty":
        record.write_text("")
    elif record_change == "no-hash":
        record.write_text("isaaclab/__init__.py,,\n" + "".join(rows[1:]))
    elif record_change == "duplicate":
        record.write_text("".join(rows + [rows[0]]))
    else:
        record.write_text("".join(rows[:-1]))
    with pytest.raises(ValueError, match="RECORD|lacks required"):
        inspect_isaaclab_runtime(runtime_bundle)


def test_runtime_rejects_shadowed_import(
    runtime_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rc_car.provenance.importlib.util.find_spec",
        lambda _: importlib.util.spec_from_file_location("isaaclab", runtime_bundle / "shadow.py"),
    )
    with pytest.raises(ValueError, match="shadowed"):
        inspect_isaaclab_runtime(runtime_bundle)


def test_runtime_rejects_special_file_without_reading_it(runtime_bundle: Path) -> None:
    os.mkfifo(runtime_bundle / "isaaclab/unrecorded.py")
    with pytest.raises(ValueError, match="special file"):
        inspect_isaaclab_runtime(runtime_bundle)


def test_runtime_rejects_symlinked_bytecode(runtime_bundle: Path) -> None:
    bytecode = runtime_bundle / "isaaclab/__pycache__/__init__.cpython-311.pyc"
    bytecode.parent.mkdir()
    bytecode.symlink_to(runtime_bundle / "isaaclab/__init__.py")
    with pytest.raises(ValueError, match="symlink"):
        inspect_isaaclab_runtime(runtime_bundle)


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    identity = {"files_sha256": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "source-provenance.json").write_text(json.dumps(identity))
    run = output / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"seed": 1},
                "source": identity,
                "isaaclab_runtime": asdict(
                    IsaacLabRuntimeIdentity("isaaclab", "2.3.0", "/synthetic/bundle", 8, "a" * 64)
                ),
            }
        )
    )
    monkeypatch.setattr(
        "rc_car.provenance.inspect_isaaclab_runtime",
        lambda _: IsaacLabRuntimeIdentity("isaaclab", "2.3.0", "/synthetic/bundle", 8, "a" * 64),
    )
    return tmp_path, run


def test_train_and_matched_evaluations_use_verified_identity(experiment: tuple[Path, Path]) -> None:
    root, run = experiment
    for mode in ("train", "start", "end"):
        identity = verify_experiment_identity(root, run, mode, {"seed": 1})
        assert identity.isaaclab_runtime.version == "2.3.0"
        assert set(identity.source["files_sha256"]) == {"source.py"}


def test_recapture_cannot_disguise_changed_evaluation_source(experiment: tuple[Path, Path]) -> None:
    root, run = experiment
    source = root / "source.py"
    source.write_text("modified\n")
    (root / "outputs/source-provenance.json").write_text(
        json.dumps(
            {
                "files_sha256": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
        )
    )
    verify_experiment_identity(root, run, "train", {"seed": 1})
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_experiment_identity(root, run, "end", {"seed": 1})


def test_evaluation_config_cannot_silently_change(experiment: tuple[Path, Path]) -> None:
    root, run = experiment
    with pytest.raises(ValueError, match="recipe differs"):
        verify_experiment_identity(root, run, "start", {"seed": 2})


def test_runtime_changes_are_explicit(
    experiment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run = experiment
    monkeypatch.setattr(
        "rc_car.provenance.inspect_isaaclab_runtime",
        lambda _: IsaacLabRuntimeIdentity("isaaclab", "2.3.0", "/new/bundle", 8, "b" * 64),
    )
    with pytest.raises(ValueError, match="runtime differs"):
        verify_experiment_identity(root, run, "end", {"seed": 1})
    monkeypatch.setattr(
        "rc_car.provenance.inspect_isaaclab_runtime",
        lambda _: IsaacLabRuntimeIdentity("isaaclab", "2.3.0", "/new/bundle", 8, "a" * 64),
    )
    # The install location may change; the matched runtime bytes may not.
    verify_experiment_identity(root, run, "end", {"seed": 1})


def test_legacy_run_requires_its_frozen_runtime(experiment: tuple[Path, Path]) -> None:
    root, run = experiment
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.pop("isaaclab_runtime")
    manifest["isaaclab_checkout"] = {"commit": "historical"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="legacy checkout runtime"):
        verify_experiment_identity(root, run, "end", {"seed": 1})
