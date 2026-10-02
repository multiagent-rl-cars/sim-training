"""Guard user-facing repository simplicity, not simulator behavior."""

import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("assets/tracks/intersection.usda", False),
        ("source/car_training/assets/car.usd", False),
        ("assets/car.usdc", False),
        ("assets/car.usdz", False),
        ("generated_assets/track.usd", True),
        ("outputs/run/track.usd", True),
        ("outputs/run/video.mp4", True),
        ("scratch/publish_review.py", True),
    ],
)
def test_asset_ignore_policy(tmp_path: Path, path: str, ignored: bool) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((ROOT / ".gitignore").read_text())
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", path],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if ignored else 1), result.stderr


def test_template_changelog_is_not_present() -> None:
    assert not (ROOT / "source/car_training/docs/CHANGELOG.rst").exists()


def test_typed_package_marker_is_retained() -> None:
    assert (ROOT / "src/rc_car/py.typed").is_file()


def test_one_dependency_manifest_without_local_framework_sources() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/rc_car",
        "source/car_training/car_training",
    ]
    for entries in project["tool"]["uv"]["sources"].values():
        for entry in entries if isinstance(entries, list) else [entries]:
            assert "path" not in entry
            assert "editable" not in entry
    for obsolete in (
        "environments/isaac/pyproject.toml",
        "environments/isaac/uv.lock",
        "source/car_training/pyproject.toml",
        "source/car_training/setup.py",
    ):
        assert not (ROOT / obsolete).exists()


@pytest.mark.parametrize("isaac", [False, True])
def test_locked_dependency_options_select_only_requested_accelerator(
    tmp_path: Path, isaac: bool
) -> None:
    command = [
        "uv",
        "export",
        "--frozen",
        "--offline",
        "--no-header",
        "--no-hashes",
        "--no-annotate",
        "--no-emit-project",
    ]
    if isaac:
        command += ["--no-group", "cpu", "--extra", "isaac"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "cache")},
        check=True,
        capture_output=True,
        text=True,
    )
    linux = {"sys_platform": "linux", "platform_machine": "x86_64", "python_version": "3.11"}
    requirements = [Requirement(line) for line in result.stdout.splitlines() if line.strip()]
    selected = {
        item.name: str(item.specifier)
        for item in requirements
        if item.marker is None or item.marker.evaluate(linux)
    }
    assert selected["stable-baselines3"] == "==2.7.0"
    assert selected["torch"] == ("==2.7.0+cu128" if isaac else "==2.7.0+cpu")
    if isaac:
        assert selected["isaaclab"] == "==2.3.0"
        assert selected["isaacsim"] == "==5.1.0.0"
        assert selected["torchvision"] == "==0.22.0+cu128"
    else:
        assert not any(name.startswith(("isaac", "nvidia-")) for name in selected)
        assert "triton" not in selected


def test_active_docs_use_the_supported_workflow() -> None:
    for relative in (
        "README.md",
        "docs/operations.md",
        "docs/architecture.md",
        "docs/merge-review.template.html",
    ):
        text = (ROOT / relative).read_text()
        assert "python -m" not in text, relative
        assert "environments/isaac" not in text, relative
        assert "two lockfiles" not in text, relative
        assert "independent lockfiles" not in text, relative
    assert "uv run rc-car" in (ROOT / "README.md").read_text()


def test_committed_workers_have_no_scratch_or_dated_session_dependencies() -> None:
    for name in (
        "bundle.py",
        "check_online_tracking.py",
        "inspect_results.py",
        "prepare_session.py",
        "sync_wandb_backup.py",
    ):
        path = ROOT / "scripts" / name
        text = path.read_text()
        assert "20260927" not in text, path
        assert "scratch/" not in text, path
    assert not (ROOT / "ops/README.md").exists()
    assert not (ROOT / "ops/run_gpu_checks.sh").exists()
