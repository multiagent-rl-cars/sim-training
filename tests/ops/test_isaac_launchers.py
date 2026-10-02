"""Exercise actual runtime setup/launch scripts with fake tools, never a GPU."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch_host(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "copied repository"
    (root / "scripts").mkdir(parents=True)
    for name in ("setup_isaac.sh", "isaac.sh"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    tool = (
        f"#!{sys.executable}\n"
        + r"""
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": name, "args": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
if name == "uname":
    print(os.environ.get("FAKE_PLATFORM", "Linux") if sys.argv[1] == "-s"
          else os.environ.get("FAKE_ARCH", "x86_64"))
elif name == "uv":
    if sys.argv[1] == "sync" and not os.environ.get("FAKE_UV_EXIT"):
        python = Path(".venv/bin/python")
        python.parent.mkdir(parents=True)
        python.write_text(Path(__file__).read_text())
        python.chmod(0o755)
    if sys.argv[1:3] == ["pip", "check"]:
        sys.exit(int(os.environ.get("FAKE_PIP_CHECK_EXIT", "0")))
    sys.exit(int(os.environ.get("FAKE_UV_EXIT", "0")))
elif name == "python":
    sys.exit(int(os.environ.get("FAKE_RUNTIME_EXIT", "0")))
else:
    sys.exit(91)
"""
    )
    for name in ("uname", "git", "uv"):
        command = fake_bin / name
        command.write_text(tool)
        command.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f"{fake_bin}:{env['PATH']}", FAKE_CALLS=str(tmp_path / "calls.jsonl"))
    return root, env


def run(root: Path, env: dict[str, str], name: str, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / name), *arguments],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def calls(env: dict[str, str], name: str) -> list[dict]:
    return [
        entry
        for line in Path(env["FAKE_CALLS"]).read_text().splitlines()
        if (entry := json.loads(line))["name"] == name
    ]


@pytest.mark.parametrize("name", ["setup_isaac.sh", "isaac.sh"])
@pytest.mark.parametrize("platform,arch", [("Darwin", "arm64"), ("Linux", "aarch64")])
def test_unsupported_host_stops_before_uv_or_runtime_changes(
    launch_host: tuple[Path, dict[str, str]], name: str, platform: str, arch: str
) -> None:
    root, env = launch_host
    env.update(FAKE_PLATFORM=platform, FAKE_ARCH=arch)
    result = run(root, env, name)
    assert result.returncode == 2
    assert "Linux x86_64" in result.stderr
    assert calls(env, "git") == []
    assert calls(env, "uv") == []
    assert not (root / ".isaaclab").exists()
    assert not (root / ".venv").exists()


def test_setup_selects_root_isaac_dependencies_and_verifies_installed_bundle(
    launch_host: tuple[Path, dict[str, str]],
) -> None:
    root, env = launch_host
    result = run(root, env, "setup_isaac.sh")
    assert result.returncode == 0, result.stderr
    assert calls(env, "git") == []
    assert not (root / ".isaaclab").exists()
    assert [call["args"] for call in calls(env, "uv")] == [
        ["sync", "--locked", "--no-group", "cpu", "--extra", "isaac"],
        ["pip", "check", "--python", ".venv/bin/python"],
    ]
    assert all(call["cwd"] == str(root) for call in calls(env, "uv"))
    assert calls(env, "python") == [
        {
            "name": "python",
            "args": [
                "-c",
                "from rc_car.provenance import inspect_isaaclab_runtime; inspect_isaaclab_runtime()",
            ],
            "cwd": str(root),
        }
    ]


@pytest.mark.parametrize("variable", ["FAKE_PIP_CHECK_EXIT", "FAKE_RUNTIME_EXIT"])
def test_setup_propagates_failed_dependency_or_runtime_verification(
    launch_host: tuple[Path, dict[str, str]], variable: str
) -> None:
    root, env = launch_host
    env[variable] = "23"
    assert run(root, env, "setup_isaac.sh").returncode == 23
    if variable == "FAKE_PIP_CHECK_EXIT":
        assert calls(env, "python") == []


@pytest.mark.parametrize("name", ["setup_isaac.sh", "isaac.sh"])
def test_uv_failure_is_not_hidden(launch_host: tuple[Path, dict[str, str]], name: str) -> None:
    root, env = launch_host
    env["FAKE_UV_EXIT"] = "19"
    assert run(root, env, name).returncode == 19
    assert len(calls(env, "uv")) == 1


def test_launcher_selects_same_isaac_dependencies_and_preserves_arguments(
    launch_host: tuple[Path, dict[str, str]],
) -> None:
    root, env = launch_host
    arguments = ["script with spaces.py", "--example", "one value", "$(not-executed)"]
    result = run(root, env, "isaac.sh", *arguments)
    assert result.returncode == 0, result.stderr
    assert calls(env, "git") == []
    assert calls(env, "uv") == [
        {
            "name": "uv",
            "args": [
                "run",
                "--locked",
                "--no-sync",
                "--no-group",
                "cpu",
                "--extra",
                "isaac",
                "python",
                *arguments,
            ],
            "cwd": str(root),
        }
    ]
