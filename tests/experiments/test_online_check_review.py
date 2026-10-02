"""Independent no-network regressions for tracking preflight and launch ordering."""

from __future__ import annotations

import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from rc_car.experiments import online_check
from rc_car.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "tests/fixtures/legacy_recipes/basic_smoke.json"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-review-secret")
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent-netrc"))
    transport = MagicMock(
        return_value=SimpleNamespace(
            status_code=200,
            json=lambda: {"data": {"viewer": {"id": "a"}, "project": {"name": "sim-training"}}},
        )
    )
    monkeypatch.setattr(requests, "post", transport)
    return transport


def test_explicit_online_overrides_offline_fixture_recipe_without_heavy_imports(
    no_network: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_config(RECIPE)
    assert recipe.wandb is not None and recipe.wandb.mode == "offline"
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".", 1)[0] not in {"wandb", "torch", "isaaclab", "isaacsim", "omni"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert online_check.check_online(recipe)["online_verified"] is True
    no_network.assert_called_once()


def test_optional_tracking_never_reads_credentials_or_calls_transport(
    no_network: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_config(RECIPE).model_copy(update={"wandb": None})
    monkeypatch.setattr(online_check, "configured_key", MagicMock(side_effect=AssertionError))
    assert online_check.check_online(recipe) == {
        "status": "tracking-not-configured",
        "online_verified": False,
    }
    no_network.assert_not_called()


@pytest.mark.parametrize("failure", ["http", "transport", "payload", "netrc", "mode"])
def test_cli_failures_do_not_print_credentials_or_private_server_text(
    no_network: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    private_text = "synthetic-review-secret-private-server-text"
    if failure == "http":
        no_network.return_value.status_code = 403
    elif failure == "transport":
        no_network.side_effect = requests.Timeout(private_text)
    elif failure == "payload":
        no_network.return_value.json = lambda: {"errors": [private_text]}
    elif failure == "netrc":
        monkeypatch.delenv("WANDB_API_KEY")
        path = tmp_path / "invalid-netrc"
        path.write_text(private_text)
        monkeypatch.setenv("NETRC", str(path))
    else:
        monkeypatch.setenv("WANDB_MODE", private_text)
    monkeypatch.setattr(sys, "argv", ["online-check", "--config", str(RECIPE)])
    assert online_check.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WANDB_MODE=offline" in captured.err
    assert "synthetic-review-secret" not in captured.err
    assert private_text not in captured.err
    if failure in {"netrc", "mode"}:
        no_network.assert_not_called()


@pytest.fixture
def runner(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "outputs").mkdir()
    shutil.copy2(ROOT / "scripts/run_experiment.sh", root / "scripts/run_experiment.sh")
    (root / "outputs/source-provenance.json").write_text("{}")
    script = root / "scripts/isaac.sh"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if 'rc_car.experiments.online_check' in sys.argv:\n"
        "    sys.exit(int(os.environ.get('FAKE_PREFLIGHT_EXIT', '0')))\n"
    )
    script.chmod(0o755)
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text("#!/usr/bin/env bash\nexit 1\n")
    git.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "FAKE_CALLS": str(root / "calls.jsonl"),
        }
    )
    return root, env


def test_standalone_online_failure_precedes_output_and_any_simulator_phase(
    runner: tuple[Path, dict[str, str]],
) -> None:
    root, env = runner
    env["FAKE_PREFLIGHT_EXIT"] = "23"
    result = subprocess.run(
        ["bash", str(root / "scripts/run_experiment.sh"), "outputs/run", "recipe.json"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 23, result.stderr
    calls = [json.loads(line) for line in Path(env["FAKE_CALLS"]).read_text().splitlines()]
    assert calls == [
        ["-m", "rc_car.cli", "validate-config", "recipe.json"],
        ["-m", "rc_car.experiments.online_check", "--config", "recipe.json"],
    ]
    assert not (root / "outputs/run").exists()


def test_standalone_success_preflights_before_training(
    runner: tuple[Path, dict[str, str]],
) -> None:
    root, env = runner
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/run_experiment.sh"),
            "outputs/run",
            "recipe.json",
            "--externally-supervised",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["FAKE_CALLS"]).read_text().splitlines()]
    assert calls[1] == ["-m", "rc_car.experiments.online_check", "--config", "recipe.json"]
    phases = [call for call in calls if call[0] == "scripts/training_experiment.py"]
    assert [call[call.index("--mode") + 1] for call in phases] == ["train", "start", "end"]
    assert calls.index(phases[0]) > 1
