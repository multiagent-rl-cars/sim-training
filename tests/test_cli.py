import json
import subprocess
import sys
from pathlib import Path

from rc_car.cli import main

RECIPE = Path(__file__).resolve().parent / "fixtures/legacy_recipes/multiagent_baseline.json"


def test_show_config_exposes_defaults(capsys) -> None:
    assert main(["show-config", str(RECIPE)]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["enable_obstacles"] is True
    assert resolved["training_video_frames"] == 90
    assert "max_training_seconds" not in resolved


def test_cli_from_another_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rc_car.cli", "validate-config", str(RECIPE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid:" in result.stdout


def test_missing_file_is_actionable(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rc_car.cli", "validate-config", str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Invalid experiment configuration" in result.stderr
    assert "Traceback" not in result.stderr


def test_portable_imports_do_not_pull_simulator_or_learning_runtime() -> None:
    code = (
        "import sys; import rc_car; import rc_car.experiments.config; "
        "assert not any(name.split('.')[0] in "
        "{'isaaclab', 'isaacsim', 'omni', 'car_training', 'torch', 'wandb'} "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
