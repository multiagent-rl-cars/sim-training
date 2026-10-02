"""Exercise the no-argument workflow with fake subprocesses/network adapters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from rc_car import workflow
from rc_car.experiments import online_check, upload
from rc_car.ops import evidence
from rc_car.workflow_config import WorkflowConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def payload(workflow_payload: dict) -> dict:
    workflow_payload["job"]["output"] = "outputs/workflow/test-run"
    return workflow_payload


@pytest.fixture(autouse=True)
def no_external_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", MagicMock(side_effect=AssertionError("Unexpected process"))
    )
    monkeypatch.setattr(
        online_check, "check_online", MagicMock(side_effect=AssertionError("Unexpected auth"))
    )
    monkeypatch.setattr(
        upload,
        "upload_completed_experiment",
        MagicMock(side_effect=AssertionError("Unexpected upload")),
    )


def test_check_runs_cpu_gate_without_tracking_or_gpu(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", process)
    config = WorkflowConfig.model_validate(payload)
    assert workflow.execute(config, tmp_path) == 0
    process.assert_called_once_with([sys.executable, "scripts/check.py"], cwd=tmp_path, check=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_plan_is_read_only_and_environment_cannot_override_yaml(
    tmp_path: Path,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    payload["operation"] = "plan"
    payload["tracking"]["mode"] = mode
    environment_mode = "offline" if mode == "online" else "online"
    monkeypatch.setenv("WANDB_MODE", environment_mode)
    assert workflow.execute(WorkflowConfig.model_validate(payload), tmp_path) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["effective_recipe"]["wandb"]["mode"] == mode
    assert result["cloud_allocation"] is False
    assert os.environ["WANDB_MODE"] == environment_mode
    assert list(tmp_path.iterdir()) == []


def test_automatic_output_plan_does_not_allocate_an_id_or_directory(
    tmp_path: Path,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload["operation"] = "plan"
    payload["job"].pop("output")
    monkeypatch.setattr(
        workflow.uuid, "uuid4", MagicMock(side_effect=AssertionError("No allocation in plan"))
    )
    assert workflow.execute(WorkflowConfig.model_validate(payload), tmp_path) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["output"] is None
    assert "outputs/runs/<uuid>" in result["output_policy"]
    assert list(tmp_path.iterdir()) == []


def test_automatic_training_outputs_are_fresh_and_recorded(
    tmp_path: Path,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload["operation"] = "train"
    payload["job"].pop("output")
    monkeypatch.setattr(sys, "platform", "linux")
    process = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", process)
    config = WorkflowConfig.model_validate(payload)
    assert workflow.execute(config, tmp_path) == workflow.execute(config, tmp_path) == 0
    outputs = sorted((tmp_path / "outputs/runs").iterdir())
    assert len(outputs) == 2 and outputs[0] != outputs[1]
    for output in outputs:
        assert uuid.UUID(output.name).hex == output.name
        record = json.loads((output / "workflow-config.json").read_text())
        assert record["requested_config"]["job"]["output"] is None
        assert record["resolved_output"] == str(output)
        assert record["output_kind"] == "automatic"
        assert (output / "requested-recipe.json").is_file()
    assert capsys.readouterr().out.count("Local output:") == 2
    assert process.call_count == 2


def test_automatic_output_collision_retries_without_overwriting(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload["job"].pop("output")
    old_id, new_id = uuid.UUID(int=1), uuid.UUID(int=2)
    existing = tmp_path / "outputs/runs" / old_id.hex
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("preserve")
    monkeypatch.setattr(workflow.uuid, "uuid4", MagicMock(side_effect=[old_id, new_id]))
    output = workflow._create_output(WorkflowConfig.model_validate(payload), tmp_path)
    assert output.name == new_id.hex
    assert (existing / "keep.txt").read_text() == "preserve"
    assert not (existing / "workflow-config.json").exists()


def test_automatic_output_rejects_symlink_parent_without_writing_outside(
    tmp_path: Path, payload: dict
) -> None:
    payload["job"].pop("output")
    (tmp_path / "outputs").mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "outputs/runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        workflow._create_output(WorkflowConfig.model_validate(payload), tmp_path)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_tracking_check_uses_yaml_mode_and_restores_environment(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch, failed: bool, mode: str
) -> None:
    payload["operation"] = "tracking-check"
    payload["tracking"]["mode"] = mode
    environment_mode = "offline" if mode == "online" else "online"
    monkeypatch.setenv("WANDB_MODE", environment_mode)

    def check(config):
        assert config.wandb.mode == mode
        assert os.environ["WANDB_MODE"] == mode
        if failed:
            raise RuntimeError("synthetic failure")
        return {"status": "synthetic-check-only"}

    monkeypatch.setattr(online_check, "check_online", check)
    config = WorkflowConfig.model_validate(payload)
    if failed:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            workflow.execute(config, tmp_path)
    else:
        assert workflow.execute(config, tmp_path) == 0
    assert os.environ["WANDB_MODE"] == environment_mode
    assert list(tmp_path.iterdir()) == []


def test_training_fails_on_mac_without_creating_output(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload["operation"] = "train"
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux RTX"):
        workflow.execute(WorkflowConfig.model_validate(payload), tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["online", "offline"])
@pytest.mark.parametrize("timeout_seconds", [2400, 137])
def test_training_materializes_exact_requested_recipe_and_uses_existing_supervisor(
    tmp_path: Path,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    timeout_seconds: int,
) -> None:
    payload["operation"] = "train"
    payload["tracking"]["mode"] = mode
    payload["job"]["timeout_seconds"] = timeout_seconds
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret-not-copied")
    process = MagicMock(return_value=subprocess.CompletedProcess([], 7))
    monkeypatch.setattr(subprocess, "run", process)
    config = WorkflowConfig.model_validate(payload)
    assert workflow.execute(config, tmp_path) == 7
    output = tmp_path / payload["job"]["output"]
    recipe = output / "requested-recipe.json"
    assert json.loads(recipe.read_text()) == config.requested_recipe()
    assert "synthetic-secret-not-copied" not in recipe.read_text()
    assert "synthetic-secret-not-copied" not in (output / "workflow-config.json").read_text()
    command = process.call_args.args[0]
    assert command[:4] == ["bash", "scripts/run_gpu_checks.sh", "--mode", "experiment"]
    assert command[command.index("--config") + 1] == str(recipe)
    assert command[command.index("--output") + 1] == str(output / "gpu")
    assert command[command.index("--tracking-mode") + 1] == mode
    assert command[command.index("--timeout-seconds") + 1] == str(timeout_seconds)
    assert process.call_args.kwargs == {"cwd": tmp_path, "check": False}
    with pytest.raises(ValueError, match="fresh job.output"):
        workflow.execute(config, tmp_path)
    process.assert_called_once()


def test_inspect_preserves_input_and_writes_fresh_evidence(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload["operation"] = "inspect"
    payload["job"]["input_run"] = "outputs/prior/experiment"
    inspect = MagicMock(return_value={"integration_passed": False, "driving_proven": False})
    monkeypatch.setattr(evidence, "inspect_run", inspect)
    config = WorkflowConfig.model_validate(payload)
    assert workflow.execute(config, tmp_path) == 1
    inspect.assert_called_once_with(tmp_path / "outputs/prior/experiment")
    assert (
        json.loads((tmp_path / payload["job"]["output"] / "evidence.json").read_text())
        == inspect.return_value
    )


@pytest.mark.parametrize("different_destination", [False, True])
def test_upload_uses_saved_run_and_cannot_relabel_destination(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch, different_destination: bool
) -> None:
    payload["operation"] = "upload"
    payload["job"]["input_run"] = "outputs/prior/experiment"
    config = WorkflowConfig.model_validate(payload)
    monkeypatch.setattr(
        upload,
        "validate_upload",
        MagicMock(
            return_value={
                "identity": {
                    "entity": "other" if different_destination else config.tracking.entity,
                    "project": config.tracking.project,
                }
            }
        ),
    )
    send = MagicMock(return_value={"verified": True})
    monkeypatch.setattr(upload, "upload_completed_experiment", send)
    if different_destination:
        with pytest.raises(ValueError, match="refusing to relabel"):
            workflow.execute(config, tmp_path)
        send.assert_not_called()
    else:
        assert workflow.execute(config, tmp_path) == 0
        send.assert_called_once_with(
            tmp_path / "outputs/prior/experiment",
            work_dir=tmp_path / payload["job"]["output"] / "upload",
            receipt_path=tmp_path / payload["job"]["output"] / "upload-receipt.json",
        )


def test_cannot_write_into_saved_run(tmp_path: Path, payload: dict) -> None:
    payload["operation"] = "inspect"
    payload["job"]["input_run"] = "outputs/prior"
    payload["job"]["output"] = "outputs/prior/evidence"
    with pytest.raises(ValueError, match="outside the saved input"):
        workflow.execute(WorkflowConfig.model_validate(payload), tmp_path)


def test_front_door_rejects_all_arguments_without_loading_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["workflow", "--anything"])
    monkeypatch.setattr(workflow, "load_workflow", MagicMock(side_effect=AssertionError))
    assert workflow.main() == 2
    assert "uv run rc-car" in capsys.readouterr().err


def test_installed_entrypoint_selects_yaml_workflow() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["scripts"]["rc-car"] == "rc_car.workflow:main"


def test_plain_uv_run_includes_required_check_and_tracking_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = project["dependency-groups"]
    assert set(project["tool"]["uv"]["default-groups"]) == {"dev", "cpu", "tracking"}
    assert {"torch==2.7.0", "stable-baselines3==2.7.0"} <= set(groups["cpu"])
    assert "wandb==0.25.1" in groups["tracking"]
    # Installing the portable package into Isaac must not pull CPU dev groups.
    assert "torch==2.7.0" not in project["project"]["dependencies"]


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_front_door_uses_repository_config_not_cwd(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    payload["tracking"]["mode"] = mode
    payload["job"]["timeout_seconds"] = 137
    (root / "config.yaml").write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(workflow, "REPO_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["workflow"])
    monkeypatch.chdir(tmp_path)
    execute = MagicMock(return_value=0)
    monkeypatch.setattr(workflow, "execute", execute)
    assert workflow.main() == 0
    assert execute.call_args.args[0] == WorkflowConfig.model_validate(payload)
    assert execute.call_args.args[1] == root
