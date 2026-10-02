"""Workflow YAML is strict, self-contained, secret-free and preserves recipe values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rc_car.experiments.config import load_config, resolve_config
from rc_car.workflow_config import WorkflowConfig, load_workflow, resolve_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def payload(workflow_payload: dict) -> dict:
    return workflow_payload


def test_editable_configuration_is_valid() -> None:
    load_workflow(ROOT / "config.yaml")


@pytest.mark.parametrize(
    "profile", ["basic-smoke", "multiagent-smoke", "line-following-15min", "multiagent-baseline"]
)
def test_yaml_profile_preserves_every_numerical_recipe_field(
    tmp_path: Path,
    payload: dict,
    profile: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    canonical = load_workflow(path)
    actual = resolve_config(canonical.experiment(profile))
    legacy_path = ROOT / "tests/fixtures/legacy_recipes" / (profile.replace("-", "_") + ".json")
    expected = resolve_config(load_config(legacy_path))
    # Shared tracking replaces legacy identity; numerical values must survive
    # YAML loading unchanged. The editable user recipe may intentionally differ.
    expected.pop("wandb", None)
    expected.pop("name", None)
    expected.pop("known_limitations")
    actual.pop("known_limitations", None)
    assert actual.pop("wandb") == canonical.tracking.model_dump(mode="json")
    assert actual == expected


@pytest.mark.parametrize(
    "text",
    [
        "schema_version: 1\nschema_version: 1\n",
        "defaults:\n  train_seed: 42\n  train_seed: 43\n",
        "x: &value 3\ny: *value\n",
        "x: {<<: {a: 1}}\n",
        "x: !!python/object:example.Foo {}\n",
        "[not, a, mapping]\n",
        "1: no\n",
        "---\nx: 1\n---\ny: 2\n",
    ],
)
def test_rejects_ambiguous_or_nonplain_yaml(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_workflow(path)


@pytest.mark.parametrize("field", ["wandb_api_key", "access-token", "password", "credentials"])
def test_secret_fields_are_rejected_before_schema_errors(
    tmp_path: Path, payload: dict, field: str
) -> None:
    payload["profiles"]["basic-smoke"][field] = "synthetic-secret"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="Credentials must not appear") as error:
        load_workflow(path)
    assert "synthetic-secret" not in str(error.value)


def test_key_value_is_rejected_even_in_an_allowed_note(tmp_path: Path, payload: dict) -> None:
    payload["profiles"]["basic-smoke"]["note"] = "wandb_v1_synthetic-never-real"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="Credential-like"):
        load_workflow(path)


@pytest.mark.parametrize(
    "mutation", ["unknown", "bool-version", "unknown-profile", "unselected", "wandb"]
)
def test_all_profiles_and_unknown_fields_are_validated(payload: dict, mutation: str) -> None:
    if mutation == "unknown":
        payload["quietly_ignored"] = 1
    elif mutation == "bool-version":
        payload["schema_version"] = True
    elif mutation == "unknown-profile":
        payload["profile"] = "absent"
    elif mutation == "unselected":
        payload["profiles"]["multiagent-baseline"]["ppo"]["batch_size"] = "256"
    else:
        payload["profiles"]["basic-smoke"]["wandb"] = {"mode": "offline"}
    with pytest.raises(ValueError):
        WorkflowConfig.model_validate(payload)


def test_profiles_are_self_contained_and_return_independent_recipe_copies(payload: dict) -> None:
    payload["profiles"]["basic-smoke"]["ppo"]["gamma"] = 0.98
    config = WorkflowConfig.model_validate(payload)
    actual = config.experiment("basic-smoke")
    assert actual.ppo.gamma == 0.98
    assert actual.ppo.learning_rate == 0.0003
    assert config.experiment("multiagent-smoke").ppo.gamma == 0.99
    first = config.requested_recipe()
    first["ppo"]["gamma"] = 0.1
    assert config.requested_recipe()["ppo"]["gamma"] == 0.98


def test_output_is_optional_and_profile_names_are_not_recipe_names(payload: dict) -> None:
    payload["job"].pop("output", None)
    config = WorkflowConfig.model_validate(payload)
    assert config.job.output is None
    recipe = config.requested_recipe()
    assert recipe["schema_version"] == config.schema_version
    assert "name" not in recipe and "known_limitations" not in recipe
    assert "defaults" not in payload


@pytest.mark.parametrize("field", ["name", "known_limitations", "schema_version"])
def test_profiles_reject_descriptive_or_duplicate_global_metadata(
    payload: dict, field: str
) -> None:
    payload["profiles"]["basic-smoke"][field] = "not-an-execution-setting"
    with pytest.raises(ValueError, match="execution settings only"):
        WorkflowConfig.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/run",
        "../run",
        "outputs/../run",
        "outputs",
        "outputs//run",
        "outputs/./run",
        "src/run",
        "outputs\\run",
    ],
)
def test_output_paths_are_canonical_and_confined(payload: dict, path: str) -> None:
    payload["job"]["output"] = path
    with pytest.raises(ValueError, match="canonical paths below outputs"):
        WorkflowConfig.model_validate(payload)


def test_rejects_symlink_output_parent(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "outputs/alias").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        resolve_path(tmp_path, "outputs/alias/run")


@pytest.mark.parametrize("kind", ["symlink", "oversize", "directory"])
def test_configuration_is_a_bounded_regular_file(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "config.yaml"
    if kind == "symlink":
        target = tmp_path / "target.yaml"
        target.write_text("schema_version: 1\n")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_text("x" * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded regular"):
        load_workflow(path)


@pytest.mark.parametrize("operation", ["inspect", "upload"])
def test_existing_run_operations_require_explicit_input(payload: dict, operation: str) -> None:
    payload["operation"] = operation
    with pytest.raises(ValueError, match="require job.input_run"):
        WorkflowConfig.model_validate(payload)


def test_upload_requires_explicit_online_tracking(payload: dict) -> None:
    payload["operation"] = "upload"
    payload["job"]["input_run"] = "outputs/prior/experiment"
    payload["tracking"]["mode"] = "offline"
    with pytest.raises(ValueError, match="tracking.mode: online"):
        WorkflowConfig.model_validate(payload)
