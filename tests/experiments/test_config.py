"""Portable recipe contracts; these checks never construct a simulator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rc_car.experiments import load_config, resolve_config, validate_config

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "tests/fixtures/legacy_recipes"


@pytest.fixture
def recipe() -> dict[str, Any]:
    return json.loads((RECIPES / "line_following_15min.json").read_text())


@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.json")), ids=lambda p: p.stem)
def test_legacy_fixture_recipes_round_trip_without_changing_values(path: Path) -> None:
    original = json.loads(path.read_text())
    resolved = resolve_config(load_config(path))
    for key, value in original.items():
        if key in ("wandb", "ppo"):
            for nested_key, nested_value in value.items():
                assert resolved[key][nested_key] == nested_value
        else:
            assert resolved[key] == value
    assert resolve_config(validate_config(resolved)) == resolved


def test_optional_defaults_preserve_runner_semantics() -> None:
    resolved = resolve_config(load_config(RECIPES / "multiagent_baseline.json"))
    assert "max_training_seconds" not in resolved
    assert "training_video_interval_s" not in resolved
    assert "wandb" not in resolved
    assert resolved["training_video_frames"] == 90
    assert resolved["enable_obstacles"] is True


def test_resolution_is_independent_of_input_mutation(recipe: dict[str, Any]) -> None:
    before = copy.deepcopy(recipe)
    parsed = validate_config(recipe)
    resolved = resolve_config(parsed)
    resolved["ppo"]["gamma"] = 0.5
    recipe["known_limitations"].append("new limitation")
    assert parsed.ppo.gamma == before["ppo"]["gamma"]
    assert parsed.known_limitations is not None
    assert list(parsed.known_limitations) == before["known_limitations"]
    with pytest.raises(ValidationError, match="frozen"):
        # Intentionally violate the static contract to verify runtime immutability.
        parsed.num_envs = 1  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("section", [None, "ppo", "wandb"])
def test_unknown_fields_rejected_without_echoing_values(
    recipe: dict[str, Any], section: str | None
) -> None:
    target = recipe if section is None else recipe[section]
    target["unexpected_field"] = "private-test-value"
    with pytest.raises(ValidationError) as error:
        validate_config(recipe)
    assert "private-test-value" not in str(error.value)


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "num_envs",
        "train_seed",
        "eval_seed",
        "total_timesteps",
        "video_stride",
        "eval_steps",
        "checkpoint_interval",
        "torch_threads",
        "training_video_frames",
        "max_training_seconds",
        "training_video_interval_s",
    ],
)
def test_booleans_are_not_numbers(recipe: dict[str, Any], key: str) -> None:
    recipe[key] = True
    with pytest.raises(ValidationError):
        validate_config(recipe)


@pytest.mark.parametrize("key", ["max_training_seconds", "training_video_interval_s"])
@pytest.mark.parametrize("value", [0, -1, None, float("nan"), float("inf"), "900"])
def test_invalid_budgets(recipe: dict[str, Any], key: str, value: object) -> None:
    recipe[key] = value
    with pytest.raises(ValidationError):
        validate_config(recipe)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_steps", True),
        ("n_epochs", 0),
        ("batch_size", 1),
        ("batch_size", 10**9),
        ("gamma", 1.1),
        ("gae_lambda", -0.1),
        ("clip_range", 0),
        ("learning_rate", float("nan")),
        ("ent_coef", float("inf")),
        ("vf_coef", -1),
        ("max_grad_norm", 0),
        ("use_sde", 1),
        ("sde_sample_freq", True),
        ("sde_sample_freq", 0),
    ],
)
def test_invalid_ppo_parameters(recipe: dict[str, Any], key: str, value: object) -> None:
    recipe["ppo"][key] = value
    with pytest.raises(ValidationError):
        validate_config(recipe)


def test_tracking_test_metadata_is_explicitly_supported(recipe: dict[str, Any]) -> None:
    recipe.update(
        test_only=True, media_provenance="synthetic.mp4", note="test", synthetic_media=True
    )
    assert validate_config(recipe).test_only is True


@pytest.mark.parametrize("key", ["train_seed", "eval_seed"])
def test_seeds_match_numpy_bounds(recipe: dict[str, Any], key: str) -> None:
    recipe[key] = 2**32 - 1
    assert getattr(validate_config(recipe), key) == 2**32 - 1
    recipe[key] = 2**32
    with pytest.raises(ValidationError):
        validate_config(recipe)


def test_partial_ppo_minibatches_remain_supported(recipe: dict[str, Any]) -> None:
    # SB3 warns about a partial minibatch, but supports it without changing size.
    recipe["ppo"]["batch_size"] = 255
    assert validate_config(recipe).ppo.batch_size == 255


def test_nonobject_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(path)


@pytest.mark.parametrize(
    ("document", "key"),
    [
        ('{"task": "private-first-value", "task": "private-second-value"}', "task"),
        (
            '{"ppo": {"learning_rate": "private-first-value", '
            '"learning_rate": "private-second-value"}}',
            "learning_rate",
        ),
        (
            '{"nested": [{"label": "private-first-value", "label": "private-second-value"}]}',
            "label",
        ),
    ],
)
def test_duplicate_json_keys_rejected_recursively_without_values(
    tmp_path: Path, document: str, key: str
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(document)
    with pytest.raises(ValueError, match="Duplicate experiment configuration key") as error:
        load_config(path)
    assert key in str(error.value)
    assert "private-first-value" not in str(error.value)
    assert "private-second-value" not in str(error.value)


def test_recipe_import_needs_no_simulator_or_tracking_sdk() -> None:
    program = """
import importlib.abc
import socket
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'isaaclab', 'isaacsim', 'omni', 'torch', 'wandb'}:
            raise AssertionError('unexpected heavy import: ' + fullname)

def no_network(*args, **kwargs):
    raise AssertionError('unexpected network access')

sys.meta_path.insert(0, BlockHeavyImports())
socket.socket.connect = no_network
from rc_car.experiments import load_config, resolve_config
assert resolve_config(load_config(sys.argv[1]))['num_envs'] == 12
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(RECIPES / "line_following_15min.json")],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
