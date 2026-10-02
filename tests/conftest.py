"""Deterministic workflow inputs, independent of the editable user configuration."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def workflow_payload() -> dict:
    recipes = Path(__file__).parent / "fixtures/legacy_recipes"
    profiles = {}
    for path in sorted(recipes.glob("*.json")):
        recipe = json.loads(path.read_text())
        profiles[path.stem.replace("_", "-")] = {
            key: value
            for key, value in recipe.items()
            if key not in {"schema_version", "name", "known_limitations", "wandb"}
        }
    return {
        "schema_version": 1,
        "operation": "check",
        "profile": "basic-smoke",
        "job": {
            "fixture_manifest": "outputs/fixture-bundle-manifest.json",
            "timeout_seconds": 2400,
        },
        "tracking": {"entity": "test-entity", "project": "test-project", "mode": "online"},
        "profiles": profiles,
    }
