"""One editable recipe source; historical JSON is test data, never runtime input."""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_HASHES = {
    "basic_smoke.json": "76a347eba935bd366c24ec1cbd9c178aad62ce464098634618f0a98a633295b1",
    "line_following_15min.json": "dc2c6c56cb3e289f1dffff792df9fb1b16779aab0079f83e74da888619dae754",
    "multiagent_baseline.json": "1d0a415939f70b2f9ba4519561c04311a9f9ea165bb9c3a414f03c6ed70de57e",
    "multiagent_smoke.json": "9dc93cbfc211d7ace9ba2c208841259c194dd8c7b78b5a63623b813b72276a3e",
}


def test_legacy_recipes_are_byte_preserved_test_fixtures_only() -> None:
    fixture_root = ROOT / "tests/fixtures/legacy_recipes"
    assert {path.name for path in fixture_root.glob("*.json")} == set(LEGACY_HASHES)
    for name, expected in LEGACY_HASHES.items():
        assert hashlib.sha256((fixture_root / name).read_bytes()).hexdigest() == expected
    assert not (ROOT / "configs").exists()


def test_runtime_code_does_not_load_historical_recipe_locations() -> None:
    for directory in ("src/rc_car", "scripts", "ops"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text()
            assert "configs/experiments" not in text, path
            assert "tests/fixtures/legacy_recipes" not in text, path


def test_training_worker_requires_explicit_generated_recipe() -> None:
    tree = ast.parse((ROOT / "scripts/training_experiment.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--config"
    ]
    assert len(calls) == 1
    keywords = {entry.arg: entry.value for entry in calls[0].keywords}
    assert "default" not in keywords
    required = keywords["required"]
    assert isinstance(required, ast.Constant) and required.value is True


def test_shell_worker_has_no_implicit_configuration(tmp_path: Path) -> None:
    output = tmp_path / "must-not-be-created"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_experiment.sh"), str(output)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "explicit generated recipe" in result.stderr
    assert "uv run rc-car" in result.stderr
    assert not output.exists()
