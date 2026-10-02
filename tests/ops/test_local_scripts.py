"""Local preparation guards: fixture coverage and source identity cannot be faked."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from rc_car.ops.bundles import create_bundle
from rc_car.workflow_config import load_workflow

ROOT = Path(__file__).resolve().parents[2]


def script(name: str) -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "scripts" / name))


def test_prepare_plan_is_nonmutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = script("prepare_session.py")
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "REPO_ROOT", tmp_path)
    destination = tmp_path / "outputs/new"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(destination), "--plan"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["cloud_actions"] == []
    assert not destination.exists()


def test_prepare_rejects_broad_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main = script("prepare_session.py")["main"]
    monkeypatch.setitem(main.__globals__, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path), "--plan"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2


def test_prepare_validates_unselected_yaml_profiles_before_commands_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow_payload: dict
) -> None:
    assets = tmp_path / "generated_assets"
    assets.mkdir()
    (assets / "fixture.json").write_text('{"kind":"procedural-smoke-test-only"}')
    workflow_payload["profiles"]["multiagent-baseline"]["ppo"]["batch_size"] = "invalid"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(workflow_payload))
    main = script("prepare_session.py")["main"]
    monkeypatch.setitem(main.__globals__, "REPO_ROOT", tmp_path)
    destination = tmp_path / "outputs/new"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(destination)])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Invalid YAML must fail before running commands")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        main()
    assert not destination.exists()


def test_fixture_manifest_must_cover_exact_fixture_tree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assets = root / "generated_assets"
    assets.mkdir(parents=True)
    (assets / "fixture.json").write_text(json.dumps({"kind": "procedural-smoke-test-only"}))
    (assets / "track.usd").write_text("synthetic fixture")
    verify = script("bundle.py")["verify_fixtures"]
    names = ["generated_assets/fixture.json", "generated_assets/track.usd"]
    paths = create_bundle(root, names, tmp_path / "full")
    verify(root, paths.manifest)
    incomplete = create_bundle(root, names[:1], tmp_path / "incomplete")
    with pytest.raises(ValueError, match="exactly"):
        verify(root, incomplete.manifest)
    (assets / "extra.usd").write_text("not captured")
    with pytest.raises(ValueError, match="exactly"):
        verify(root, paths.manifest)


def test_result_selector_never_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    (root / "result.json").write_text("{}")
    (root / "external").symlink_to(tmp_path, target_is_directory=True)
    (root / "alias.json").symlink_to(root / "result.json")
    select = script("bundle.py")["regular_files"]
    names, links = select(root, skip_links=True)
    assert names == ["result.json"]
    assert links == ["alias.json", "external"]


def test_result_selector_does_not_silently_drop_unreadable_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = script("bundle.py")

    def failed_walk(_root: Path, *, followlinks: bool, onerror: Any) -> list[Any]:
        assert followlinks is False
        onerror(PermissionError("injected directory scan failure"))
        return []

    monkeypatch.setattr(module["os"], "walk", failed_walk)
    with pytest.raises(PermissionError, match="scan failure"):
        module["regular_files"](tmp_path, skip_links=True)


@pytest.mark.parametrize("mutation", [None, "during-checks", "during-packing"])
@pytest.mark.parametrize("tracking_mode,timeout_seconds", [("online", 2400), ("offline", 137)])
def test_preparation_binds_checks_and_bundle_to_same_source(
    mutation: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_payload: dict,
    tracking_mode: str,
    timeout_seconds: int,
) -> None:
    root = tmp_path / "repo"
    assets = root / "generated_assets"
    assets.mkdir(parents=True)
    (assets / "fixture.json").write_text(json.dumps({"kind": "procedural-smoke-test-only"}))
    (assets / "track.usd").write_text("fixture")
    source = root / "source.py"
    source.write_text("original")
    workflow_payload["tracking"]["mode"] = tracking_mode
    workflow_payload["job"]["timeout_seconds"] = timeout_seconds
    (root / "config.yaml").write_text(yaml.safe_dump(workflow_payload))
    destination = root / "outputs/prepared"
    main = script("prepare_session.py")["main"]
    monkeypatch.setitem(main.__globals__, "REPO_ROOT", root)
    workflow = load_workflow(root / "config.yaml")
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(destination)])

    def local_command(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "scripts/check.py" in command:
            if mutation == "during-checks":
                source.write_text("changed while checking")
        else:
            assert "scripts/capture_provenance.py" in command
            target = Path(command[command.index("--output") + 1])
            target.write_text(
                json.dumps(
                    {
                        "files_sha256": {
                            "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
                        },
                        "branch": "test",
                        "git_commit": "test",
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", local_command)
    real_bundle = main.__globals__["create_bundle"]

    def bundle(root: Path, names: list[str], output: Path) -> Any:
        if mutation == "during-packing" and output.name == "source":
            source.write_text("changed after source capture")
        return real_bundle(root, names, output)

    monkeypatch.setitem(main.__globals__, "create_bundle", bundle)
    if mutation is None:
        assert main() == 0
        session = json.loads((destination / "session.json").read_text())
        assert session["local_checks_passed"] is True
        assert session["tracking_mode"] == workflow.tracking.mode
        assert session["workflow"] == {
            "config": "config.yaml",
            "operation": workflow.operation,
            "selected_profile": workflow.profile,
            "profiles": sorted(workflow.profiles),
        }
        assert "recipes" not in session
    else:
        with pytest.raises(SystemExit) as caught:
            main()
        assert caught.value.code == 2
        assert not (destination / "session.json").exists()
