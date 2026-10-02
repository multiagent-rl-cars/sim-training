"""CPU-only saved evidence fixtures; no simulator, SDK, or cloud access."""

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from rc_car.ops import evidence


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value))


def _checkpoint(path: Path, *, updated: bool = False, bn_only: bool = False) -> str:
    weights = {
        "actor.weight": torch.tensor([[1.0 + float(updated), 2.0]]),
        "actor.bias": torch.zeros(1),
        "bn.running_mean": torch.tensor([float(bn_only)]),
        "bn.running_var": torch.ones(1),
        "bn.num_batches_tracked": torch.tensor(int(bn_only)),
    }
    _save_checkpoint(path, weights)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_checkpoint(path: Path, weights: Any) -> None:
    stream = io.BytesIO()
    torch.save(weights, stream)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("policy.pth", stream.getvalue())
        # Deliberately not a valid SB3 metadata payload. The checker must not read it.
        archive.writestr("data", b"DO NOT UNPICKLE SB3 METADATA")


def _checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in report["checks"]}


@pytest.fixture
def smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _checkpoint(tmp_path / "initial.zip")
    _checkpoint(tmp_path / "final.zip", updated=True)
    _write_json(
        tmp_path / "result.json",
        {
            "num_envs": 2,
            "steps": 8,
            "ppo_updates": 1,
            "nonreset_path_length_m": [0.1, 0.2],
            "videos": [{"path": "smoke.mp4", "expected_frames": 8}],
        },
    )
    (tmp_path / "smoke.mp4").write_bytes(b"synthetic media probe fixture")
    monkeypatch.setattr(evidence, "_probe_video", lambda _path, frames: {"decoded_frames": frames})
    return tmp_path


@pytest.fixture
def rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    initial_hash = _checkpoint(tmp_path / "initial.zip")
    final_hash = _checkpoint(tmp_path / "final.zip", updated=True)
    _write_json(
        tmp_path / "training_summary.json",
        {
            "ppo_updates": 2,
            "initial_checkpoint_sha256": initial_hash,
            "final_checkpoint_sha256": final_hash,
        },
    )
    _write_json(
        tmp_path / "config.json",
        {
            "num_envs": 2,
            "eval_steps": 5,
            "video_stride": 2,
            "eval_seed": 4,
        },
    )
    comparison = {"matched_initial_states": True, "tolerance": 1e-6}
    for phase, digest in (("start", initial_hash), ("end", final_hash)):
        evaluation = {
            "phase": phase,
            "seed": 4,
            "deterministic": True,
            "control_steps": 5,
            "nonreset_distance_per_car_m": [0.1, 0.2],
            "checkpoint_sha256": digest,
        }
        comparison[phase] = evaluation
        _write_json(tmp_path / f"{phase}_evaluation.json", evaluation)
        _write_json(
            tmp_path / f"{phase}_initial_state.json",
            {
                "root_state": [[0.0] * 13] * 2,
                "joint_pos": [[0.0, 1.0]] * 2,
                "joint_vel": [[0.0, 0.0]] * 2,
            },
        )
        (tmp_path / f"{phase}.mp4").write_bytes(b"media fixture")
    _write_json(tmp_path / "comparison.json", comparison)
    monkeypatch.setattr(evidence, "_probe_video", lambda _path, frames: {"decoded_frames": frames})
    return tmp_path


def test_smoke_compares_tensors_not_zip_bytes(smoke: Path) -> None:
    report = evidence.inspect_run(smoke)
    assert report["integration_passed"]
    assert not report["driving_proven"]
    assert report["artifact_preservation_recommended"]
    assert _checks(report)["policy_tensor_update"]["details"]["changed_tensor_names"] == [
        "actor.weight"
    ]
    json.dumps(report, allow_nan=False)


def test_legacy_checkpoint_filename_supported(smoke: Path) -> None:
    (smoke / "final.zip").rename(smoke / "fixture_ppo.zip")
    assert evidence.inspect_run(smoke)["integration_passed"]


@pytest.mark.parametrize("bn_only", [False, True])
def test_unchanged_parameters_and_batchnorm_only_fail(smoke: Path, bn_only: bool) -> None:
    _checkpoint(smoke / "final.zip", bn_only=bn_only)
    report = evidence.inspect_run(smoke)
    assert not report["integration_passed"]
    assert "No policy tensor changed" in _checks(report)["policy_tensor_update"]["reason"]


@pytest.mark.parametrize("updates", [0, -1, True, 1.2, None])
def test_update_count_cannot_be_inferred_from_weights(smoke: Path, updates: Any) -> None:
    result = json.loads((smoke / "result.json").read_text())
    result["ppo_updates"] = updates
    _write_json(smoke / "result.json", result)
    report = evidence.inspect_run(smoke)
    assert _checks(report)["ppo_update_count"]["status"] == "fail"
    assert _checks(report)["policy_tensor_update"]["status"] == "pass"


@pytest.mark.parametrize("distances", [[0.0, 1.0], [-1.0, 1.0], [1.0], None, [True, 1]])
def test_each_car_must_move_without_resets(smoke: Path, distances: Any) -> None:
    result = json.loads((smoke / "result.json").read_text())
    result.update(nonreset_path_length_m=distances, max_xy_displacement_m=100)
    _write_json(smoke / "result.json", result)
    assert _checks(evidence.inspect_run(smoke))["per_car_motion"]["status"] == "fail"


@pytest.mark.parametrize(
    "bad_weights",
    [
        {"actor.weight": torch.tensor([[float("nan"), 1.0]])},
        {"actor.weight": torch.ones(3)},
        {"unexpected": torch.ones(1)},
        {"not_tensor": "invalid"},
        {},
    ],
)
def test_policy_schema_and_finiteness_fail(smoke: Path, bad_weights: Any) -> None:
    _save_checkpoint(smoke / "final.zip", bad_weights)
    assert _checks(evidence.inspect_run(smoke))["policy_tensor_update"]["status"] == "fail"


def test_weights_only_and_cpu_loading_are_explicit(
    smoke: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_load = torch.load
    calls = []

    def load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    assert evidence.inspect_run(smoke)["integration_passed"]
    assert calls == [{"map_location": "cpu", "weights_only": True}] * 2


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite_buffer"])
def test_tensor_contract_checked_even_for_excluded_buffers(smoke: Path, change: str) -> None:
    weights = evidence._policy(smoke / "final.zip")
    if change == "shape":
        weights["actor.weight"] = torch.ones(3)
    elif change == "dtype":
        weights["actor.weight"] = weights["actor.weight"].double()
    else:
        weights["bn.running_mean"] = torch.tensor([float("nan")])
    _save_checkpoint(smoke / "final.zip", weights)
    assert _checks(evidence.inspect_run(smoke))["policy_tensor_update"]["status"] == "fail"


def test_duplicate_policy_member_rejected(smoke: Path) -> None:
    with zipfile.ZipFile(smoke / "final.zip", "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("policy.pth", b"duplicate")
    assert "one bounded" in _checks(evidence.inspect_run(smoke))["policy_tensor_update"]["reason"]


def test_partial_failed_run_still_reports_preserve_artifacts(tmp_path: Path) -> None:
    (tmp_path / "interrupted.zip").write_bytes(b"incomplete")
    report = evidence.inspect_run(tmp_path)
    assert not report["integration_passed"]
    assert report["artifact_preservation_recommended"]
    assert _checks(report)["run_summary"]["status"] == "fail"
    assert (tmp_path / "interrupted.zip").read_bytes() == b"incomplete"


def test_rehearsal_requires_actual_matching_states(rehearsal: Path) -> None:
    report = evidence.inspect_run(rehearsal)
    assert report["integration_passed"]
    assert _checks(report)["video:start.mp4"]["details"]["decoded_frames"] == 3
    states = json.loads((rehearsal / "end_initial_state.json").read_text())
    states["root_state"][0][0] = 0.25
    _write_json(rehearsal / "end_initial_state.json", states)
    assert (
        "Actual initial root_state values differ"
        in _checks(evidence.inspect_run(rehearsal))["matched_evaluation"]["reason"]
    )


@pytest.mark.parametrize("field", ["eval_steps", "eval_seed"])
def test_evaluation_must_match_recorded_recipe(rehearsal: Path, field: str) -> None:
    config = json.loads((rehearsal / "config.json").read_text())
    config[field] += 1
    _write_json(rehearsal / "config.json", config)
    assert _checks(evidence.inspect_run(rehearsal))["matched_evaluation"]["status"] == "fail"


@pytest.mark.parametrize("location", ["end_seed", "end_steps", "config_seed", "config_steps"])
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_equal_but_wrong_type_evaluation_integers_fail(
    rehearsal: Path, location: str, invalid: Any
) -> None:
    config = json.loads((rehearsal / "config.json").read_text())
    config.update(eval_steps=1, eval_seed=1)
    if location.startswith("config"):
        config["eval_seed" if location == "config_seed" else "eval_steps"] = invalid
    _write_json(rehearsal / "config.json", config)
    comparison = json.loads((rehearsal / "comparison.json").read_text())
    for phase in ("start", "end"):
        evaluation = json.loads((rehearsal / f"{phase}_evaluation.json").read_text())
        evaluation.update(seed=1, control_steps=1)
        if phase == "end" and location.startswith("end"):
            evaluation["seed" if location == "end_seed" else "control_steps"] = invalid
        comparison[phase] = evaluation
        _write_json(rehearsal / f"{phase}_evaluation.json", evaluation)
    _write_json(rehearsal / "comparison.json", comparison)
    assert _checks(evidence.inspect_run(rehearsal))["matched_evaluation"]["status"] == "fail"


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_range_is_checked(rehearsal: Path, seed: int) -> None:
    config = json.loads((rehearsal / "config.json").read_text())
    config["eval_seed"] = seed
    _write_json(rehearsal / "config.json", config)
    assert _checks(evidence.inspect_run(rehearsal))["matched_evaluation"]["status"] == "fail"


@pytest.mark.parametrize("frames", [1, 7, 9])
def test_truncated_smoke_cannot_lower_expected_frame_declaration(smoke: Path, frames: int) -> None:
    result = json.loads((smoke / "result.json").read_text())
    result["videos"][0]["expected_frames"] = frames
    _write_json(smoke / "result.json", result)
    assert _checks(evidence.inspect_run(smoke))["video_expectations"]["status"] == "fail"


@pytest.mark.parametrize(
    "payload", ['{"ppo_updates":0,"ppo_updates":1}', '{"nested":{"x":0,"x":1}}']
)
def test_duplicate_evidence_json_keys_fail(smoke: Path, payload: str) -> None:
    (smoke / "result.json").write_text(payload)
    assert "duplicate" in _checks(evidence.inspect_run(smoke))["run_summary"]["reason"]


@pytest.mark.parametrize(
    "change", ["flag", "seed", "empty_state", "checkpoint", "stale_comparison"]
)
def test_rehearsal_rejects_unmatched_or_stale_evidence(rehearsal: Path, change: str) -> None:
    comparison = json.loads((rehearsal / "comparison.json").read_text())
    if change == "flag":
        comparison["matched_initial_states"] = False
    elif change == "empty_state":
        _write_json(rehearsal / "end_initial_state.json", {})
    else:
        evaluation = json.loads((rehearsal / "end_evaluation.json").read_text())
        evaluation["seed" if change == "seed" else "checkpoint_sha256"] = "bad"
        _write_json(rehearsal / "end_evaluation.json", evaluation)
        if change != "stale_comparison":
            comparison["end"] = evaluation
    _write_json(rehearsal / "comparison.json", comparison)
    assert _checks(evidence.inspect_run(rehearsal))["matched_evaluation"]["status"] == "fail"


def test_nonfinite_json_does_not_serialize_as_success(smoke: Path) -> None:
    result = json.loads((smoke / "result.json").read_text())
    result["unrelated_metric"] = float("inf")
    _write_json(smoke / "result.json", result)
    report = evidence.inspect_run(smoke)
    assert not report["integration_passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "path", ["../secret.mp4", "/secret.mp4", "sub/../smoke.mp4", "./smoke.mp4"]
)
def test_media_paths_cannot_escape_or_alias(smoke: Path, path: str) -> None:
    result = json.loads((smoke / "result.json").read_text())
    result["videos"][0]["path"] = path
    _write_json(smoke / "result.json", result)
    assert not evidence.inspect_run(smoke)["integration_passed"]


def test_symlink_checkpoint_is_rejected(smoke: Path) -> None:
    (smoke / "final.zip").unlink()
    (smoke / "final.zip").symlink_to(smoke / "initial.zip")
    assert "symlink" in _checks(evidence.inspect_run(smoke))["policy_tensor_update"]["reason"]


@pytest.mark.parametrize("failure", ["wrong_frames", "blank", "timeout", "missing_tools"])
def test_media_probe_rejects_incomplete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(
        evidence.shutil, "which", lambda name: None if failure == "missing_tools" else name
    )
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["timeout"] == evidence.MEDIA_TIMEOUT_SECONDS
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert command[command.index("-f") + 1] == "mov"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {"streams": [{"nb_read_frames": "1" if failure == "wrong_frames" else "8"}]}
                ).encode(),
            )
        return subprocess.CompletedProcess(command, 0, bytes(4096))

    monkeypatch.setattr(evidence.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        evidence._probe_video(tmp_path / "video.mp4", 8)
    if failure == "missing_tools":
        assert calls == []


def test_media_probe_reports_decoded_nonblank_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence.shutil, "which", lambda name: name)

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert command[command.index("-f") + 1] == "mov"
        value = (
            b'{"streams":[{"nb_read_frames":"8"}]}'
            if command[0] == "ffprobe"
            else bytes(range(256)) * 16
        )
        return subprocess.CompletedProcess(command, 0, value)

    monkeypatch.setattr(evidence.subprocess, "run", run)
    result = evidence._probe_video(tmp_path / "video.mp4", 8)
    assert result["decoded_frames"] == 8
    assert result["nonblank_samples"] == 1
