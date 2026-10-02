"""Inspect saved local integration evidence without importing Isaac or unpickling SB3.

This is not a driving-quality scorer or a backup validator. Inspect a run leaf
directory (``result.json`` for smoke, or ``training_summary.json`` for rehearsal).
Missing or invalid evidence produces failed checks rather than preventing backup.
Torch is imported only when reading checkpoint tensors; ffmpeg/ffprobe are local
optional executables. No tracking SDK, GPU, network, or service credentials are used.
"""

import hashlib
import io
import json
import math
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_POLICY_BYTES = 256 * 1024 * 1024
MEDIA_TIMEOUT_SECONDS = 30
STATE_TOLERANCE = 1e-6
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def _file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != relative
        or "\\" in relative
    ):
        raise ValueError("Artifact path must be canonical and relative to the run")
    target = root.joinpath(*path.parts)
    if target.resolve() != target or not target.is_file():
        raise ValueError(f"Missing, nonregular, or symlink artifact: {relative}")
    return target


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Evidence JSON contains a duplicate object key")
        result[key] = value
    return result


def _json(root: Path, name: str) -> dict[str, Any]:
    path = _file(root, name)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds size limit: {name}")
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {name}")
    _finite_json(value)
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected finite numeric values")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected finite numeric values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(path: Path) -> dict[str, Any]:
    import torch

    # Never load the SB3 `data` member: it may contain cloudpickled Python objects.
    with zipfile.ZipFile(path) as archive:
        entries = [info for info in archive.infolist() if info.filename == "policy.pth"]
        if len(entries) != 1 or not 0 < entries[0].file_size <= MAX_POLICY_BYTES:
            raise ValueError("Checkpoint needs one bounded, nonempty policy.pth member")
        state = torch.load(
            io.BytesIO(archive.read(entries[0])), map_location="cpu", weights_only=True
        )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("policy.pth must contain a nonempty tensor state dict")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Policy state keys must be strings and all values tensors")
        if value.layout != torch.strided or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Policy tensor must be dense and finite: {key}")
    return dict(state)


def _checkpoints(root: Path, final_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    import torch

    initial_path, final_path = _file(root, "initial.zip"), _file(root, final_name)
    initial, final = _policy(initial_path), _policy(final_path)
    if initial.keys() != final.keys():
        raise ValueError("Initial/final policy state dict keys differ")
    changed, excluded = [], []
    for key, before in initial.items():
        after = final[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Initial/final policy tensor shape or dtype differs: {key}")
        if key.rsplit(".", 1)[-1] in BN_BUFFER_SUFFIXES:
            excluded.append(key)
        elif (before.is_floating_point() or before.is_complex()) and not torch.equal(before, after):
            changed.append(key)
    if not changed:
        raise ValueError("No policy tensor changed apart from excluded BatchNorm buffers")
    hashes = {"initial": _sha256(initial_path), "final": _sha256(final_path)}
    for phase, actual in hashes.items():
        declared = summary.get(f"{phase}_checkpoint_sha256")
        if declared is not None and declared != actual:
            raise ValueError(f"{phase} checkpoint SHA256 differs from training summary")
    return {
        "changed_tensor_names": changed,
        "excluded_batchnorm_buffers": excluded,
        "tensor_count": len(initial),
        "sha256": hashes,
        "note": "State tensors compared on CPU; ZIP-byte differences are not update evidence",
    }


def _motion(summary: dict[str, Any], name: str, num_cars: int) -> dict[str, Any]:
    distances = summary.get(name)
    if not isinstance(distances, list) or len(distances) != num_cars:
        raise ValueError(f"{name} must contain exactly {num_cars} per-car distances")
    distances = [_number(value) for value in distances]
    if any(value <= 0 for value in distances):
        raise ValueError(
            "Every car needs positive reset-excluded motion; max/total is insufficient"
        )
    return {"per_car_m": distances, "note": "Recorded travel is not route progress or safe driving"}


def _state_rows(state: dict[str, Any], num_cars: int) -> dict[str, list[list[float]]]:
    if set(state) != {"root_state", "joint_pos", "joint_vel"}:
        raise ValueError("Initial state requires root_state, joint_pos, and joint_vel")
    result = {}
    for key, rows in state.items():
        if not isinstance(rows, list) or len(rows) != num_cars:
            raise ValueError(f"Initial {key} needs exactly {num_cars} rows")
        if any(not isinstance(row, list) or not row for row in rows):
            raise ValueError(f"Initial {key} rows must be nonempty arrays")
        width = len(rows[0])
        if any(len(row) != width for row in rows) or (key == "root_state" and width != 13):
            raise ValueError(f"Initial {key} rows have inconsistent or invalid widths")
        result[key] = [[_number(value) for value in row] for row in rows]
    if len(result["joint_pos"][0]) != len(result["joint_vel"][0]):
        raise ValueError("Joint position/velocity state widths differ")
    return result


def _comparison(root: Path, num_cars: int, config: dict[str, Any]) -> dict[str, Any]:
    comparison = _json(root, "comparison.json")
    if comparison.get("matched_initial_states") is not True:
        raise ValueError("comparison.json does not declare matched initial states")
    start, end = (_json(root, f"{phase}_evaluation.json") for phase in ("start", "end"))
    if start.get("phase") != "start" or end.get("phase") != "end":
        raise ValueError("Evaluation phases do not match their filenames")
    for field in ("seed", "control_steps"):
        if (
            type(start.get(field)) is not int
            or type(end.get(field)) is not int
            or start[field] != end[field]
        ):
            raise ValueError(f"Evaluation {field} differs or is not an integer")
    _positive_int(start["control_steps"], "Evaluation control_steps")
    _positive_int(config.get("eval_steps"), "Configured eval_steps")
    if type(config.get("eval_seed")) is not int or not 0 <= config["eval_seed"] < 2**32:
        raise ValueError("Configured eval_seed must be an integer in [0, 2**32)")
    if start["control_steps"] != config.get("eval_steps") or start["seed"] != config.get(
        "eval_seed"
    ):
        raise ValueError("Evaluation control steps or seed differs from saved run config")
    if start.get("deterministic") is not True or end.get("deterministic") is not True:
        raise ValueError("Both evaluations must be deterministic")
    for phase, summary, checkpoint in (("start", start, "initial.zip"), ("end", end, "final.zip")):
        if comparison.get(phase) != summary:
            raise ValueError(f"Comparison contains stale {phase} evaluation metrics")
        if summary.get("checkpoint_sha256") != _sha256(_file(root, checkpoint)):
            raise ValueError(f"{phase} evaluation does not identify the actual checkpoint")
    states = [
        _state_rows(_json(root, f"{phase}_initial_state.json"), num_cars)
        for phase in ("start", "end")
    ]
    for key, rows in states[0].items():
        other = states[1][key]
        if len(rows[0]) != len(other[0]) or any(
            not math.isclose(a, b, rel_tol=STATE_TOLERANCE, abs_tol=STATE_TOLERANCE)
            for row_a, row_b in zip(rows, other, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        ):
            raise ValueError(f"Actual initial {key} values differ")
    return {"matched": True, "absolute_and_relative_tolerance": STATE_TOLERANCE}


def _probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise ValueError("Local ffprobe and ffmpeg are required for decoded video evidence")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            "-f",
            "mov",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=MEDIA_TIMEOUT_SECONDS,
    )
    if probe.stderr:
        raise ValueError("Video probe reported decoding errors")
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected decodable video stream")
    frames = int(streams[0].get("nb_read_frames", 0))
    if frames != expected_frames:
        raise ValueError(f"Decoded frame count {frames} differs from expected {expected_frames}")
    interval = max(1, frames // 8)
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-xerror",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"select=not(mod(n\\,{interval})),scale=64:64",
            "-fps_mode",
            "vfr",
            "-frames:v",
            "8",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=MEDIA_TIMEOUT_SECONDS,
    ).stdout
    if not decoded or len(decoded) % 4096:
        raise ValueError("Video sample decoding produced no complete frames")
    samples = [decoded[index : index + 4096] for index in range(0, len(decoded), 4096)]
    nonblank = sum(max(frame) - min(frame) >= 8 for frame in samples)
    if not nonblank:
        raise ValueError("All sampled video frames are spatially blank")
    return {
        "decoded_frames": frames,
        "expected_frames": expected_frames,
        "sampled_frames": len(samples),
        "nonblank_samples": nonblank,
        "note": "Spatial contrast only; camera correctness and driving quality require review",
    }


def inspect_run(run_directory: Path) -> dict[str, Any]:
    """Return a JSON-serializable report; a failed run may still be backed up safely.

    ``ppo_updates`` is the producer's recorded optimizer count, cross-checked
    against changed finite policy tensors; this does not reconstruct optimizer
    execution. BatchNorm buffers alone cannot pass. Initial-state matching uses
    a fixed 1e-6 tolerance, never a caller-supplied permissive comparison tolerance.
    """
    root = run_directory.absolute()
    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        try:
            details = operation()
        except Exception as error:
            checks.append({"name": name, "status": "fail", "reason": str(error)[:500]})
            return None
        checks.append({"name": name, "status": "pass", "details": details})
        return details

    kind = "smoke" if (root / "result.json").is_file() else "rehearsal"
    filename = "result.json" if kind == "smoke" else "training_summary.json"
    summary = check("run_summary", lambda: _json(root, filename))
    config = summary if kind == "smoke" else check("run_config", lambda: _json(root, "config.json"))
    if summary is not None:
        check(
            "ppo_update_count",
            lambda: {"ppo_updates": _positive_int(summary.get("ppo_updates"), "ppo_updates")},
        )
        final_name = "final.zip"
        if kind == "smoke" and not (root / final_name).exists():
            final_name = "fixture_ppo.zip"
        check("policy_tensor_update", lambda: _checkpoints(root, final_name, summary))
    if config is not None:
        dimensions = check(
            "car_count", lambda: {"num_cars": _positive_int(config.get("num_envs"), "num_envs")}
        )
        if dimensions:
            num_cars = dimensions["num_cars"]
            if kind == "smoke" and summary is not None:
                check(
                    "per_car_motion", lambda: _motion(summary, "nonreset_path_length_m", num_cars)
                )
            elif kind == "rehearsal":
                for phase in ("start", "end"):
                    check(
                        f"{phase}_per_car_motion",
                        lambda phase=phase: _motion(
                            _json(root, f"{phase}_evaluation.json"),
                            "nonreset_distance_per_car_m",
                            num_cars,
                        ),
                    )
                check("matched_evaluation", lambda: _comparison(root, num_cars, config))

        def media_spec() -> dict[str, Any]:
            if kind == "smoke":
                videos = config.get("videos")
                steps = _positive_int(config.get("steps"), "Smoke control steps")
                if (
                    not isinstance(videos, list)
                    or len(videos) != 1
                    or not isinstance(videos[0], dict)
                    or videos[0].get("path") != "smoke.mp4"
                    or videos[0].get("expected_frames") != steps
                ):
                    raise ValueError(
                        "Smoke needs one smoke.mp4 video with expected_frames equal to control steps"
                    )
            else:
                steps = _positive_int(config.get("eval_steps"), "eval_steps")
                stride = _positive_int(config.get("video_stride"), "video_stride")
                videos = [
                    {"path": f"{phase}.mp4", "expected_frames": (steps + stride - 1) // stride}
                    for phase in ("start", "end")
                ]
            for video in videos:
                if not isinstance(video, dict) or not isinstance(video.get("path"), str):
                    raise ValueError("Video evidence needs a relative path and expected_frames")
                _positive_int(video.get("expected_frames"), "expected_frames")
            return {"videos": videos}

        specs = check("video_expectations", media_spec)
        if specs:
            for spec in specs["videos"]:
                check(
                    f"video:{spec['path']}",
                    lambda spec=spec: _probe_video(
                        _file(root, spec["path"]), spec["expected_frames"]
                    ),
                )
    return {
        "schema_version": 1,
        "kind": kind,
        "checks": checks,
        "integration_passed": bool(checks) and all(item["status"] == "pass" for item in checks),
        "driving_proven": False,
        "artifact_preservation_recommended": True,
        "limitations": [
            "Local saved evidence only; not an independent simulator rerun or billing check",
            "A failed/incomplete run can still have a valid backup; backup integrity is separate",
            "Recorded reset-excluded distances are not recomputed from trajectories",
            "Motion, reward increases, and optimizer updates do not establish safe driving or convergence",
            "Initial/final observation PNGs and camera content still require human visual review",
            "Finite tensor checks and restricted loading are not a sandbox for hostile checkpoint files",
        ],
    }
