"""Attach completed outputs to an existing W&B run and verify server metadata.

No network or SDK import occurs during import or ``validate_upload``. The upload
uses an isolated, exclusive working directory, never the source run directory.
Call from a dedicated process: W&B's artifact cache/data locations are process
environment settings. This does not sync or verify historical training metrics.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from rc_car.ops.bundles import _open_regular, _root
from rc_car.ops.wandb_handoff import EVALUATION_FILES, EVALUATION_MEDIA, MODEL_FILES, _unique_object
from rc_car.ops.wandb_sync import normalize_summary

UPLOAD_FILES = (*MODEL_FILES, *EVALUATION_FILES, *EVALUATION_MEDIA)
MEDIA_KEYS = {
    "evaluation/start_video": "start.mp4",
    "evaluation/end_video": "end.mp4",
    "evaluation/comparison_video": "start-vs-end.mp4",
    "evaluation/chart": "training-and-evaluation.png",
}


def _equal_summary(expected: Any, actual: Any) -> bool:
    """Compare JSON summaries with bounded float serialization tolerance only."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_equal_summary(value, actual[key]) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _equal_summary(left, right) for left, right in zip(expected, actual, strict=True)
            )
        )
    if type(expected) in (bool, str) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if type(expected) in (int, float):
        if type(actual) not in (int, float) or not math.isfinite(actual):
            return False
        if isinstance(expected, int):
            return actual == expected
        return math.isfinite(expected) and math.isclose(
            expected, actual, rel_tol=1e-6, abs_tol=1e-8
        )
    return False


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(root: Path, name: str) -> dict[str, Any]:
    with _open_regular(root, name) as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError(f"Public JSON exceeds 1 MiB: {name}")
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Public JSON must be an object: {name}")
    return value


def _hash(root: Path, name: str) -> dict[str, Any]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with _open_regular(root, name) as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            md5.update(block)
            size += len(block)
    if size == 0:
        raise ValueError(f"Required upload file is empty: {name}")
    return {
        "size": size,
        "sha256": sha.hexdigest(),
        "wandb_content_digest": base64.b64encode(md5.digest()).decode("ascii"),
    }


def validate_upload(run_dir: str | Path) -> dict[str, Any]:
    """Validate and hash the fixed public allowlist, without SDK or network."""
    root = _root(run_dir)
    identity = _json(root, "wandb_run.json")
    if set(identity) not in (
        {"id", "entity", "project", "mode", "url"},
        {"id", "entity", "project", "mode", "url", "name"},
    ):
        raise ValueError("Unexpected tracking identity fields")
    if (
        "name" in identity
        and identity["name"] is not None
        and (not isinstance(identity["name"], str) or not identity["name"].strip())
    ):
        raise ValueError("Invalid tracking display name")
    for key in ("id", "entity", "project"):
        if not isinstance(identity[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity[key]):
            raise ValueError(f"Invalid tracking {key}")
    if identity["mode"] not in ("offline", "online"):
        raise ValueError("Only offline or online run identities can be uploaded")
    config = _json(root, "config.json")
    destination = config.get("wandb")
    if not isinstance(destination, dict) or any(
        destination.get(key) != identity[key] for key in ("entity", "project")
    ):
        raise ValueError("Config and tracking destinations differ")
    files = {name: _hash(root, name) for name in UPLOAD_FILES}
    files["wandb_run.json"] = _hash(root, "wandb_run.json")
    summaries = {phase: _json(root, f"{phase}_evaluation.json") for phase in ("start", "end")}
    comparison = _json(root, "comparison.json")
    if type(comparison.get("matched_initial_states")) is not bool:
        raise ValueError("Comparison must state whether initial states match")
    return {
        "root": str(root),
        "identity": identity,
        "files": files,
        "summaries": summaries,
        "matched_initial_states": comparison["matched_initial_states"],
    }


def _identity(run: Any, expected: dict[str, Any]) -> None:
    if any(getattr(run, key, None) != expected[key] for key in ("id", "entity", "project")):
        raise RuntimeError("W&B returned a different run identity")


@contextmanager
def _sdk_directories(work: Path):
    values = {
        "WANDB_DIR": str(work / "sdk"),
        "WANDB_DATA_DIR": str(work / "data"),
        "WANDB_CACHE_DIR": str(work / "cache"),
    }
    previous = {key: os.environ.get(key) for key in values}
    for path in values.values():
        Path(path).mkdir()
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _outside_source(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError("SDK work and receipts must be outside the source run directory")
    return resolved


def _verify_artifact(server: Any, expected: dict[str, Any], names: tuple[str, ...]) -> None:
    entries = server.manifest.entries
    if set(entries) != set(names):
        raise RuntimeError("Server artifact member names differ")
    for name in names:
        entry = entries[name]
        if (entry.size, entry.digest) != (
            expected[name]["size"],
            expected[name]["wandb_content_digest"],
        ):
            raise RuntimeError(f"Server artifact content metadata differs: {name}")


def upload_completed_experiment(
    run_dir: str | Path,
    *,
    work_dir: str | Path | None = None,
    receipt_path: str | Path | None = None,
    sdk: Any = None,
    artifact_timeout: int = 120,
) -> dict[str, Any]:
    """Attach outputs to an existing run; never create a replacement run.

    Offline runs must first be synced by the caller. Server metadata verification
    is not a redownload or historical training-metrics verification. Failed
    uploads retain a partial receipt without potentially secret error text.
    Every attempt needs fresh work/receipt paths. Also use a process timeout.
    """
    plan = validate_upload(run_dir)
    root = Path(plan["root"])
    if type(artifact_timeout) is not int or artifact_timeout <= 0:
        raise ValueError("artifact_timeout must be a positive integer")
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM"):
        if os.environ.get(key):
            raise ValueError(f"{key} conflicts with an explicit existing-run upload")
    work = _outside_source(
        Path(work_dir) if work_dir else root.parent / f"wandb-upload-{uuid.uuid4().hex}", root
    )
    receipt_file = _outside_source(
        Path(receipt_path) if receipt_path else work / "receipt.json", root
    )
    if receipt_file.exists() or receipt_file.is_symlink():
        raise FileExistsError("Upload receipt already exists")
    work.mkdir(parents=True, exist_ok=False)
    payload = work / "payload"
    payload.mkdir()
    for name, expected in plan["files"].items():
        with _open_regular(root, name) as source, (payload / name).open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if _hash(payload, name) != expected:
            raise ValueError(f"Source changed while preparing upload: {name}")
    copied = validate_upload(payload)
    if any(copied[key] != plan[key] for key in ("identity", "summaries", "matched_initial_states")):
        raise ValueError("Public identity/evaluation changed while preparing upload")
    expected_identity = plan["identity"]
    run_path = "/".join(expected_identity[key] for key in ("entity", "project", "id"))
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "wandb-completed-output-upload",
        "run_path": run_path,
        "source_files": plan["files"],
        "uploaded": False,
        "server_verified": False,
        "training_history_verified": False,
        "summary_float_tolerance": {"relative": 1e-6, "absolute": 1e-8},
        "verification_scope": "Evaluation summary, media file metadata and artifact manifests; "
        "no training-history verification or server-byte redownload",
        "artifacts": [],
    }
    try:
        with _sdk_directories(work):
            if sdk is None:
                sdk = importlib.import_module("wandb")
            api = sdk.Api(timeout=30)
            _identity(api.run(run_path), expected_identity)
            run = sdk.init(
                entity=expected_identity["entity"],
                project=expected_identity["project"],
                id=expected_identity["id"],
                mode="online",
                resume="must",
                dir=str(work / "sdk"),
                save_code=False,
                reinit="create_new",
                settings=sdk.Settings(init_timeout=60, disable_git=True, x_disable_stats=True),
            )
            try:
                _identity(run, expected_identity)
                for phase, summary in plan["summaries"].items():
                    run.summary[f"evaluation/{phase}"] = summary
                run.summary["matched_initial_states"] = plan["matched_initial_states"]
                run.log(
                    {
                        key: sdk.Image(str(payload / name))
                        if name.endswith(".png")
                        else sdk.Video(str(payload / name), format="mp4")
                        for key, name in MEDIA_KEYS.items()
                    }
                )
                for suffix, kind, names in (
                    ("checkpoints", "model", MODEL_FILES),
                    ("evaluation", "evaluation", EVALUATION_FILES),
                ):
                    artifact = sdk.Artifact(root.name + "-" + suffix, type=kind)
                    for name in names:
                        artifact.add_file(str(payload / name), name=name)
                    logged = run.log_artifact(artifact)
                    logged.wait(timeout=artifact_timeout)
                    server = api.artifact(logged.qualified_name, type=kind)
                    if server.id != logged.id:
                        raise RuntimeError("Server artifact identity differs")
                    _verify_artifact(server, plan["files"], names)
                    receipt["artifacts"].append(
                        {
                            "id": server.id,
                            "qualified_name": logged.qualified_name,
                            "manifest_verified": True,
                        }
                    )
                run.finish()
            except BaseException as error:
                try:
                    run.finish(exit_code=1)
                except BaseException:
                    error.add_note("W&B failed to finish the upload run")
                raise
            receipt["uploaded"] = True
            api.flush()
            remote = api.run(run_path)
            _identity(remote, expected_identity)
            linked = {artifact.id for artifact in remote.logged_artifacts()}
            if any(artifact["id"] not in linked for artifact in receipt["artifacts"]):
                raise RuntimeError("Uploaded artifacts are not linked to the expected run")
            remote_summary = normalize_summary(remote.summary)
            for phase, summary in plan["summaries"].items():
                prefix = f"evaluation/{phase}"
                expected_summary = normalize_summary({prefix: summary})
                actual_summary = {
                    key: value
                    for key, value in remote_summary.items()
                    if key == prefix or key.startswith(prefix + ".")
                }
                if not _equal_summary(expected_summary, actual_summary):
                    raise RuntimeError("Server evaluation summary differs")
            if not _equal_summary(
                plan["matched_initial_states"], remote_summary.get("matched_initial_states")
            ):
                raise RuntimeError("Server initial-state comparison differs")
            for key, name in MEDIA_KEYS.items():
                media = remote_summary.get(key)
                expected = plan["files"][name]
                if not isinstance(media, dict) or media.get("sha256") != expected["sha256"]:
                    raise RuntimeError(f"Server media summary digest differs: {key}")
                path = media.get("path")
                if (
                    not isinstance(path, str)
                    or PurePosixPath(path).is_absolute()
                    or ".." in PurePosixPath(path).parts
                ):
                    raise RuntimeError("Server returned an invalid media path")
                files = list(remote.files(names=[path]))
                if len(files) != 1 or (files[0].name, files[0].size, files[0].md5) != (
                    path,
                    expected["size"],
                    expected["wandb_content_digest"],
                ):
                    raise RuntimeError(f"Server media file metadata differs: {key}")
            receipt["server_verified"] = True
            receipt["url"] = remote.url
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        with receipt_file.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt
