"""Operational recovery: read-only checks of W&B synchronization evidence.

Only ``verify_online_run`` consults the supplied public API run. This module never
logs in, initializes a run, uploads, rewrites streams, or reads credentials. The
binary reader intentionally targets the repository's pinned SDK 0.25.1. Its
private datastore normally opens files ``r+b``; here its scanner receives an
``rb`` file handle instead. A receipt establishes selected server evidence, not
model quality or artifact completeness (artifact manifests need separate checks).
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

SDK_VERSION = "0.25.1"
REL_TOL = 1e-6
ABS_TOL = 1e-8


def _identity(value: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for key in ("id", "entity", "project"):
        part = value.get(key)
        if not isinstance(part, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", part):
            raise ValueError(f"Invalid public run identity field: {key}")
        result[key] = part
    return result


def _item_path(item: Any) -> list[str]:
    nested = list(item.nested_key)
    if item.key and nested:
        raise ValueError("Ambiguous protobuf item key")
    parts = nested or [item.key]
    if any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("Empty protobuf item key")
    return parts


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite public metric in saved stream")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _set(root: dict[str, Any], parts: list[str], value: Any, *, unique: bool = False) -> None:
    current = root
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ValueError("Conflicting nested protobuf item keys")
        current = current[part]
    if unique and parts[-1] in current:
        raise ValueError("Duplicate protobuf item key")
    current[parts[-1]] = value


def _remove(root: dict[str, Any], parts: list[str]) -> None:
    current = root
    for part in parts[:-1]:
        current = current.get(part, {})
        if not isinstance(current, dict):
            return
    current.pop(parts[-1], None)


def _read_items(items: Iterable[Any], root: dict[str, Any], *, unique: bool = False) -> None:
    for item in items:
        parts = _item_path(item)
        # Media's nested _type is public evidence; exclude only SDK top-level keys.
        if parts[0].startswith("_") and parts != ["_step"]:
            continue
        value = json.loads(item.value_json)
        _finite(value)
        _set(root, parts, value, unique=unique)


def _hash_file(path: Path) -> tuple[int, str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    return size, sha256.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def _media_entry(files_root: Path, step: int, key: str, value: dict[str, Any]) -> dict[str, Any]:
    name = value.get("path")
    if not isinstance(name, str):
        raise ValueError("Video record has no relative path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
        or "\\" in name
        or not name.startswith("media/videos/")
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("Unsafe or unsupported training video path")
    candidate = files_root
    for component in relative.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError("Symlink in training video path")
    size, sha256, md5 = _hash_file(candidate)
    if type(value.get("size")) is not int or (size, sha256) != (
        value.get("size"),
        value.get("sha256"),
    ):
        raise ValueError("Training video bytes disagree with the saved stream")
    return {"step": step, "key": key, "path": name, "size": size, "sha256": sha256, "md5": md5}


def inspect_stream(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a completed saved stream and verify its adjacent video bytes.

    ``path`` is the original ``run-ID.wandb`` file, with media under its sibling
    ``files/`` directory. Returns public scalar/media evidence only: run configs,
    environment records, console logs, artifact staging paths and credentials are
    not included. Refuse malformed, truncated, failed or ambiguous streams.
    """
    if importlib.metadata.version("wandb") != SDK_VERSION:
        raise RuntimeError(f"Saved stream verification requires W&B {SDK_VERSION}")
    identity = _identity(expected_identity)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular saved stream file")
    size, digest, _ = _hash_file(path)
    datastore_module = importlib.import_module("wandb.sdk.internal.datastore")
    protobuf = importlib.import_module("wandb.proto.wandb_internal_pb2")
    scanner = datastore_module.DataStore()
    history: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    counts: dict[str, int] = {}
    exit_code: int | None = None
    with path.open("rb") as stream:
        scanner._fname = str(path)
        scanner._fp = stream
        scanner._index = 0
        scanner._size_bytes = size
        scanner._opened_for_scan = True
        try:
            scanner._read_header()
            while True:
                before = stream.tell()
                data = scanner.scan_data()
                if data is None:
                    if stream.tell() != before:
                        raise ValueError("Truncated saved stream")
                    break
                # Generated protobuf exports vary with the installed protobuf runtime.
                record = cast(Any, protobuf).Record()
                record.ParseFromString(data)
                kind = record.WhichOneof("record_type")
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "run":
                    actual = {
                        "id": record.run.run_id,
                        "entity": record.run.entity,
                        "project": record.run.project,
                    }
                    if actual != identity:
                        raise ValueError("Saved stream identity differs from expected identity")
                elif kind == "history":
                    row: dict[str, Any] = {}
                    _read_items(record.history.item, row, unique=True)
                    step = row.get("_step")
                    if type(step) is not int or step < 0 or step != record.history.step.num:
                        raise ValueError("Invalid or inconsistent saved history step")
                    if history and step <= history[-1]["_step"]:
                        raise ValueError("Saved history steps are not strictly increasing")
                    history.append(row)
                elif kind == "summary":
                    _read_items(record.summary.update, summary)
                    for item in record.summary.remove:
                        _remove(summary, _item_path(item))
                elif kind == "exit":
                    exit_code = record.exit.exit_code
        except (AssertionError, IndexError) as error:
            raise ValueError("Corrupt or truncated saved stream") from error
    if counts.get("run") != 1 or counts.get("exit") != 1 or exit_code != 0 or not history:
        raise ValueError("Expected one completed successful run with nonempty history")
    summary.pop("_step", None)
    media = []
    for row in history:
        for key, value in row.items():
            if isinstance(value, dict) and value.get("_type") == "video-file":
                media.append(_media_entry(path.parent / "files", row["_step"], key, value))
    # Detect replacement or modification while decoding, without changing the backup.
    if _hash_file(path)[:2] != (size, digest):
        raise ValueError("Saved stream changed during inspection")
    return {
        "schema_version": 1,
        "kind": "saved-wandb-stream-expectation",
        "sdk_version": SDK_VERSION,
        "identity": identity,
        "stream_sha256": digest,
        "stream_bytes": size,
        "record_counts": counts,
        "history": history,
        "summary": summary,
        "media": media,
        "source_completed_successfully": True,
    }


def _compare(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            return False
        try:
            return all(_compare(value, actual[key]) for key, value in expected.items())
        except (KeyError, TypeError, IndexError):
            return False
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_compare(left, right) for left, right in zip(expected, actual, strict=True))
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if type(expected) in (int, float):
        if type(actual) not in (int, float) or not math.isfinite(actual):
            return False
        if isinstance(expected, int):
            return actual == expected
        return math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return False


def normalize_summary(value: Any) -> dict[str, Any]:
    """Normalize public SDK summaries without treating media metadata as scalars.

    W&B 0.25.1 can return nested scalar summaries as dotted top-level keys while
    retaining media dictionaries. Public API ``SummarySubDict`` exposes the
    underlying JSON via ``_json_dict`` rather than implementing ``Mapping``.
    Normalize both saved expectations and server values; never split literal
    slash keys. Ambiguous dotted/nested collisions fail instead of overwriting.
    """
    result: dict[str, Any] = {}

    def plain(item: Any) -> Any:
        if not isinstance(item, Mapping):
            item = getattr(item, "_json_dict", item)
        if isinstance(item, Mapping):
            return {key: plain(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [plain(nested) for nested in item]
        return item

    def flatten(prefix: str, item: Any) -> None:
        if isinstance(item, dict) and item and "_type" not in item:
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError("Summary keys must be strings")
                flatten(f"{prefix}.{key}" if prefix else key, nested)
        else:
            if prefix in result:
                raise ValueError("Ambiguous dotted/nested summary keys")
            result[prefix] = item

    public = plain(value)
    if not isinstance(public, dict):
        raise ValueError("Expected a public summary mapping")
    for key, nested in public.items():
        if not isinstance(key, str):
            raise ValueError("Summary keys must be strings")
        if not key.startswith("_"):
            flatten(key, nested)
    return result


def verify_online_run(expected: Mapping[str, Any], remote_run: Any) -> dict[str, Any]:
    """Read server history/media metadata and return a bounded verification receipt.

    The caller constructs a fresh ``wandb.Api().run(...)`` after authorized sync.
    Unlike sampled ``history()``, ``scan_history()`` with no keys returns all rows.
    Original rows must match by their exact SDK step; later appended evaluation
    rows are allowed. Final summary is checked independently: an unoptimized
    partial rollout can make final sample counts exceed the last training row.
    Media verification checks recorded SHA256/size plus server file size/MD5;
    it does not download or decode remote videos or verify model artifacts.
    API failures propagate: they are not successful receipts.
    """
    if (
        expected.get("kind") != "saved-wandb-stream-expectation"
        or expected.get("schema_version") != 1
        or not expected.get("source_completed_successfully")
        or not expected.get("history")
    ):
        raise ValueError("Expected a nonempty completed saved-stream inspection")
    identity = _identity(expected["identity"])
    errors: list[str] = []
    for key in ("id", "entity", "project"):
        if getattr(remote_run, key, None) != identity[key]:
            errors.append(f"identity/{key}")
    # Do not scan history on a different run, even if the caller resolved it incorrectly.
    if errors:
        return {
            "kind": "wandb-server-verification",
            "identity": identity,
            "passed": False,
            "errors": errors,
            "server_verified": False,
        }
    if remote_run.state != "finished":
        errors.append("state/not-finished")
    rows: dict[int, dict[str, Any]] = {}
    for row in remote_run.scan_history():
        step = row.get("_step")
        if type(step) is not int or step < 0 or step in rows:
            errors.append("history/invalid-or-duplicate-step")
            continue
        rows[step] = row
    for row in expected["history"]:
        step = row["_step"]
        if step not in rows:
            errors.append(f"history/{step}/missing")
        elif not _compare(row, rows[step]):
            errors.append(f"history/{step}/mismatch")
    expected_summary = normalize_summary(expected["summary"])
    actual_summary = normalize_summary(remote_run.summary)
    for key, value in expected_summary.items():
        if key not in actual_summary:
            errors.append(f"summary/{key}/missing")
            continue
        if not _compare(value, actual_summary[key]):
            errors.append(f"summary/{key}/mismatch")
    files: dict[str, Any] = {}
    if expected["media"]:
        for file in remote_run.files():
            if file.name in files:
                errors.append("files/duplicate-name")
            files[file.name] = file
    for entry in expected["media"]:
        file = files.get(entry["path"])
        if file is None:
            errors.append(f"media/{entry['step']}/missing-file")
            continue
        # Public File.md5 is base64 in the pinned API; accept its hex spelling too.
        md5 = getattr(file, "md5", None)
        expected_md5 = entry["md5"]
        hex_md5 = base64.b64decode(expected_md5, validate=True).hex()
        if file.size != entry["size"] or md5 not in (expected_md5, hex_md5):
            errors.append(f"media/{entry['step']}/file-digest-or-size-mismatch")
    return {
        "schema_version": 1,
        "kind": "wandb-server-verification",
        "identity": identity,
        "source_stream_sha256": expected["stream_sha256"],
        "passed": not errors,
        "server_verified": not errors,
        "errors": errors,
        "expected_history_rows": len(expected["history"]),
        "server_history_rows": len(rows),
        "expected_summary_fields": len(expected_summary),
        "expected_training_videos": len(expected["media"]),
        "float_tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "limitations": [
            "Checks original history, public summary and training-media metadata only.",
            "Model/evaluation artifacts require separate server manifest verification.",
            "Remote media was not downloaded or decoded; this is not driving-quality evidence.",
        ],
    }
