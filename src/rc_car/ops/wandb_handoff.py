"""Operational recovery: prepare a later, explicitly authorized W&B upload.

This module does not import W&B, consult authentication settings, edit its binary
stream, or upload anything. It consumes an output-bundle inventory. Matching
hashes prove listed bytes, not completeness, authenticity, driving quality, or
successful server ingestion. Keep the verified backup immutable; sync a copy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from rc_car.ops.bundles import FileEntry, _digest, _open_regular, _root, load_manifest

MODEL_FILES = ("initial.zip", "final.zip", "manifest.json", "config.json", "resolved_env.yaml")
EVALUATION_FILES = (
    "start_evaluation.json",
    "end_evaluation.json",
    "start_initial_state.json",
    "end_initial_state.json",
    "start_trajectory.jsonl",
    "end_trajectory.jsonl",
    "comparison.json",
    "events.jsonl",
)
EVALUATION_MEDIA = ("start.mp4", "end.mp4", "start-vs-end.mp4", "training-and-evaluation.png")
MAX_PUBLIC_JSON_BYTES = 1024 * 1024
FORBIDDEN_COMPONENTS = {
    ".env",
    ".netrc",
    ".aws",
    ".ssh",
    "credentials",
    "credentials.json",
    "api_key",
    "apikey",
    "secrets",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    "settings",
}


def _credential_like(name: str) -> bool:
    return any(
        part.lower() in FORBIDDEN_COMPONENTS
        or part.lower().startswith(".env.")
        or PurePosixPath(part.lower()).suffix in {".key", ".pem"}
        for part in PurePosixPath(name).parts
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate public JSON key")
        result[key] = value
    return result


def _public_json(root: Path, name: str, entries: dict[str, FileEntry]) -> dict[str, Any]:
    if name not in entries:
        raise ValueError(f"Required public file is not in the verified inventory: {name}")
    with _open_regular(root, name) as stream:
        data = stream.read(MAX_PUBLIC_JSON_BYTES + 1)
    if len(data) > MAX_PUBLIC_JSON_BYTES:
        raise ValueError("Public JSON exceeds the 1 MiB limit")
    if hashlib.sha256(data).hexdigest() != entries[name].sha256:
        raise ValueError("Public JSON changed after backup verification")
    value = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Public JSON must be an object")
    return value


def _run_prefix(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("run_directory must be a canonical relative POSIX directory")
    return value + "/"


def _artifact_entry(root: Path, entry: FileEntry, name: str) -> dict[str, Any]:
    """Compute the SDK's content digest, without constructing an SDK artifact."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with _open_regular(root, entry.path) as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    if size != entry.size or sha256.hexdigest() != entry.sha256:
        raise ValueError("Artifact file changed after backup verification")
    return {
        "name": name,
        "path": entry.path,
        "size": size,
        "sha256": entry.sha256,
        "wandb_content_digest": base64.b64encode(md5.digest()).decode("ascii"),
    }


def prepare_handoff(
    backup_root: Path, manifest_path: Path, *, run_directory: str
) -> dict[str, Any]:
    """Verify a copied job and return a plan, never an upload-success receipt.

    ``run_directory`` is relative to ``backup_root`` (normally ``experiment``).
    The manifest must list public identity/config and exactly one matching
    offline stream. Missing final/evaluation outputs mark groups incomplete;
    partial runs can still be valuable verified backups. Unlisted files are
    never included. The binary stream and its artifact references are NOT
    decoded here, so inventory membership alone cannot establish portability.
    """
    prefix = _run_prefix(run_directory)
    # Guard exactly the immutable manifest object subsequently verified; do not
    # reload it after checks, which would permit a manifest-swap window.
    manifest = load_manifest(manifest_path)
    if any(_credential_like(entry.path) for entry in manifest.files):
        raise ValueError("Credential-like inventory member: inspect the backup manually")
    root = _root(backup_root)
    for entry in manifest.files:
        with _open_regular(root, entry.path) as stream:
            size, digest = _digest(stream)
        if (size, digest) != (entry.size, entry.sha256):
            raise ValueError(f"Directory file checksum mismatch: {entry.path}")
    entries = {entry.path: entry for entry in manifest.files}
    identity = _public_json(root, prefix + "wandb_run.json", entries)
    if set(identity) not in (
        {"id", "url", "entity", "project", "mode"},
        {"id", "url", "entity", "project", "mode", "name"},
    ):
        raise ValueError("Unexpected public tracking identity fields")
    if (
        "name" in identity
        and identity["name"] is not None
        and (not isinstance(identity["name"], str) or not identity["name"].strip())
    ):
        raise ValueError("Invalid tracking display name")
    for key in ("id", "entity", "project"):
        if not isinstance(identity[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity[key]):
            raise ValueError(f"Unsupported public tracking {key}")
    if identity["mode"] != "offline" or identity["url"] is not None:
        raise ValueError("Expected an offline run identity with no server URL")
    config = _public_json(root, prefix + "config.json", entries)
    destination = config.get("wandb")
    if not isinstance(destination, dict) or any(
        destination.get(key) != identity[key] for key in ("entity", "project")
    ):
        raise ValueError("Saved config and public tracking destination differ")

    streams = [
        entry
        for name, entry in entries.items()
        if name.startswith(prefix + "wandb/") and name.endswith(".wandb")
    ]
    if len(streams) != 1:
        raise ValueError("Expected exactly one inventoried W&B binary stream for this experiment")
    record = streams[0]
    record_path = PurePosixPath(record.path)
    relative_parts = record_path.relative_to(prefix + "wandb").parts
    if (
        len(relative_parts) != 2
        or not relative_parts[0].startswith("offline-run-")
        or not relative_parts[0].endswith("-" + identity["id"])
        or relative_parts[1] != "run-" + identity["id"] + ".wandb"
        or record.size == 0
    ):
        raise ValueError("Offline stream filename/layout does not match the public run identity")

    groups: dict[str, Any] = {}
    for group, names in (
        ("model", MODEL_FILES),
        ("evaluation", EVALUATION_FILES),
        ("evaluation_media", EVALUATION_MEDIA),
    ):
        available = [name for name in names if prefix + name in entries]
        empty = [name for name in available if entries[prefix + name].size == 0]
        groups[group] = {
            "complete": len(available) == len(names) and not empty,
            "missing": [name for name in names if name not in available],
            "empty": empty,
            "files": [_artifact_entry(root, entries[prefix + name], name) for name in available],
        }
    basename = PurePosixPath(run_directory).name
    groups["model"].update(artifact_name=basename + "-checkpoints", artifact_type="model")
    groups["evaluation"].update(artifact_name=basename + "-evaluation", artifact_type="evaluation")
    sdk_files_prefix = record_path.parent.as_posix() + "/files/"
    transport = [
        entry.model_dump()
        for name, entry in sorted(entries.items())
        if name == record.path or name.startswith(sdk_files_prefix)
    ]
    return {
        "schema_version": 1,
        "kind": "offline-wandb-handoff-plan",
        "backup_root": str(root),
        "run_directory": run_directory,
        "listed_backup_bytes_verified": True,
        "backup_file_count": len(entries),
        "identity": identity,
        "offline_record": record.path,
        "transport_files": transport,
        "reupload_groups": groups,
        "all_reupload_groups_complete": all(group["complete"] for group in groups.values()),
        "uploaded": False,
        "server_verified": False,
        "limitations": [
            "The binary stream is unchanged and not parsed; its identity/content are not verified.",
            "The inventory proves listed bytes only, not complete media or artifact references.",
            "Credential-like filenames are rejected before member reads; file contents are not "
            "scanned for secrets. Review the selected public outputs before upload.",
            "GPU artifact records can retain inaccessible absolute staging paths after copying.",
            "Stock SDK 0.25.1 sync has no artifact-skip or staging-remap option.",
            "Sync exit status or a .synced marker does not prove artifact upload success.",
            "No media decoding, checkpoint loading, driving, or convergence test is performed.",
            "Reverify bytes immediately before any future upload; this plan is not authorization.",
        ],
        "later_authorized_steps": [
            "Keep the verified backup unchanged; create a disposable full-job working copy.",
            "Run one bounded, explicit-path stock sync for this run's metrics/media; do not use "
            "--sync-all or --clean. Artifact errors may occur; retain the sync log.",
            "Resume only the recorded entity/project/id with mode='online' and resume='must'; "
            "verify the returned identity. Re-log complete model/evaluation groups from their "
            "verified local files and attach evaluation media. Preserve original names.",
            "Wait for artifacts and verify server manifest entry names, sizes and content "
            "digests against this plan; check scalar history/media independently. Do not claim "
            "success from the CLI exit alone. Validate this flow with a synthetic online run first.",
        ],
    }
