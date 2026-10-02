"""Allowlisted local backups with checked tar members and no output replacement.

Callers must choose the regular files to include explicitly. These functions do
not search directories, collect credentials, contact a service, or authenticate
a sender: matching checksums establish consistency with the supplied manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Annotated, Any, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rc_car.lifecycle import best_effort_cleanup

CHUNK_SIZE = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _relative_name(value: str) -> str:
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
        raise ValueError("Bundle members must have canonical relative POSIX paths")
    return value


class FileEntry(BaseModel):
    """One explicitly selected regular file; modes never include special bits."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    path: str
    size: Annotated[int, Field(ge=0)]
    sha256: Digest
    mode: Annotated[int, Field(ge=0, le=0o755)]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _relative_name(value)

    @field_validator("mode")
    @classmethod
    def safe_mode(cls, value: int) -> int:
        if value & ~0o755:
            raise ValueError("Bundle file modes cannot grant group/other write permissions")
        return value


class BundleManifest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    archive_sha256: Digest
    files: tuple[FileEntry, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Bundle schema_version must be integer 1")
        return value

    @field_validator("files", mode="before")
    @classmethod
    def json_array(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        names = [entry.path for entry in self.files]
        if not names or len(names) != len(set(names)):
            raise ValueError("Bundle manifests require nonempty, unique file entries")
        names_set = set(names)
        for name in names:
            if any(str(parent) in names_set for parent in PurePosixPath(name).parents):
                raise ValueError("Bundle file paths conflict with parent directories")
        return self


@dataclass(frozen=True)
class BundlePaths:
    archive: Path
    manifest: Path


def _root(directory: str | Path) -> Path:
    path = Path(directory)
    if path.is_symlink():
        raise ValueError("Bundle root must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Bundle root must be a directory")
    return resolved


@contextmanager
def _open_regular(root: Path, relative: str) -> Iterator[BinaryIO]:
    """Open beneath an explicit root without following any member symlink.

    Descriptor-relative traversal protects both intermediate directories and the
    final file. Supported hosts are the project's POSIX Mac/Linux environments.
    """
    components = PurePosixPath(_relative_name(relative)).parts
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in components[:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"Not a regular bundle file: {relative}")
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream
    finally:
        os.close(directory_fd)


def _digest(stream: IO[bytes]) -> tuple[int, str]:
    total = 0
    digest = hashlib.sha256()
    while data := stream.read(CHUNK_SIZE):
        total += len(data)
        digest.update(data)
    return total, digest.hexdigest()


def inventory_files(
    root: str | Path, relative_paths: Iterable[str | Path]
) -> tuple[FileEntry, ...]:
    """Hash only the explicit regular-file allowlist; directories are not expanded."""
    base = _root(root)
    names = [
        _relative_name(path.as_posix() if isinstance(path, Path) else path)
        for path in relative_paths
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError("A nonempty explicit allowlist without duplicates is required")
    entries = []
    for name in sorted(names):
        with _open_regular(base, name) as stream:
            mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode) & 0o755
            size, digest = _digest(stream)
        entries.append(FileEntry(path=name, size=size, sha256=digest, mode=mode))
    return tuple(entries)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate bundle manifest key: {key!r}")
        result[key] = value
    return result


def load_manifest(path: str | Path) -> BundleManifest:
    """Read the sidecar schema, rejecting ambiguous JSON and unsafe paths."""
    target = Path(path)
    with _open_regular(_root(target.parent), target.name) as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Bundle manifest exceeds the 16 MiB limit")
    return BundleManifest.model_validate(json.loads(data, object_pairs_hook=_unique_object))


def verify_directory(root: str | Path, manifest_path: str | Path) -> BundleManifest:
    """Verify listed file bytes only; unrelated runtime/generated files are allowed."""
    base = _root(root)
    manifest = load_manifest(manifest_path)
    for entry in manifest.files:
        with _open_regular(base, entry.path) as stream:
            size, digest = _digest(stream)
        if (size, digest) != (entry.size, entry.sha256):
            raise ValueError(f"Directory file checksum mismatch: {entry.path}")
    return manifest


def _inspect_archive(
    archive_path: Path, manifest: BundleManifest, stage: Path | None = None
) -> None:
    expected = {entry.path: entry for entry in manifest.files}
    with _open_regular(_root(archive_path.parent), archive_path.name) as stream:
        if _digest(stream)[1] != manifest.archive_sha256:
            raise ValueError("Bundle archive checksum mismatch")
        stream.seek(0)
        seen: set[str] = set()
        with tarfile.open(fileobj=stream, mode="r:gz", ignore_zeros=True) as archive:
            for member in archive:
                name = _relative_name(member.name)
                if name in seen:
                    raise ValueError(f"Duplicate archive member: {name}")
                if name not in expected:
                    raise ValueError(f"Unexpected archive member: {name}")
                if (
                    member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or member.sparse is not None
                ):
                    raise ValueError(f"Archive member is not an ordinary regular file: {name}")
                entry = expected[name]
                if member.size != entry.size or member.mode != entry.mode:
                    raise ValueError(f"Archive member metadata mismatch: {name}")
                member_stream = archive.extractfile(member)
                if member_stream is None:
                    raise ValueError(f"Cannot read archive member: {name}")
                with member_stream:
                    if stage is None:
                        size, digest = _digest(member_stream)
                    else:
                        target = stage.joinpath(*PurePosixPath(name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as destination:
                            digest_object = hashlib.sha256()
                            size = 0
                            while data := member_stream.read(CHUNK_SIZE):
                                destination.write(data)
                                digest_object.update(data)
                                size += len(data)
                            digest = digest_object.hexdigest()
                if (size, digest) != (entry.size, entry.sha256):
                    raise ValueError(f"Archive member checksum mismatch: {name}")
                seen.add(name)
        if seen != expected.keys():
            raise ValueError("Archive is missing manifest entries")


def verify_bundle(archive_path: str | Path, manifest_path: str | Path) -> BundleManifest:
    """Check compressed bytes, exact members, modes, sizes, and file checksums."""
    manifest = load_manifest(manifest_path)
    try:
        _inspect_archive(Path(archive_path), manifest)
    except (tarfile.TarError, EOFError) as error:
        raise ValueError("Bundle archive is unreadable or truncated") from error
    return manifest


def create_bundle(
    root: str | Path, relative_paths: Iterable[str | Path], output_dir: str | Path
) -> BundlePaths:
    """Write and verify a fresh bundle directory; never replace previous output."""
    base = _root(root)
    entries = inventory_files(base, relative_paths)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    paths = BundlePaths(output / "bundle.tar.gz", output / "manifest.json")
    try:
        with tarfile.open(paths.archive, "x:gz", format=tarfile.PAX_FORMAT) as archive:
            for entry in entries:
                with _open_regular(base, entry.path) as stream:
                    info = tarfile.TarInfo(entry.path)
                    info.size, info.mode, info.mtime = entry.size, entry.mode, 0
                    archive.addfile(info, stream)
        with paths.archive.open("rb") as stream:
            archive_digest = _digest(stream)[1]
        manifest = BundleManifest(archive_sha256=archive_digest, files=entries)
        with paths.manifest.open("x", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2) + "\n")
        verify_bundle(paths.archive, paths.manifest)
        # Catch a checkpoint/source file that changed while the archive was made.
        verify_directory(base, paths.manifest)
    except BaseException as error:
        best_effort_cleanup(error, (("remove failed bundle", lambda: shutil.rmtree(output)),))
        raise
    return paths


def extract_bundle(
    archive_path: str | Path, manifest_path: str | Path, destination: str | Path
) -> BundleManifest:
    """Verify before extraction, stage privately, then publish to a fresh directory.

    A destination is reserved with exclusive mkdir because portable POSIX rename
    would replace an existing empty directory. Staged payload bytes are checked a
    second time before any destination is created. Unlisted files are never read
    or extracted, and existing destinations (including empty ones) are rejected.
    """
    manifest = verify_bundle(archive_path, manifest_path)
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Extraction destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=".bundle-extract-", dir=target.parent)
    try:
        stage = Path(temporary.name)
        _inspect_archive(Path(archive_path), manifest, stage)
        target.mkdir(exist_ok=False)
        try:
            for entry in manifest.files:
                relative = PurePosixPath(entry.path)
                output = target.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with (
                    stage.joinpath(*relative.parts).open("rb") as source,
                    output.open("xb") as sink,
                ):
                    shutil.copyfileobj(source, sink, CHUNK_SIZE)
                output.chmod(entry.mode)
        except BaseException as error:
            best_effort_cleanup(
                error, (("remove failed extraction", lambda: shutil.rmtree(target)),)
            )
            raise
    except BaseException as error:
        best_effort_cleanup(error, (("remove extraction staging", temporary.cleanup),))
        raise
    else:
        temporary.cleanup()
    return manifest
