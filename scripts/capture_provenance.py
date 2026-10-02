"""Capture an explicit source selection, failing on unreadable or linked entries.

This is a text-source snapshot, not a whole checkout/runtime/assets backup or a
secret scanner. Optional missing source trees are recorded; existing source
trees must be enumerable in full. Generated output/cache trees are not visited.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path

SOURCE_ROOTS = (
    "src",
    "scripts",
    "source/car_training",
    "docs",
    "tests",
    ".github",
    "experiments",
)
ROOT_FILES = (
    "AGENTS.md",
    "README.md",
    "config.yaml",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".pre-commit-config.yaml",
    ".gitignore",
)
SOURCE_SUFFIXES = frozenset(
    (
        ".py",
        ".sh",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".rst",
        ".html",
        ".typed",
    )
)
PRUNED_DIRECTORIES = frozenset(
    (
        "__pycache__",
        ".git",
        ".venv",
        "outputs",
        "scratch",
        "wandb",
        "generated_assets",
    )
)
CREDENTIAL_NAMES = frozenset(("credentials", "secrets", "credential", "secret", "api_key"))


def _pruned(name: str) -> bool:
    return name in PRUNED_DIRECTORIES or name.endswith(".egg-info")


def _mode(path: Path) -> int:
    # lstat must propagate errors; Path.is_file()/exists() can hide some OS errors.
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"Source selection rejects symlinks: {path}")
    return mode


def _fail_scan(error: OSError) -> None:
    raise error


def _parents(root: Path, path: Path) -> None:
    parent = root
    for name in path.relative_to(root).parts[:-1]:
        parent /= name
        if not stat.S_ISDIR(_mode(parent)):
            raise ValueError(f"Expected a source directory: {parent}")


def _credential_like(path: Path) -> bool:
    name = path.name.lower()
    return (
        name == ".netrc"
        or name == ".env"
        or name.startswith(".env.")
        or path.stem.lower() in CREDENTIAL_NAMES
        or path.suffix.lower() in {".key", ".pem"}
    )


def select_source_files(root: Path) -> tuple[list[Path], list[str]]:
    """Return sorted root-relative source paths and explicitly absent optional trees.

    Known output/cache directories are pruned before traversal. All other entries
    in existing source trees must be stat-able and ordinary files/directories,
    even when their suffix is excluded. Never follow a symlink to read its target.
    """
    root = root.resolve(strict=True)
    selected: set[Path] = set()
    absent: list[str] = []
    for relative in SOURCE_ROOTS:
        directory = root / relative
        try:
            _parents(root, directory)
            mode = _mode(directory)
        except FileNotFoundError:
            absent.append(relative)
            continue
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Expected a source directory: {relative}")
        for current, directories, filenames in os.walk(
            directory, followlinks=False, onerror=_fail_scan
        ):
            parent = Path(current)
            if not stat.S_ISDIR(_mode(parent)):
                raise ValueError(f"Expected a source directory: {parent}")
            directories[:] = sorted(name for name in directories if not _pruned(name))
            for name in directories:
                if _credential_like(Path(name)) or name in {".ssh", ".aws", ".azure"}:
                    raise ValueError(f"Credential-like directory in source tree: {parent / name}")
                if not stat.S_ISDIR(_mode(parent / name)):
                    raise ValueError(f"Expected a source directory: {parent / name}")
            for name in sorted(filenames):
                if _pruned(name):
                    continue
                path = parent / name
                if not stat.S_ISREG(_mode(path)):
                    raise ValueError(f"Expected a regular source file: {path}")
                if _credential_like(path):
                    raise ValueError(
                        f"Credential-like file in source tree; inspect before capture: {path}"
                    )
                if path.suffix in SOURCE_SUFFIXES or path.name in {"LICENSE", "NOTICE"}:
                    selected.add(path.relative_to(root))
    for relative in ROOT_FILES:
        _parents(root, root / relative)
        if not stat.S_ISREG(_mode(root / relative)):
            raise ValueError(f"Expected required regular source file: {relative}")
        selected.add(Path(relative))
    return sorted(selected), absent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/source-provenance.json"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    paths, absent = select_source_files(root)
    if args.output.is_symlink() or (args.archive is not None and args.archive.is_symlink()):
        parser.error("Capture outputs must not be symlinks")
    if args.output.resolve() in {root / path for path in paths}:
        parser.error("Provenance output must not replace a selected source file")
    if args.archive and (
        args.archive.resolve() == args.output.resolve()
        or args.archive.resolve() in {root / path for path in paths}
    ):
        parser.error("Archive must not replace source files or the provenance output")

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], text=True, cwd=root, timeout=10)

    hashes = {}
    for path in paths:
        _parents(root, root / path)
        if not stat.S_ISREG(_mode(root / path)):
            raise ValueError(f"Source file changed type during capture: {path}")
        hashes[path.as_posix()] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    manifest = {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "branch": git("branch", "--show-current").strip(),
        "working_tree_status": git("status", "--short"),
        "files_sha256": hashes,
        "source_selection": {
            "roots": list(SOURCE_ROOTS),
            "absent_optional_roots": absent,
            "required_files": list(ROOT_FILES),
            "suffixes": sorted(SOURCE_SUFFIXES),
            "pruned_directories": sorted(PRUNED_DIRECTORIES),
            "pruned_suffix": ".egg-info",
            "scope": "Listed text source only; excludes assets/runtime/outputs; not a secret scan",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            for path in paths:
                _parents(root, root / path)
                if not stat.S_ISREG(_mode(root / path)):
                    raise ValueError(f"Source file changed type during archiving: {path}")
                archive.add(root / path, arcname=path.as_posix(), recursive=False)
            archive.add(args.output, arcname="source-provenance.json", recursive=False)
    print(args.output)


if __name__ == "__main__":
    main()
