"""Pack or verify local artifacts. No cloud access, deletion, or simulator startup."""

import argparse
import json
import os
from pathlib import Path

from rc_car.ops.bundles import create_bundle, extract_bundle, verify_bundle, verify_directory

REPO_ROOT = Path(__file__).resolve().parents[1]


def regular_files(root: Path, *, skip_links: bool) -> tuple[list[str], list[str]]:
    """Select regular artifacts under an explicit root without following links."""
    names: list[str] = []
    skipped: list[str] = []

    def fail(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=fail):
        parent = Path(directory)
        for name in list(directories):
            path = parent / name
            if path.is_symlink():
                relative = str(path.relative_to(root))
                if not skip_links:
                    raise ValueError(f"Symlink in fixture tree: {relative}")
                skipped.append(relative)
                directories.remove(name)
        for name in files:
            path = parent / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                if not skip_links:
                    raise ValueError(f"Symlink in fixture tree: {relative}")
                skipped.append(relative)
            elif path.is_file():
                names.append(relative)
            else:
                raise ValueError(f"Nonregular artifact: {relative}")
    return sorted(names), sorted(skipped)


def verify_fixtures(root: Path, manifest_path: Path) -> None:
    root = root.resolve(strict=True)
    manifest = verify_directory(root, manifest_path)
    assets = root / "generated_assets"
    if assets.is_symlink() or not assets.is_dir():
        raise ValueError("Generated fixture directory is missing or symlinked")
    names, _ = regular_files(assets, skip_links=False)
    actual = {f"generated_assets/{name}" for name in names}
    listed = {entry.path for entry in manifest.files}
    if listed != actual or "generated_assets/fixture.json" not in listed:
        raise ValueError("Fixture manifest must cover exactly the current generated_assets files")
    if not any(name.endswith((".usd", ".usda", ".usdc")) for name in listed):
        raise ValueError("Fixture bundle has no USD geometry")
    if (
        json.loads((assets / "fixture.json").read_text()).get("kind")
        != "procedural-smoke-test-only"
    ):
        raise ValueError("Only explicitly marked procedural smoke fixtures are supported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "extract"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "extract":
            command.add_argument("--destination", type=Path, required=True)
    command = commands.add_parser("verify-tree")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--require-fixtures", action="store_true")
    command = commands.add_parser("pack-results")
    command.add_argument("--root", type=Path, required=True, help="One explicit job below outputs/")
    command.add_argument(
        "--output", type=Path, required=True, help="Fresh bundle directory, outside job"
    )
    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify_bundle(args.archive, args.manifest)
        elif args.command == "extract":
            extract_bundle(args.archive, args.manifest, args.destination)
        elif args.command == "verify-tree":
            if args.require_fixtures:
                verify_fixtures(args.root, args.manifest)
            else:
                verify_directory(args.root, args.manifest)
        else:
            root = args.root.resolve(strict=True)
            output = args.output.resolve()
            if (
                args.root.is_symlink()
                or not root.is_relative_to(REPO_ROOT / "outputs")
                or root == REPO_ROOT / "outputs"
            ):
                raise ValueError(
                    "Choose one non-symlink job directory below this checkout's outputs/"
                )
            if output.is_relative_to(root):
                raise ValueError("Backup destination must be outside the job being archived")
            names, links = regular_files(root, skip_links=True)
            forbidden = {".env", ".netrc", "credentials", "credentials.json"}
            if any(
                Path(name).name in forbidden or Path(name).suffix in {".key", ".pem"}
                for name in names
            ):
                raise ValueError("Credential-like file found; inspect the job before archiving")
            paths = create_bundle(root, names, output)
            verify_bundle(paths.archive, paths.manifest)
            (output / "backup-context.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_directory": str(root),
                        "excluded_symlinks": links,
                        "regular_files_verified": len(names),
                        "training_success_required": False,
                        "limitations": [
                            "Checksums are not sender authentication",
                            "Cross-machine W&B sync is unverified",
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
        print(f"Completed local bundle operation: {args.command}")
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f"Bundle operation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
