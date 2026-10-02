"""Prepare verified source/fixture bundles locally. Never calls a cloud service."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rc_car.ops.bundles import create_bundle, verify_bundle
from rc_car.workflow_config import load_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory below outputs/")
    parser.add_argument(
        "--plan", action="store_true", help="Print local steps without writing/running"
    )
    args = parser.parse_args()
    if args.output.is_symlink():
        parser.error("Output must not be a symlink, including a dangling one")
    destination = args.output.resolve()
    if (
        not destination.is_relative_to(REPO_ROOT / "outputs")
        or destination == REPO_ROOT / "outputs"
    ):
        parser.error("Choose a new subdirectory of this checkout's outputs/")
    if destination.exists():
        parser.error("Output already exists; choose a new directory")
    if args.plan:
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "cloud_actions": [],
                    "steps": [
                        "CPU quality gate",
                        "validate every config.yaml profile",
                        "capture source identity",
                        "bundle source",
                        "bundle generated fixtures",
                        "verify bundles",
                        "write checksums",
                    ],
                    "unattended_gpu_allowed": False,
                },
                indent=2,
            )
        )
        return 0
    fixture = REPO_ROOT / "generated_assets/fixture.json"
    if (
        not fixture.is_file()
        or json.loads(fixture.read_text()).get("kind") != "procedural-smoke-test-only"
    ):
        parser.error("Generate and inspect the procedural fixture before preparation")
    workflow = load_workflow(REPO_ROOT / "config.yaml")
    destination.mkdir(parents=True, exist_ok=False)
    before = destination / "precheck-provenance.json"
    subprocess.run(
        [sys.executable, "scripts/capture_provenance.py", "--output", str(before)],
        cwd=REPO_ROOT,
        check=True,
    )
    with (destination / "local-checks.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "scripts/check.py"],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        parser.exit(
            result.returncode, f"Local checks failed; see {destination / 'local-checks.log'}\n"
        )
    provenance = destination / "source-provenance.json"
    subprocess.run(
        [sys.executable, "scripts/capture_provenance.py", "--output", str(provenance)],
        cwd=REPO_ROOT,
        check=True,
    )
    source = json.loads(provenance.read_text())
    if source["files_sha256"] != json.loads(before.read_text())["files_sha256"]:
        parser.exit(
            2,
            "Source changed while the local gate ran; freeze edits and prepare a fresh session.\n",
        )
    create_bundle(REPO_ROOT, sorted(source["files_sha256"]), destination / "source")
    assets: list[str] = []

    def fail_scan(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(
        REPO_ROOT / "generated_assets", followlinks=False, onerror=fail_scan
    ):
        parent = Path(directory)
        if parent.is_symlink() or any((parent / name).is_symlink() for name in directories):
            parser.exit(2, "Fixture directories cannot be symlinks\n")
        # The bundle verifier rejects nonregular files and symlink leaves.
        assets.extend(str((parent / name).relative_to(REPO_ROOT)) for name in filenames)
    assets.sort()
    create_bundle(REPO_ROOT, assets, destination / "fixtures")
    for kind in ("source", "fixtures"):
        checked = verify_bundle(
            destination / kind / "bundle.tar.gz", destination / kind / "manifest.json"
        )
        if (
            kind == "source"
            and {entry.path: entry.sha256 for entry in checked.files} != source["files_sha256"]
        ):
            parser.exit(
                2,
                "Source changed between provenance capture and bundling; prepare a fresh session.\n",
            )
    (destination / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prepared_at_utc": datetime.now(UTC).isoformat(),
                "branch": source["branch"],
                "git_commit": source["git_commit"],
                "local_checks_passed": True,
                "cloud_created": False,
                "tracking_mode": workflow.tracking.mode,
                "max_concurrent_instances": 1,
                "unattended_gpu_allowed": False,
                "billing_safety_note": "No independently verified server-side expiry is configured",
                "workflow": {
                    "config": "config.yaml",
                    "operation": workflow.operation,
                    "selected_profile": workflow.profile,
                    "profiles": sorted(workflow.profiles),
                },
            },
            indent=2,
        )
        + "\n"
    )
    names = [
        "source/bundle.tar.gz",
        "source/manifest.json",
        "fixtures/bundle.tar.gz",
        "fixtures/manifest.json",
        "source-provenance.json",
        "session.json",
        "local-checks.log",
        "precheck-provenance.json",
    ]
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((destination / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        )
    )
    print(f"Local session prepared: {destination}\nNo cloud instance was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
