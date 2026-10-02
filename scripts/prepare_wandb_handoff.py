"""Print an offline-only upload plan for one verified copied experiment.

No SDK import, authentication, network call, stream rewrite, or upload occurs.
Credential-like filenames are rejected; this is not secret-content scanning.
Output is JSON on stdout; save it outside the immutable backup if desired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rc_car.ops.wandb_handoff import prepare_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-directory", required=True, help="Relative to backup root")
    args = parser.parse_args()
    report = prepare_handoff(args.backup_root, args.manifest, run_directory=args.run_directory)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
