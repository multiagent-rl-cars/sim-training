"""Attach completed outputs to their existing online W&B run, then verify them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rc_car.experiments.upload import upload_completed_experiment, validate_upload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--work-dir", type=Path, help="New directory outside the source run")
    parser.add_argument("--receipt", type=Path, help="New receipt outside the source run")
    parser.add_argument("--validate-only", action="store_true", help="No SDK import or network")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_upload(args.run_dir)
        print(json.dumps({"identity": result["identity"], "validated_files": len(result["files"])}))
    else:
        result = upload_completed_experiment(
            args.run_dir, work_dir=args.work_dir, receipt_path=args.receipt
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
