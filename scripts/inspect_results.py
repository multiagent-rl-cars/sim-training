"""Inspect local GPU evidence; never starts Isaac or loads SB3 cloudpickle metadata."""

import argparse
import json
from pathlib import Path

from rc_car.ops.evidence import inspect_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="The smoke/ or experiment/ leaf"
    )
    parser.add_argument(
        "--report", type=Path, help="New report file; existing reports are not overwritten"
    )
    args = parser.parse_args()
    report = inspect_run(args.run_dir)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["integration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
