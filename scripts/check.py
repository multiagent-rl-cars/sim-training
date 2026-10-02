"""Run the same offline CPU quality gates used by CI, from any working directory."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = (
        ("ruff", "check", "."),
        ("ruff", "format", "--check", "."),
        ("ty", "check"),
        ("pytest", "-m", "not gpu and not network"),
    )
    for command in commands:
        print(f"\nChecking: {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, "-m", *command], cwd=root, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
