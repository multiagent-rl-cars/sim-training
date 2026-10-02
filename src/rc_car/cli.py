"""Local inspection commands. None of these commands launch Isaac or training."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rc_car.experiments.config import load_config, resolve_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "Validate a recipe without a simulator or network access"),
        ("show-config", "Print the validated recipe with all effective recipe defaults"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Invalid experiment configuration: {error}\n")
    if args.command == "show-config":
        print(json.dumps(resolve_config(config), indent=2, allow_nan=False))
    else:
        print(f"Valid: {config.name or config.task} ({config.task}; {config.num_envs} slots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
