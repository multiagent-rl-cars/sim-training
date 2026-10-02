"""Noninteractive, read-only W&B authentication check before starting Isaac.

Only the public W&B service is supported here. Credentials are read from the
standard environment or netrc, never serialized or included in error messages.
Project readability is not proof of permission to create a run or upload files.
"""

from __future__ import annotations

import argparse
import json
import netrc
import os
import sys
from pathlib import Path
from typing import Any

from .config import ConfigInput, load_config, validate_config
from .tracking import _tracking_mode

API_BASE = "https://api.wandb.ai"


def configured_key() -> str:
    """Return a credential only to the authenticated transport, never to output."""
    if os.environ.get("WANDB_BASE_URL", API_BASE).rstrip("/") != API_BASE:
        raise RuntimeError("Online checks support only https://api.wandb.ai")
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    path = Path(os.environ.get("NETRC", str(Path.home() / ".netrc")))
    try:
        auth = netrc.netrc(str(path)).authenticators("api.wandb.ai")
    except FileNotFoundError:
        auth = None
    except (OSError, netrc.NetrcParseError):
        raise RuntimeError("Cannot read W&B netrc credentials; run wandb login") from None
    if auth and auth[2]:
        return auth[2]
    raise RuntimeError("W&B credentials missing; run wandb login or set WANDB_API_KEY securely")


def check_online(config: ConfigInput) -> dict[str, Any]:
    """Verify identity and destination without creating a run or invoking Kit."""
    recipe = validate_config(config)
    if recipe.wandb is None:
        return {"status": "tracking-not-configured", "online_verified": False}
    if _tracking_mode(recipe.wandb.mode) == "offline":
        return {"status": "explicit-offline", "online_verified": False}
    key = configured_key()
    import requests

    query = """query($entity: String!, $project: String!) {
      viewer { id }
      project(name: $project, entityName: $entity) { name }
    }"""
    try:
        response = requests.post(
            API_BASE + "/graphql",
            auth=("api", key),
            json={
                "query": query,
                "variables": {"entity": recipe.wandb.entity, "project": recipe.wandb.project},
            },
            timeout=(5, 15),
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError("W&B authentication/project check failed; verify login and access")
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError(
            "W&B is unreachable or returned invalid data; retry before training"
        ) from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, dict)
        or payload.get("errors")
        or not isinstance(data.get("viewer"), dict)
        or not data["viewer"].get("id")
        or not isinstance(data.get("project"), dict)
        or data["project"].get("name") != recipe.wandb.project
    ):
        raise RuntimeError("W&B destination is not readable; verify login and project access")
    return {
        "status": "authenticated-project-readable",
        "online_verified": True,
        "entity": recipe.wandb.entity,
        "project": recipe.wandb.project,
        "project_url": f"https://wandb.ai/{recipe.wandb.entity}/{recipe.wandb.project}",
        "run_write_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = check_online(load_config(args.config))
    except (OSError, ValueError, RuntimeError) as error:
        # All transport errors above have fixed public text, never raw SDK/HTTP
        # exception strings (which can contain request data or credentials).
        print(f"W&B preflight failed: {error}", file=sys.stderr)
        print("For an intentional offline run, explicitly set WANDB_MODE=offline.", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
