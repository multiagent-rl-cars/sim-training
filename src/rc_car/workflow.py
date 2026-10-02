"""One no-argument front door: edit config.yaml, then uv run rc-car."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .experiments.config import resolve_config
from .workflow_config import WorkflowConfig, load_workflow, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def plan(config: WorkflowConfig, root: Path) -> dict[str, Any]:
    output = resolve_path(root, config.job.output) if config.job.output is not None else None
    fixture = resolve_path(root, config.job.fixture_manifest)
    source = resolve_path(root, config.job.input_run) if config.job.input_run else None
    if (
        source is not None
        and output is not None
        and (output == source or output.is_relative_to(source))
    ):
        raise ValueError("job.output must be outside the saved input run")
    return {
        "operation": config.operation,
        "profile": config.profile,
        "output": str(output) if output is not None else None,
        "output_policy": "explicit"
        if output is not None
        else "outputs/runs/<uuid>, allocated only when writing",
        "input_run": str(source) if source else None,
        "fixture_manifest": str(fixture),
        "timeout_seconds": config.job.timeout_seconds,
        "effective_recipe": resolve_config(config.experiment()),
        "cloud_allocation": False,
    }


def _create_output(config: WorkflowConfig, root: Path, source: Path | None = None) -> Path:
    """Reserve a unique local directory atomically, independently of W&B names."""
    for _ in range(10):
        relative = config.job.output or f"outputs/runs/{uuid.uuid4().hex}"
        output = resolve_path(root, relative)
        if source is not None and (output == source or output.is_relative_to(source)):
            raise ValueError("job.output must be outside the saved input run")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.mkdir()
        except FileExistsError:
            if config.job.output is not None:
                raise ValueError(
                    "Choose a fresh job.output; existing results are never overwritten"
                ) from None
            continue
        record = {
            "schema_version": 1,
            "requested_config": config.model_dump(mode="json"),
            "resolved_output": str(output),
            "output_kind": "explicit" if config.job.output is not None else "automatic",
        }
        (output / "workflow-config.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        print(f"Local output: {output}", flush=True)
        return output
    raise RuntimeError("Could not reserve a fresh automatic output directory")


def execute(config: WorkflowConfig, root: Path) -> int:
    """Run the selected operation. Local defaults never authenticate or use a GPU."""
    root = root.resolve()
    resolved = plan(config, root)
    if config.operation == "plan":
        print(json.dumps(resolved, indent=2, allow_nan=False))
        return 0
    if config.operation == "check":
        return subprocess.run(
            [sys.executable, "scripts/check.py"], cwd=root, check=False
        ).returncode

    # Only the YAML selects tracking mode. Credentials remain external and are
    # consumed by the SDK/preflight; never copied into the resolved documents.
    previous_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = config.tracking.mode
    try:
        if config.operation == "tracking-check":
            from .experiments.online_check import check_online

            print(json.dumps(check_online(config.experiment()), indent=2))
            return 0

        if config.operation == "train" and sys.platform != "linux":
            raise ValueError(
                "Training requires an already prepared Linux RTX host; use operation: plan locally"
            )

        if config.operation == "inspect":
            from .ops.evidence import inspect_run

            source = Path(resolved["input_run"])
            report = inspect_run(source)
            output = _create_output(config, root, source)
            (output / "evidence.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["integration_passed"] else 1

        if config.operation == "upload":
            from .experiments.upload import upload_completed_experiment, validate_upload

            source = Path(resolved["input_run"])
            identity = validate_upload(source)["identity"]
            if (identity["entity"], identity["project"]) != (
                config.tracking.entity,
                config.tracking.project,
            ):
                raise ValueError(
                    "Saved run and YAML tracking destination differ; refusing to relabel results"
                )
            output = _create_output(config, root, source)
            receipt = upload_completed_experiment(
                source, work_dir=output / "upload", receipt_path=output / "upload-receipt.json"
            )
            print(json.dumps(receipt, indent=2, allow_nan=False))
            return 0

        output = _create_output(config, root)
        recipe = output / "requested-recipe.json"
        recipe.write_text(json.dumps(config.requested_recipe(), indent=2, allow_nan=False) + "\n")
        # The established supervisor retains locking, source/fixture preflight,
        # W&B isolation, and process-group timeouts. Flags are internal plumbing.
        command = [
            "bash",
            "scripts/run_gpu_checks.sh",
            "--mode",
            "experiment",
            "--config",
            str(recipe),
            "--output",
            str(output / "gpu"),
            "--fixture-manifest",
            str(resolved["fixture_manifest"]),
            "--tracking-mode",
            config.tracking.mode,
            "--timeout-seconds",
            str(config.job.timeout_seconds),
        ]
        return subprocess.run(command, cwd=root, check=False).returncode
    finally:
        if previous_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = previous_mode


def main() -> int:
    if len(sys.argv) != 1:
        print(
            "This command takes no arguments. Edit config.yaml, then uv run rc-car.",
            file=sys.stderr,
        )
        return 2
    try:
        return execute(load_workflow(REPO_ROOT / "config.yaml"), REPO_ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        # Schema/transport errors can contain arbitrary user data. Keep output
        # public; detailed validation is available to tests and library callers.
        print(
            f"Workflow failed ({type(error).__name__}); check config.yaml and the operation's local logs.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
