"""Optional W&B integration and a local public run identity record.

Importing this module does not load W&B or access the network. Only an explicit
``start_tracking`` call starts an SDK run; offline mode remains available.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

from .config import ConfigInput, TrackingMode, resolve_config, validate_config


class TrackingRun(Protocol):
    """The SDK surface used here, also implementable by offline test fakes."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str | None: ...

    @property
    def url(self) -> str | None: ...

    @property
    def entity(self) -> str: ...

    @property
    def project(self) -> str: ...

    def define_metric(self, name: str, *, step_metric: str) -> object: ...

    def log(self, data: dict[str, Any]) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...


class TrackingSDK(Protocol):
    def init(self, **kwargs: Any) -> TrackingRun | None: ...

    def Video(self, path: str, *, format: str) -> object: ...  # noqa: N802


def _load_wandb() -> TrackingSDK:
    try:
        return cast(TrackingSDK, importlib.import_module("wandb"))
    except ModuleNotFoundError as error:
        if error.name != "wandb":
            raise
        raise RuntimeError(
            "W&B tracking is optional. Install the tracking dependency group to use it."
        ) from error


def _tracking_mode(config_mode: TrackingMode) -> TrackingMode:
    mode = os.environ.get("WANDB_MODE", config_mode)
    if mode not in ("online", "offline"):
        raise ValueError("Tracking mode must be online or offline")
    return mode


def _persist_identity(path: Path, identity: dict[str, str | None]) -> None:
    """Publish a complete identity atomically without replacing an existing one.

    Creating a hard link commits the prepared file with exclusive-create semantics
    on the supported Mac/Linux filesystems. A competing writer cannot be clobbered.
    Failure before publication leaves no misleading partial identity behind.
    """
    payload = json.dumps(identity, indent=2, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".wandb-identity-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def start_tracking(
    config: ConfigInput,
    run_dir: str | Path,
    *,
    job_type: str = "training",
) -> TrackingRun:
    """Start a run explicitly and save its identity beside local outputs.

    ``WANDB_MODE`` overrides the recipe mode. Existing identity files are never
    overwritten: resuming requires a separate deliberate workflow. If identity
    persistence fails, close the new SDK run.
    """
    recipe = validate_config(config)
    if recipe.wandb is None:
        raise ValueError("Tracking requires a wandb destination in the recipe")
    if not job_type.strip():
        raise ValueError("job_type must not be empty")
    for setting in (
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RESUME_FROM",
        "WANDB_FORK_FROM",
        "WANDB_NAME",
    ):
        if os.environ.get(setting):
            raise ValueError(f"{setting} is incompatible with starting a fresh tracked run")
    mode = _tracking_mode(recipe.wandb.mode)
    directory = Path(run_dir)
    identity_path = directory / "wandb_run.json"
    if identity_path.exists() or identity_path.is_symlink():
        raise FileExistsError(f"Tracking identity already exists: {identity_path}")
    sdk = _load_wandb()
    directory.mkdir(parents=True, exist_ok=True)
    resolved = resolve_config(recipe)
    resolved["wandb"]["mode"] = mode
    run = sdk.init(
        entity=recipe.wandb.entity,
        project=recipe.wandb.project,
        config=resolved,
        dir=str(directory),
        mode=mode,
        job_type=job_type,
        save_code=False,
        reinit="create_new",
        id=uuid.uuid4().hex,
        resume="never",
    )
    if run is None:
        raise RuntimeError("W&B did not return a run")
    try:
        run.define_metric("training/*", step_metric="training/agent_timesteps")
        identity = {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "entity": run.entity,
            "project": run.project,
            "mode": mode,
        }
        for key in ("id", "entity", "project"):
            if not isinstance(identity[key], str) or not identity[key]:
                raise RuntimeError(f"W&B returned an invalid {key}")
        if identity["url"] is not None and not isinstance(identity["url"], str):
            raise RuntimeError("W&B returned an invalid url")
        display_name = identity["name"]
        if display_name is None:
            if mode == "online":
                raise RuntimeError("W&B did not return an online display name")
        elif not isinstance(display_name, str) or not display_name.strip():
            raise RuntimeError("W&B returned an invalid display name")
        _persist_identity(identity_path, identity)
    except BaseException as error:
        try:
            run.finish(exit_code=1)
        except BaseException:
            error.add_note("Could not close the tracking run after setup failed")
        raise
    return run


def log_video(
    run: TrackingRun | None,
    path: str | Path,
    *,
    key: str = "training/video",
    timesteps: int = 0,
) -> None:
    """Log a nonempty MP4; encoding/decoding is the recorder's concern."""
    video_path = Path(path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError(f"Missing or empty video: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("Video logging requires an MP4 file")
    if run is None:
        raise ValueError("Video logging requires an active tracking run")
    if type(timesteps) is not int or timesteps < 0:
        raise ValueError("timesteps must be a nonnegative integer")
    if not key.strip():
        raise ValueError("Video key must not be empty")
    sdk = _load_wandb()
    run.log(
        {
            key: sdk.Video(str(video_path), format="mp4"),
            "training/agent_timesteps": timesteps,
        }
    )
