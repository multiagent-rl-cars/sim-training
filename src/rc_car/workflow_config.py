"""Strict, simulator-free configuration for the single supported workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from .experiments.config import ExperimentConfig, RecipeModel, WandbConfig, validate_config

MAX_CONFIG_BYTES = 128 * 1024


class _StrictLoader(yaml.SafeLoader):
    """No aliases, merges, nonstring keys or ambiguous duplicate mappings."""

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        if self.check_event(yaml.AliasEvent) or getattr(self.peek_event(), "anchor", None):
            raise ValueError(
                "YAML anchors and aliases are not supported; keep profiles self-contained"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Configuration mapping keys must be strings")
            if key in result:
                raise ValueError("Duplicate YAML configuration key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _without_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if any(
                word in normalized
                for word in ("api_key", "password", "secret", "token", "credential", "private_key")
            ):
                raise ValueError(
                    "Credentials must not appear in config.yaml; use host authentication"
                )
            _without_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _without_secrets(item)
    elif isinstance(value, str) and (
        "wandb_v1_" in value or "WANDB_API_KEY=" in value or "PRIVATE KEY-----" in value
    ):
        raise ValueError("Credential-like values must not appear in config.yaml")


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
        or "\\" in value
        or path.parts[0] != "outputs"
        or len(path.parts) < 2
    ):
        raise ValueError("Workflow paths must be canonical paths below outputs/ in this repository")
    return value


class JobConfig(RecipeModel):
    output: str | None = None
    fixture_manifest: str
    timeout_seconds: Annotated[int, Field(gt=0, le=36000)] = 2400
    input_run: str | None = None

    @field_validator("output", "fixture_manifest", "input_run")
    @classmethod
    def confined_path(cls, value: str | None) -> str | None:
        return relative_path(value) if value is not None else None


class WorkflowConfig(RecipeModel):
    schema_version: Literal[1]
    operation: Literal["check", "plan", "tracking-check", "train", "inspect", "upload"]
    profile: str
    job: JobConfig
    tracking: WandbConfig
    profiles: dict[str, dict[str, Any]]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not self.profiles or self.profile not in self.profiles:
            raise ValueError("Select an existing experiment profile")
        for name in self.profiles:
            if not name or not all(character.isalnum() or character in "-_" for character in name):
                raise ValueError(
                    "Profile names must contain only letters, numbers, dash or underscore"
                )
            self.experiment(name)
        if self.operation in {"inspect", "upload"} and self.job.input_run is None:
            raise ValueError("inspect/upload require job.input_run")
        if self.operation == "upload" and self.tracking.mode != "online":
            raise ValueError("upload requires tracking.mode: online")
        return self

    def requested_recipe(self, profile: str | None = None) -> dict[str, Any]:
        """One self-contained profile, plus schema and shared tracking identity."""
        override = self.profiles[self.profile if profile is None else profile]
        if any(key in override for key in ("wandb", "schema_version", "name", "known_limitations")):
            raise ValueError("Profiles contain execution settings only; schema/tracking are shared")
        merged = deepcopy(override)
        merged["schema_version"] = self.schema_version
        merged["wandb"] = self.tracking.model_dump(mode="json")
        return merged

    def experiment(self, profile: str | None = None) -> ExperimentConfig:
        return validate_config(self.requested_recipe(profile))


def load_workflow(path: str | Path) -> WorkflowConfig:
    """Load one bounded YAML document; no interpolation, imports or environment reads."""
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("config.yaml must be a bounded regular file, not a link")
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except (yaml.YAMLError, RecursionError) as error:
        raise ValueError(
            "Invalid config.yaml; use plain YAML mappings without custom tags"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must contain one mapping")
    _without_secrets(payload)
    return WorkflowConfig.model_validate(payload)


def resolve_path(root: Path, relative: str) -> Path:
    """Reject aliases/symlink parents as well as traversal before any operation."""
    relative_path(relative)
    base = root.resolve()
    path = base / relative
    if path.resolve() != path or not path.is_relative_to(base):
        raise ValueError("Workflow paths cannot traverse symlinks or escape the repository")
    return path
