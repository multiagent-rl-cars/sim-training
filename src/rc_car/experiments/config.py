"""Validated recipes independent of the simulator and tracking SDK.

These schemas describe the existing fixture runner, not every SB3 option.
Validation establishes configuration consistency, not correct task semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]
Seed = Annotated[int, Field(ge=0, lt=2**32)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonnegativeFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
NonemptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
TrackingMode = Literal["online", "offline"]


class RecipeModel(BaseModel):
    """Reject coercion, unknown keys and non-finite numbers at input boundaries."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
        revalidate_instances="always",
    )


class PPOConfig(RecipeModel):
    """Supported numeric PPO parameters; defaults match SB3 for this subset."""

    n_steps: PositiveInt = 2048
    batch_size: Annotated[int, Field(gt=1)] = 64
    n_epochs: PositiveInt = 10
    learning_rate: PositiveFloat = 0.0003
    gamma: Probability = 0.99
    gae_lambda: Probability = 0.95
    clip_range: Annotated[float, Field(gt=0, le=1)] = 0.2
    ent_coef: NonnegativeFloat = 0.0
    vf_coef: NonnegativeFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1

    @field_validator("sde_sample_freq")
    @classmethod
    def valid_sde_frequency(cls, value: int) -> int:
        if value != -1 and value <= 0:
            raise ValueError("sde_sample_freq must be -1 or a positive integer")
        return value


class WandbConfig(RecipeModel):
    """Public destination only; credentials come from the SDK environment."""

    entity: NonemptyString
    project: NonemptyString
    mode: TrackingMode = "online"


class ExperimentConfig(RecipeModel):
    """Version-one recipe for local checks and the historical GPU runner.

    Missing time budgets preserve timestep-only behavior. A present budget bounds
    training only, not startup, evaluation, or GPU billing.
    """

    schema_version: Literal[1] = 1
    name: NonemptyString | None = None
    task: NonemptyString
    num_envs: PositiveInt
    enable_obstacles: bool = True
    train_seed: Seed
    eval_seed: Seed
    total_timesteps: PositiveInt
    max_training_seconds: PositiveFloat | None = None
    training_video_interval_s: PositiveFloat | None = None
    training_video_frames: PositiveInt = 90
    eval_steps: PositiveInt
    video_stride: PositiveInt
    checkpoint_interval: PositiveInt
    torch_threads: PositiveInt
    wandb: WandbConfig | None = None
    known_limitations: tuple[str, ...] | None = None
    ppo: PPOConfig = Field(default_factory=PPOConfig)

    # Existing CPU tracking checks add these before starting the run.
    test_only: bool | None = None
    media_provenance: NonemptyString | None = None
    note: NonemptyString | None = None
    synthetic_media: bool | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value: object) -> object:
        # Literal[1] alone accepts True and 1.0 in Pydantic.
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @field_validator("known_limitations", mode="before")
    @classmethod
    def immutable_limitations(cls, value: object) -> object:
        # JSON arrays arrive as lists; store validated contents immutably.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        for key in ("max_training_seconds", "training_video_interval_s"):
            if key in self.model_fields_set and getattr(self, key) is None:
                raise ValueError(f"{key} must be omitted or positive, not null")
        rollout_size = self.num_envs * self.ppo.n_steps
        if rollout_size <= 1:
            raise ValueError("PPO rollout must contain more than one sample")
        if self.ppo.batch_size > rollout_size:
            raise ValueError("PPO batch exceeds rollout size")
        return self


ConfigInput = ExperimentConfig | Mapping[str, object]


def validate_config(config: ConfigInput) -> ExperimentConfig:
    """Return a validated, frozen recipe; legacy callers may ignore the result."""
    return ExperimentConfig.model_validate(config)


def resolve_config(config: ConfigInput) -> dict[str, Any]:
    """Expand defaults to JSON-compatible values without mutating input.

    Optional absent fields stay absent, preserving the old runner's ``dict.get``
    defaults. Credentials are not part of the schema.
    """
    return validate_config(config).model_dump(mode="json", exclude_none=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects without including their values in errors."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate experiment configuration key: {key!r}")
        result[key] = value
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """Read an unambiguous JSON recipe without loading Isaac, Torch, or W&B."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    return validate_config(data)
