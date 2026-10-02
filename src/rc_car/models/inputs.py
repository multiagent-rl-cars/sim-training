"""Explicit preflight checks for the inherited feature extractor's input contract."""

from collections.abc import Mapping

import torch
from gymnasium import spaces


def validate_feature_extractor_inputs(
    observations: Mapping[str, torch.Tensor], observation_space: spaces.Dict
) -> None:
    """Check normalized float32 batches without modifying or rescaling them.

    Both image tensors must be channel-first with the same spatial dimensions,
    on the same device as the velocity tensor, and bounded by [0, 1]. Velocity
    is finite but intentionally not clipped to [-1, 1]: the legacy producer
    divides by 0.8 m/s without clipping. Declared per-key shapes must match.

    This diagnostic synchronizes tensor values with the CPU; use it in preflight
    and analysis, not unconditionally on every GPU training step. It raises
    ValueError for contract violations and has no effect on model state.
    """
    keys = ("track_image", "obstacle_image", "robot_vel")
    batch_size: int | None = None
    device: torch.device | None = None
    image_dimensions: tuple[int, ...] | None = None
    for key in keys:
        if key not in observations:
            raise ValueError(f"Missing observation: {key}")
        space = observation_space.spaces.get(key)
        if not isinstance(space, spaces.Box):
            raise ValueError(f"{key} must declare a Box observation space")
        expected_rank = 1 if key == "robot_vel" else 3
        if len(space.shape) != expected_rank or any(size <= 0 for size in space.shape):
            raise ValueError(f"Invalid declared shape for {key}: {space.shape}")
        tensor = observations[key]
        if tensor.ndim != expected_rank + 1 or tuple(tensor.shape[1:]) != space.shape:
            raise ValueError(
                f"{key} must have batched shape (B, {space.shape}), got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] == 0:
            raise ValueError(f"{key} must contain at least one observation")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must have dtype float32, got {tensor.dtype}")
        if batch_size is None:
            batch_size, device = tensor.shape[0], tensor.device
        if tensor.shape[0] != batch_size or tensor.device != device:
            raise ValueError(f"{key} must share the batch size and device of track_image")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{key} must contain only finite values")
        if key != "robot_vel":
            if image_dimensions is None:
                image_dimensions = tuple(tensor.shape[2:])
            if tuple(tensor.shape[2:]) != image_dimensions:
                raise ValueError("track_image and obstacle_image must share spatial dimensions")
            if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
                raise ValueError(f"{key} must be normalized to [0, 1]")
