"""Unweighted legacy reward math; task adapters own state reads and dt weighting.

These functions retain pixel-distance normalization, obstacle gating and the
absence of clipping. They do not claim centerline progress or smooth steering.
"""

import torch

from rc_car.core._validation import finite_tensor, floating_tensor, positive_scalar


def _check_batch(values: torch.Tensor, obstacle_present: torch.Tensor) -> None:
    floating_tensor(values, "values")
    if values.ndim != 1:
        raise ValueError("reward values must have shape (N,)")
    if obstacle_present.shape != values.shape or obstacle_present.dtype != torch.bool:
        raise ValueError("obstacle_present must be boolean with shape (N,)")
    if obstacle_present.device != values.device:
        raise ValueError("reward values and obstacle_present must share device")


def legacy_line_proximity(
    distance_px: torch.Tensor,
    obstacle_present: torch.Tensor,
    *,
    distance_scale_px: float = 40.0,
    validate_values: bool = False,
) -> torch.Tensor:
    """Return 1 - distance/scale, or zero when the obstacle gate is enabled.

    Distance is to the extracted legacy contour, not guaranteed to a centerline.
    Beyond scale, this term is negative. No manager weight or time factor is applied.
    """
    _check_batch(distance_px, obstacle_present)
    positive_scalar(distance_scale_px, "distance_scale_px")
    if validate_values:
        finite_tensor(distance_px, "distance_px")
        if bool((distance_px < 0).any()):
            raise ValueError("distance_px must be nonnegative")
    reward = 1.0 - distance_px / distance_scale_px
    return torch.where(obstacle_present, torch.zeros_like(reward), reward)


def legacy_steering_penalty(
    steering_rad: torch.Tensor,
    obstacle_present: torch.Tensor,
    *,
    max_steering_rad: float,
    validate_values: bool = False,
) -> torch.Tensor:
    """Return -abs(steer)/scale, gated to zero near an obstacle.

    This penalizes steering magnitude, not steering rate or jerk.
    """
    _check_batch(steering_rad, obstacle_present)
    positive_scalar(max_steering_rad, "max_steering_rad")
    if validate_values:
        finite_tensor(steering_rad, "steering_rad")
    penalty = -1 * torch.abs(steering_rad) / max_steering_rad
    return torch.where(obstacle_present, torch.zeros_like(penalty), penalty)
