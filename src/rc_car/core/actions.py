"""Legacy policy output mapping, independent of joints or a simulator."""

import math

import torch

from rc_car.core._validation import finite_tensor, floating_tensor, positive_scalar


def legacy_action_targets(
    raw_actions: torch.Tensor,
    *,
    min_wheel_speed_rad_s: float,
    max_wheel_speed_rad_s: float,
    max_steering_rad: float,
    validate_values: bool = False,
) -> torch.Tensor:
    """Map (N, 2) raw actions to [wheel rad/s, steering rad] using tanh.

    This preserves the inherited tanh and nonzero driving minimum. Raw zero
    maps to the midpoint wheel speed, not braking. Both wheels receive the same
    targets in the Isaac adapter; this function does not implement Ackermann angles.
    Finite checks are opt-in because reading CUDA values synchronizes the device.
    """
    floating_tensor(raw_actions, "raw_actions")
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 2:
        raise ValueError("raw_actions must have shape (N, 2)")
    if not math.isfinite(min_wheel_speed_rad_s) or not math.isfinite(max_wheel_speed_rad_s):
        raise ValueError("wheel speed limits must be finite")
    if min_wheel_speed_rad_s > max_wheel_speed_rad_s:
        raise ValueError("minimum wheel speed must not exceed maximum wheel speed")
    positive_scalar(max_steering_rad, "max_steering_rad")
    if validate_values:
        finite_tensor(raw_actions, "raw_actions")
    actions = raw_actions.tanh()
    throttle_norm = 0.5 * (actions[:, 0] + 1.0)
    wheel_speed = min_wheel_speed_rad_s + throttle_norm * (
        max_wheel_speed_rad_s - min_wheel_speed_rad_s
    )
    steering = max_steering_rad * actions[:, 1]
    return torch.stack((wheel_speed, steering), dim=-1)
