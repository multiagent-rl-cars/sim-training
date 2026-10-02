"""Legacy coordinate conventions expressed independently of Isaac.

World coordinates use a centered origin and positive y up; image coordinates
start at the top left with positive y down. Dimensions are (width, height).
The mapping deliberately preserves the historical width/height convention:
world edges map to 0 and width/height, not to the last pixel center.
"""

import math

import torch

from rc_car.core._validation import finite_tensor, floating_tensor, positive_scalar


def convert_coordinates(
    input_coords: torch.Tensor,
    world_dimensions_m: tuple[float, float],
    image_dimensions_px: tuple[float, float],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    image_to_world: bool = False,
    preserve_aspect_ratio: bool = True,
    *,
    validate_values: bool = False,
) -> torch.Tensor:
    """Map (..., 2+) positions between world metres and image pixels.

    Trailing z/other coordinates are ignored. Aspect preservation letterboxes the
    world at the image center. Output is (..., 2), float32 by legacy default.
    Inputs are never modified. ``validate_values`` additionally rejects NaN/Inf;
    leave it off in an already-validated GPU loop to avoid device synchronization.
    """
    if not isinstance(input_coords, torch.Tensor):
        input_coords = torch.tensor(input_coords, dtype=dtype, device=device)
    if input_coords.ndim < 1 or input_coords.shape[-1] < 2:
        raise ValueError("input_coords must have shape (..., 2+) ")
    if len(world_dimensions_m) != 2 or len(image_dimensions_px) != 2:
        raise ValueError("world and image dimensions must each contain width and height")
    for value, name in zip(
        (*world_dimensions_m, *image_dimensions_px),
        ("world width", "world height", "image width", "image height"),
        strict=True,
    ):
        positive_scalar(value, name)
    if validate_values:
        finite_tensor(input_coords[..., :2], "input_coords")

    width_world, height_world = world_dimensions_m
    width_image, height_image = image_dimensions_px
    x_input = input_coords[..., 0]
    y_input = input_coords[..., 1]
    sx = width_image / width_world
    sy = height_image / height_world
    scale = min(sx, sy)
    pad_x = (width_image - scale * width_world) / 2
    pad_y = (height_image - scale * height_world) / 2

    if preserve_aspect_ratio:
        if image_to_world:
            x_output = (x_input - pad_x) / scale - width_world / 2
            y_output = height_world / 2 - (y_input - pad_y) / scale
        else:
            x_output = (x_input + width_world / 2) * scale + pad_x
            y_output = (height_world / 2 - y_input) * scale + pad_y
    elif image_to_world:
        x_output = (x_input / width_image) * width_world - width_world / 2
        y_output = height_world / 2 - (y_input / height_image) * height_world
    else:
        x_output = (x_input + width_world / 2) * sx
        y_output = (height_world / 2 - y_input) * sy

    output = torch.stack((x_output, y_output), dim=-1)
    if device is not None:
        output = output.to(device)
    return output.to(dtype=dtype)


def world_velocity_to_image(
    velocity_mps: torch.Tensor, *, validate_values: bool = False
) -> torch.Tensor:
    """Return (vx, -vy) in m/s, not pixels/s, without aliasing input storage."""
    floating_tensor(velocity_mps, "velocity_mps")
    if velocity_mps.ndim != 2 or velocity_mps.shape[1] < 2:
        raise ValueError("velocity_mps must have shape (N, 2+) ")
    if validate_values:
        finite_tensor(velocity_mps[:, :2], "velocity_mps")
    return torch.stack((velocity_mps[:, 0], -velocity_mps[:, 1]), dim=-1)


def heading_to_image_vector(
    heading_rad: torch.Tensor, *, validate_values: bool = False
) -> torch.Tensor:
    """World heading radians to image-frame unit direction (cos(yaw), -sin(yaw))."""
    floating_tensor(heading_rad, "heading_rad")
    if heading_rad.ndim != 1:
        raise ValueError("heading_rad must have shape (N,)")
    if validate_values:
        finite_tensor(heading_rad, "heading_rad")
    return torch.stack((torch.cos(heading_rad), -torch.sin(heading_rad)), dim=-1)


def legacy_aligned_velocity(
    velocity_mps: torch.Tensor,
    heading_rad: torch.Tensor,
    *,
    normalization_speed_mps: float | None = None,
    validate_values: bool = False,
) -> torch.Tensor:
    """Preserve legacy R(-yaw - pi/2) velocity input, with optional normalization.

    This is the historical actor convention, not a new forward/lateral convention.
    For yaw=0, world velocity (1, 0) maps approximately to (0, -1).
    Normalization divides before rotation, preserving floating-point order.
    """
    floating_tensor(velocity_mps, "velocity_mps")
    floating_tensor(heading_rad, "heading_rad")
    if velocity_mps.ndim != 2 or velocity_mps.shape[1] < 2:
        raise ValueError("velocity_mps must have shape (N, 2+) ")
    if heading_rad.shape != velocity_mps.shape[:1]:
        raise ValueError("heading_rad must have shape (N,) matching velocity_mps")
    if velocity_mps.device != heading_rad.device or velocity_mps.dtype != heading_rad.dtype:
        raise ValueError("velocity_mps and heading_rad must share device and dtype")
    if validate_values:
        finite_tensor(velocity_mps[:, :2], "velocity_mps")
        finite_tensor(heading_rad, "heading_rad")
    velocity = velocity_mps[:, :2]
    if normalization_speed_mps is not None:
        positive_scalar(normalization_speed_mps, "normalization_speed_mps")
        velocity = velocity / normalization_speed_mps
    cos_heading = torch.cos(-heading_rad - math.pi / 2)
    sin_heading = torch.sin(-heading_rad - math.pi / 2)
    rotation = torch.stack(
        (
            torch.stack((cos_heading, -sin_heading), dim=-1),
            torch.stack((sin_heading, cos_heading), dim=-1),
        ),
        dim=-2,
    )
    return torch.bmm(rotation, velocity.unsqueeze(-1)).squeeze(-1)
