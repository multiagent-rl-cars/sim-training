import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from rc_car.core.geometry import (
    convert_coordinates,
    heading_to_image_vector,
    legacy_aligned_velocity,
    world_velocity_to_image,
)


@pytest.mark.parametrize("preserve_aspect_ratio", [False, True])
def test_coordinate_edges_center_and_z_are_explicit(preserve_aspect_ratio):
    positions = torch.tensor([[-5.0, 5.0, 7.0], [0.0, 0.0, 8.0], [5.0, -5.0, 9.0]])
    original = positions.clone()
    actual = convert_coordinates(
        positions,
        (10.0, 10.0),
        (512.0, 512.0),
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    torch.testing.assert_close(actual, torch.tensor([[0.0, 0.0], [256.0, 256.0], [512.0, 512.0]]))
    torch.testing.assert_close(positions, original, rtol=0, atol=0)


def test_letterboxing_preserves_shape_and_does_not_stretch_track():
    points = torch.tensor([[[-2.0, 1.0], [2.0, -1.0], [0.0, 0.0]]])
    actual = convert_coordinates(points, (4.0, 2.0), (400.0, 400.0))
    torch.testing.assert_close(
        actual, torch.tensor([[[0.0, 100.0], [400.0, 300.0], [200.0, 200.0]]])
    )
    stretched = convert_coordinates(points, (4.0, 2.0), (400.0, 400.0), preserve_aspect_ratio=False)
    torch.testing.assert_close(
        stretched, torch.tensor([[[0.0, 0.0], [400.0, 400.0], [200.0, 200.0]]])
    )


@given(
    x=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    y=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    width=st.floats(0.5, 100, allow_nan=False, allow_infinity=False),
    height=st.floats(0.5, 100, allow_nan=False, allow_infinity=False),
    aspect=st.booleans(),
)
def test_coordinate_roundtrip_including_outside_track(x, y, width, height, aspect):
    original = torch.tensor([[x, y]], dtype=torch.float64)
    image = convert_coordinates(
        original,
        (width, height),
        (640, 480),
        dtype=torch.float64,
        preserve_aspect_ratio=aspect,
        validate_values=True,
    )
    restored = convert_coordinates(
        image,
        (width, height),
        (640, 480),
        dtype=torch.float64,
        image_to_world=True,
        preserve_aspect_ratio=aspect,
        validate_values=True,
    )
    torch.testing.assert_close(restored, original, rtol=1e-12, atol=1e-11)
    assert bool(torch.isfinite(image).all())


def test_image_velocity_is_repeatable_and_owns_its_storage():
    source = torch.tensor([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])
    original = source.clone()
    first = world_velocity_to_image(source)
    second = world_velocity_to_image(source)
    torch.testing.assert_close(first, torch.tensor([[1.0, -2.0], [-4.0, 5.0]]))
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    first[0, 0] = 123.0
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_heading_cardinal_directions_and_legacy_local_axes():
    heading = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2], dtype=torch.float64)
    expected_direction = torch.tensor(
        [[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(
        heading_to_image_vector(heading), expected_direction, atol=1e-14, rtol=0
    )
    velocity = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float64)
    expected_velocity = torch.tensor(
        [[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(
        legacy_aligned_velocity(velocity, heading), expected_velocity, atol=1e-14, rtol=0
    )
    normalized = legacy_aligned_velocity(velocity, heading, normalization_speed_mps=0.8)
    torch.testing.assert_close(normalized, expected_velocity / 0.8, atol=1e-14, rtol=0)


@given(
    vx=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    vy=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    heading=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
)
def test_alignment_preserves_speed_and_inputs(vx, vy, heading):
    velocity = torch.tensor([[vx, vy, 9.0]], dtype=torch.float64)
    angles = torch.tensor([heading], dtype=torch.float64)
    original_velocity, original_angles = velocity.clone(), angles.clone()
    transformed = legacy_aligned_velocity(velocity, angles, validate_values=True)
    torch.testing.assert_close(
        torch.linalg.vector_norm(transformed), torch.linalg.vector_norm(velocity[:, :2])
    )
    torch.testing.assert_close(velocity, original_velocity, rtol=0, atol=0)
    torch.testing.assert_close(angles, original_angles, rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [0.0, -1.0, math.inf, math.nan])
def test_invalid_coordinate_scale_is_rejected(dimension):
    with pytest.raises(ValueError, match="finite and positive"):
        convert_coordinates(torch.zeros((1, 2)), (dimension, 1.0), (100, 100))


def test_empty_batches_shapes_and_dtype():
    assert convert_coordinates(torch.empty((0, 3)), (10, 10), (100, 100)).shape == (0, 2)
    assert world_velocity_to_image(torch.empty((0, 3))).shape == (0, 2)
    assert legacy_aligned_velocity(torch.empty((0, 3)), torch.empty((0,))).shape == (0, 2)
    assert (
        convert_coordinates(torch.zeros((1, 2), dtype=torch.float64), (10, 10), (100, 100)).dtype
        == torch.float32
    )
    with pytest.raises(ValueError, match="shape"):
        world_velocity_to_image(torch.zeros((2,)))
    with pytest.raises(ValueError, match="shape"):
        legacy_aligned_velocity(torch.zeros((2, 3)), torch.zeros((3,)))
    with pytest.raises(TypeError, match="floating-point"):
        heading_to_image_vector(torch.tensor([0]))


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_debug_finite_contract_rejects_invalid_physics(invalid):
    with pytest.raises(ValueError, match="finite values"):
        world_velocity_to_image(torch.tensor([[0.0, invalid]]), validate_values=True)
    with pytest.raises(ValueError, match="finite values"):
        heading_to_image_vector(torch.tensor([invalid]), validate_values=True)
    with pytest.raises(ValueError, match="finite values"):
        legacy_aligned_velocity(torch.zeros((1, 2)), torch.tensor([invalid]), validate_values=True)
    with pytest.raises(ValueError, match="finite values"):
        convert_coordinates(
            torch.tensor([[invalid, 0.0]]), (10, 10), (100, 100), validate_values=True
        )
