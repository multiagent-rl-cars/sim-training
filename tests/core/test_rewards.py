import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from rc_car.core.rewards import legacy_line_proximity, legacy_steering_penalty


def test_legacy_proximity_is_unclipped_and_obstacle_gated():
    distance = torch.tensor([0.0, 20.0, 40.0, 80.0, 0.0, 80.0])
    obstacle = torch.tensor([False, False, False, False, True, True])
    expected = torch.tensor([1.0, 0.5, 0.0, -1.0, 0.0, 0.0])
    torch.testing.assert_close(legacy_line_proximity(distance, obstacle), expected)


def test_steering_penalizes_magnitude_not_sign_or_rate():
    steering = torch.tensor([-1.2, -0.6, 0.0, 0.6, 1.2, 0.6])
    obstacle = torch.tensor([False, False, False, False, False, True])
    expected = torch.tensor([-2.0, -1.0, 0.0, -1.0, -2.0, 0.0])
    torch.testing.assert_close(
        legacy_steering_penalty(steering, obstacle, max_steering_rad=0.6), expected
    )


@given(
    distance=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
    increment=st.floats(0, 1000, allow_nan=False, allow_infinity=False),
)
def test_farther_from_contour_cannot_improve_ungated_reward(distance, increment):
    distances = torch.tensor([distance, distance + increment], dtype=torch.float64)
    before = distances.clone()
    obstacles = torch.zeros((2,), dtype=torch.bool)
    result = legacy_line_proximity(distances, obstacles, validate_values=True)
    assert result[0] >= result[1]
    torch.testing.assert_close(distances, before, rtol=0, atol=0)


def test_reward_contracts_reject_invalid_shapes_values_and_units():
    with pytest.raises(ValueError, match="boolean"):
        legacy_line_proximity(torch.zeros((2,)), torch.zeros((2,)))
    with pytest.raises(ValueError, match="shape"):
        legacy_line_proximity(torch.zeros((2, 1)), torch.zeros((2,), dtype=torch.bool))
    with pytest.raises(ValueError, match="positive"):
        legacy_line_proximity(
            torch.zeros((1,)), torch.zeros((1,), dtype=torch.bool), distance_scale_px=0
        )
    with pytest.raises(ValueError, match="nonnegative"):
        legacy_line_proximity(torch.tensor([-1.0]), torch.tensor([False]), validate_values=True)
    with pytest.raises(ValueError, match="finite values"):
        legacy_steering_penalty(
            torch.tensor([math.nan]),
            torch.tensor([False]),
            max_steering_rad=0.6,
            validate_values=True,
        )
    assert legacy_line_proximity(torch.empty((0,)), torch.empty((0,), dtype=torch.bool)).shape == (
        0,
    )
