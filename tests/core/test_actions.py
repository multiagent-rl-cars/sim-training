import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from rc_car.core.actions import legacy_action_targets


def targets(actions, minimum=10.0, maximum=30.0, max_steer=0.6, **kwargs):
    return legacy_action_targets(
        actions,
        min_wheel_speed_rad_s=minimum,
        max_wheel_speed_rad_s=maximum,
        max_steering_rad=max_steer,
        **kwargs,
    )


def test_legacy_action_golden_values_and_zero_is_not_stop():
    raw = torch.tensor(
        [[-100.0, -100.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [100.0, 100.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [10.0, -0.6],
            [12.384058440442351, -0.4569564935734589],
            [20.0, 0.0],
            [27.61594155955765, 0.4569564935734589],
            [30.0, 0.6],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(targets(raw), expected, rtol=1e-14, atol=1e-14)
    assert targets(torch.zeros((1, 2)), minimum=-15.0)[0, 0] == 7.5


@given(
    throttle=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    steering=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_action_limits_finite_output_and_immutability(throttle, steering):
    raw = torch.tensor([[throttle, steering]], dtype=torch.float64)
    before = raw.clone()
    output = targets(raw, validate_values=True)
    assert bool(torch.isfinite(output).all())
    assert 10.0 <= output[0, 0] <= 30.0
    assert -0.6 <= output[0, 1] <= 0.6
    torch.testing.assert_close(raw, before, rtol=0, atol=0)
    output.fill_(0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_action_mapping_remains_differentiable():
    raw = torch.zeros((2, 2), requires_grad=True)
    targets(raw).sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[10.0, 0.6], [10.0, 0.6]]))


def test_action_contract_empty_batch_invalid_shapes_and_limits():
    assert targets(torch.empty((0, 2))).shape == (0, 2)
    with pytest.raises(ValueError, match="shape"):
        targets(torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="minimum"):
        targets(torch.zeros((1, 2)), minimum=40.0)
    with pytest.raises(ValueError, match="finite"):
        targets(torch.zeros((1, 2)), maximum=math.inf)
    with pytest.raises(ValueError, match="positive"):
        targets(torch.zeros((1, 2)), max_steer=0.0)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_debug_contract_catches_nonfinite_network_output(invalid):
    with pytest.raises(ValueError, match="finite values"):
        targets(torch.tensor([[0.0, invalid]]), validate_values=True)
