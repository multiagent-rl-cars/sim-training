import math

import pytest
import torch

from rc_car.telemetry import RewardRollup, count_termination_causes


def test_reward_rollup_uses_every_step_and_dt_once() -> None:
    rollup = RewardRollup(["speed", "steer"], 2)
    first = torch.tensor([[2.0, -1.0], [4.0, -2.0]], requires_grad=True)
    before = first.detach().clone()
    rollup.add(first, 0.1)
    rollup.add(torch.tensor([[10.0, -3.0], [20.0, -4.0]]), 0.2)
    snapshot = rollup.snapshot()
    assert snapshot["control_steps"] == 2
    assert snapshot["agent_steps"] == 4
    assert snapshot["sim_seconds_per_agent"] == pytest.approx(0.3)
    assert snapshot["weighted_mean_before_dt"] == {"speed": 9.0, "steer": -2.5}
    assert snapshot["contribution_mean_per_agent_step"] == pytest.approx(
        {"speed": 1.65, "steer": -0.425}
    )
    per_agent = snapshot["contribution_sum_per_agent"]
    assert isinstance(per_agent, dict)
    assert per_agent["speed"] == pytest.approx([2.2, 4.4])
    assert per_agent["steer"] == pytest.approx([-0.7, -1.0])
    torch.testing.assert_close(first, before)
    assert rollup.snapshot() == snapshot  # Reads do not advance state.
    rollup.clear()
    with pytest.raises(ValueError, match="empty"):
        rollup.snapshot()


@pytest.mark.parametrize("dt", [0.0, -1.0, math.nan, math.inf, True])
def test_invalid_time_does_not_advance_rollup(dt: float) -> None:
    rollup = RewardRollup(["speed"], 1)
    with pytest.raises(ValueError):
        rollup.add(torch.ones(1, 1), dt)
    assert rollup.steps == 0


def test_bad_shape_and_nonfinite_terms_are_rejected() -> None:
    rollup = RewardRollup(["speed"], 2)
    with pytest.raises(ValueError, match="shape"):
        rollup.add(torch.ones(2), 0.1)
    with pytest.raises(ValueError, match="floating"):
        rollup.add(torch.ones(2, 1, dtype=torch.int32), 0.1)
    rollup.add(torch.tensor([[math.nan], [0.0]]), 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        rollup.snapshot()


def test_input_storage_is_not_retained() -> None:
    rollup = RewardRollup(["speed"], 1)
    values = torch.ones(1, 1)
    rollup.add(values, 1.0)
    values.zero_()
    assert rollup.snapshot()["weighted_mean_before_dt"] == {"speed": 1.0}


def test_simultaneous_causes_and_stale_flags() -> None:
    dones = torch.tensor([True, False])
    masks = {"collision": torch.tensor([True, True]), "off_track": torch.tensor([True, False])}
    counts = count_termination_causes(masks, dones)
    assert counts == {"collision": 1, "off_track": 1}
    assert sum(counts.values()) != int(dones.sum())


def test_bad_termination_mask_rejected() -> None:
    with pytest.raises(ValueError, match="match"):
        count_termination_causes({"collision": torch.ones(2)}, torch.tensor([True, False]))


@pytest.mark.parametrize("names,agents", [([], 1), (["x", "x"], 1), ([""], 1), (["x"], 0)])
def test_invalid_reward_contract(names: list[str], agents: int) -> None:
    with pytest.raises(ValueError):
        RewardRollup(names, agents)
