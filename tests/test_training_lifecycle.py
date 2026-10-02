"""Run actual callback/recovery bodies without initializing the Isaac application.

These checks exercise the runner's CPU control flow and SB3 callback hooks. They
do not establish simulator scheduling or that a cloud shutdown preserves data.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback

from rc_car.lifecycle import best_effort_cleanup, finish_resources
from rc_car.telemetry import RewardRollup, count_termination_causes

RUNNER = Path(__file__).resolve().parents[1] / "scripts/training_experiment.py"


def callback_fixture() -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Compile the real Metrics class and termination adapter with fake state."""
    tree = ast.parse(RUNNER.read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "Metrics")
        or (isinstance(node, ast.FunctionDef) and node.name == "reset_counts")
    ]
    assert len(nodes) == 2
    events: list[dict[str, Any]] = []

    def event(name: str, **fields: Any) -> None:
        events.append({"event": name, **fields})

    namespace: dict[str, Any] = {
        "BaseCallback": BaseCallback,
        "RewardRollup": RewardRollup,
        "count_termination_causes": count_termination_causes,
        "finish_resources": finish_resources,
        "np": np,
        "time": SimpleNamespace(monotonic=lambda: 10.0),
        "config": {"num_envs": 2, "max_training_seconds": 1.0},
        "tracking": None,
        "event": event,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            reward_manager=SimpleNamespace(
                active_terms=["speed"], _step_reward=torch.tensor([[1.0], [2.0]])
            ),
            termination_manager=SimpleNamespace(
                active_terms=["off_track"],
                get_term=lambda _: torch.tensor([True, False]),
                dones=torch.tensor([True, False]),
            ),
            step_dt=0.1,
        )
    )
    callback = namespace["Metrics"](env)
    callback.model = SimpleNamespace(num_timesteps=2)
    return callback, events, namespace


def runner_learning_block(namespace: dict[str, Any]) -> None:
    """Execute the actual train() learn/except block, with supplied dependencies."""
    tree = ast.parse(RUNNER.read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    block = next(node for node in train.body if isinstance(node, ast.Try))
    assert "model.learn" in ast.unparse(block)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(RUNNER), "exec"), namespace)


def runner_resource_block(namespace: dict[str, Any]) -> None:
    """Execute the actual main() try/finally around train and resource release."""
    tree = ast.parse(RUNNER.read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    block = next(node for node in main.body if isinstance(node, ast.Try))
    assert "finish_resources" in ast.unparse(block)
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(RUNNER), "exec"), namespace)


def test_budget_stop_flushes_partial_rollout_without_claiming_optimization() -> None:
    callback, events, _ = callback_fixture()
    callback.wall_start = 0.0
    callback.update_locals({"rewards": np.array([0.1, 0.2]), "dones": np.array([True, False])})
    # SB3 calls on_training_end after on_step returns False, without optimizing
    # this interrupted rollout. Exercise those same public callback hooks.
    assert callback.on_step() is False
    callback.on_training_end()
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "partial_rollout"
    assert event["optimized"] is False
    assert event["reason"] == "training_end"
    assert event["agent_timesteps"] == 2
    assert event["reward_terms_rollout"]["control_steps"] == 1
    assert event["reward_terms_rollout"]["agent_steps"] == 2
    assert event["reward_terms_rollout"]["contribution_mean_per_agent_step"] == pytest.approx(
        {"speed": 0.15}
    )
    assert callback.completed_lengths == [1]
    assert callback.terminations == {"off_track": 1}
    assert callback.reward_rollup.steps == 0
    callback.on_training_end()
    assert len(events) == 1  # A repeated cleanup does not duplicate observations.


@pytest.mark.parametrize("invalid_terms", [False, True])
def test_training_end_flushes_even_when_video_close_fails(invalid_terms: bool) -> None:
    callback, events, _ = callback_fixture()
    callback.reward_rollup.add(torch.full((2, 1), torch.nan if invalid_terms else 1.0), 0.1)
    media_error = OSError("synthetic media failure")

    def close() -> None:
        raise media_error

    callback.video_writer = SimpleNamespace(close=close)
    with pytest.raises(OSError) as caught:
        callback.on_training_end()
    assert caught.value is media_error
    if invalid_terms:
        assert events == []
        assert any("partial rollout" in note for note in media_error.__notes__)
    else:
        assert events[0]["optimized"] is False
        assert events[0]["reason"] == "training_end"
        assert callback.reward_rollup.steps == 0


@pytest.mark.parametrize("nonfinite_rollup", [False, True])
@pytest.mark.parametrize("checkpoint_failure", [False, True])
def test_runner_failure_attempts_checkpoint_and_all_cleanup_without_masking_error(
    tmp_path: Path, nonfinite_rollup: bool, checkpoint_failure: bool
) -> None:
    callback, events, _ = callback_fixture()
    reward = torch.full((2, 1), torch.nan if nonfinite_rollup else 1.0)
    callback.reward_rollup.add(reward, 0.1)
    callback.num_timesteps = 2
    original_error = RuntimeError("synthetic learner failure")
    attempts: list[str] = []

    def learn(**_: Any) -> None:
        attempts.append("learn")
        raise original_error

    def save(path: Path) -> None:
        attempts.append("checkpoint")
        assert path == tmp_path / "interrupted"
        if checkpoint_failure:
            raise OSError("sensitive exception text must not enter recovery notes")

    def close() -> None:
        attempts.append("video")
        raise OSError("synthetic media failure")

    callback.video_writer = SimpleNamespace(close=close)
    with pytest.raises(RuntimeError) as caught:
        runner_learning_block(
            {
                "model": SimpleNamespace(learn=learn, save=save),
                "callback": callback,
                "config": {"total_timesteps": 100},
                "args": SimpleNamespace(run_dir=tmp_path),
                "best_effort_cleanup": best_effort_cleanup,
            }
        )
    assert caught.value is original_error
    assert attempts == ["learn", "checkpoint", "video"]
    notes = " ".join(original_error.__notes__)
    assert "video close" in notes
    assert "sensitive exception text" not in notes
    assert ("interrupted checkpoint" in notes) == checkpoint_failure
    assert ("partial rollout" in notes) == nonfinite_rollup
    if nonfinite_rollup:
        assert events == []  # Invalid values do not enter a JSON event.
    else:
        assert len(events) == 1
        assert events[0]["reason"] == "exception"
        assert events[0]["optimized"] is False
        assert events[0]["reward_terms_rollout"]["agent_steps"] == 2


def test_cleanup_preserves_interrupt_and_attempts_later_operations() -> None:
    interrupt = KeyboardInterrupt()
    attempts: list[str] = []

    def interrupted_cleanup() -> None:
        attempts.append("first")
        raise KeyboardInterrupt()

    best_effort_cleanup(
        interrupt,
        [("first", interrupted_cleanup), ("second", lambda: attempts.append("second"))],
    )
    assert attempts == ["first", "second"]
    assert interrupt.__notes__ == ["Recovery 'first' failed: KeyboardInterrupt"]


@pytest.mark.parametrize("learner_failure", [False, True])
@pytest.mark.parametrize("finish_failure", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
def test_outer_cleanup_preserves_learner_or_first_cleanup_error_and_closes_environment(
    learner_failure: bool, finish_failure: bool, close_failure: bool
) -> None:
    attempts: list[str] = []
    learner_error = RuntimeError("synthetic learner failure") if learner_failure else None
    finish_error = OSError("synthetic tracking finish failure") if finish_failure else None
    close_error = ValueError("synthetic environment close failure") if close_failure else None

    def train(_: object) -> None:
        attempts.append("train")
        if learner_error is not None:
            raise learner_error

    def finish(*, exit_code: int) -> None:
        attempts.append(f"finish:{exit_code}")
        if finish_error is not None:
            raise finish_error

    def close() -> None:
        attempts.append("close")
        if close_error is not None:
            raise close_error

    namespace = {
        "args": SimpleNamespace(mode="train"),
        "config": {},
        "env": SimpleNamespace(close=close),
        "tracking": SimpleNamespace(finish=finish),
        "train": train,
        "sys": sys,
        "finish_resources": finish_resources,
    }
    expected_error = learner_error or finish_error or close_error
    if expected_error is None:
        runner_resource_block(namespace)
    else:
        with pytest.raises(type(expected_error)) as caught:
            runner_resource_block(namespace)
        assert caught.value is expected_error
        notes = " ".join(getattr(expected_error, "__notes__", []))
        if finish_failure:
            assert "tracking finish" in notes
        if close_failure:
            assert "environment close" in notes
    assert attempts == ["train", f"finish:{1 if learner_failure else 0}", "close"]
