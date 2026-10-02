"""Run the real smoke recorder body on CPU fakes; no Isaac or encoder startup."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from rc_car.lifecycle import finish_resources

ROOT = Path(__file__).resolve().parents[2]


def recorder(monkeypatch: pytest.MonkeyPatch, writer: MagicMock) -> Any:
    source = ROOT / "scripts/smoke_test.py"
    tree = ast.parse(source.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"record_motion", "check_finite"}
    ]
    imageio = ModuleType("imageio")
    v2 = ModuleType("imageio.v2")
    v2.__dict__["get_writer"] = MagicMock(return_value=writer)
    imageio.__dict__["v2"] = v2
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)
    namespace = {"torch": torch, "sys": sys, "finish_resources": finish_resources}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["record_motion"]


def environment() -> SimpleNamespace:
    position = torch.zeros((2, 3))
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=position))
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)})
    )
    state = SimpleNamespace(
        num_envs=2, device="cpu", step_dt=1 / 30, scene={"robot": robot, "overhead_camera": camera}
    )
    count = 0

    def step(_action: torch.Tensor) -> tuple[Any, ...]:
        nonlocal count
        count += 1
        position[:, 0] += torch.tensor([0.01, 0.02])
        done = torch.tensor([count == 2, False])
        if count == 2:
            position[0, 0] = 10  # A reset teleport must not count as driving.
        return (
            {"policy": torch.ones(2, 2)},
            torch.ones(2),
            done,
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    return SimpleNamespace(unwrapped=state, action_space=SimpleNamespace(shape=(2, 2)), step=step)


def test_motion_video_metadata_and_every_car_distance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = MagicMock()
    record = recorder(monkeypatch, writer)
    _, result = record(environment(), 3, tmp_path)
    assert result["nonreset_path_length_m"] == pytest.approx([0.02, 0.06], abs=1e-6)
    assert result["resets"] == 1
    assert result["videos"] == [{"path": "smoke.mp4", "expected_frames": 3}]
    assert writer.append_data.call_count == 3
    writer.close.assert_called_once_with()


def test_recording_failure_still_closes_without_masking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = RuntimeError("frame failed")
    writer = MagicMock()
    writer.append_data.side_effect = original
    writer.close.side_effect = RuntimeError("close failed")
    record = recorder(monkeypatch, writer)
    with pytest.raises(RuntimeError, match="frame failed") as caught:
        record(environment(), 3, tmp_path)
    assert caught.value is original
    assert "RuntimeError" in original.__notes__[0]
    writer.close.assert_called_once_with()


def test_empty_motion_check_is_rejected_before_opening_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = MagicMock()
    record = recorder(monkeypatch, writer)
    with pytest.raises(ValueError, match="at least one step"):
        record(environment(), 0, tmp_path)
    writer.close.assert_not_called()
