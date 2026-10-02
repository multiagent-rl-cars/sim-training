"""CPU model compatibility checks; these do not establish simulated driving quality."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rc_car.models import CustomMultiInputExtractor, validate_feature_extractor_inputs

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_FIXTURE = Path(__file__).parent / "fixtures" / "legacy_feature_extractor.py"
LEGACY_SHIM = (
    REPO_ROOT
    / "source/car_training/car_training/tasks/manager_based/car_training/agents/feature_extractor.py"
)
LEGACY_MODULE_NAME = "car_training.tasks.manager_based.car_training.agents.feature_extractor"


def load_module_directly(path: Path, name: str) -> ModuleType:
    """Read a leaf file without executing the eager Isaac parent package imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_isolated_legacy_module(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Emulate only the legacy module namespace for pickle-reference compatibility.

    Empty parent namespaces avoid starting Isaac. This is intentionally not a
    claim that the real ``car_training`` import is simulator-independent.
    """
    parts = LEGACY_MODULE_NAME.split(".")
    for length in range(1, len(parts)):
        name = ".".join(parts[:length])
        package = ModuleType(name)
        package.__dict__["__path__"] = []
        monkeypatch.setitem(sys.modules, name, package)
    spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, LEGACY_MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def observation_space(obstacle_channels: int = 4, height: int = 48, width: int = 48) -> spaces.Dict:
    return spaces.Dict(
        {
            "track_image": spaces.Box(0.0, 1.0, (1, height, width), dtype=np.float32),
            "obstacle_image": spaces.Box(
                0.0, 1.0, (obstacle_channels, height, width), dtype=np.float32
            ),
            "robot_vel": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        }
    )


def observation_batch(space: spaces.Dict, batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(400)
    tensors = {}
    for key, value in space.spaces.items():
        assert isinstance(value, spaces.Box)
        tensors[key] = torch.rand((batch_size, *value.shape), generator=generator)
    tensors["robot_vel"] = 2.0 * tensors["robot_vel"] - 1.0
    return tensors


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads() -> Iterator[None]:
    # Tiny convolutions are substantially slower when dispatched to many CPU threads.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("training", [False, True])
def test_matches_frozen_legacy_state_outputs_and_gradients(training: bool) -> None:
    legacy = load_module_directly(LEGACY_FIXTURE, "legacy_model_characterization")
    space = observation_space()
    torch.manual_seed(73)
    old_model = legacy.CustomMultiInputExtractor(space, features_dim=32)
    torch.manual_seed(73)
    new_model = CustomMultiInputExtractor(space, features_dim=32)
    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    assert old_state.keys() == new_state.keys()
    for name in old_state:
        torch.testing.assert_close(old_state[name], new_state[name], rtol=0, atol=0)

    old_model.train(training)
    new_model.train(training)
    old_inputs = {key: value.requires_grad_() for key, value in observation_batch(space).items()}
    new_inputs = {key: value.detach().clone().requires_grad_() for key, value in old_inputs.items()}
    old_output, new_output = old_model(old_inputs), new_model(new_inputs)
    assert new_output.shape == (2, 32)
    assert torch.isfinite(new_output).all()
    torch.testing.assert_close(old_output, new_output, rtol=0, atol=0)
    old_output.square().mean().backward()
    new_output.square().mean().backward()
    for (old_name, old_parameter), (new_name, new_parameter) in zip(
        old_model.named_parameters(), new_model.named_parameters(), strict=True
    ):
        assert old_name == new_name
        assert old_parameter.grad is not None and new_parameter.grad is not None
        assert torch.isfinite(new_parameter.grad).all()
        torch.testing.assert_close(old_parameter.grad, new_parameter.grad, rtol=0, atol=0)
    for key in old_inputs:
        assert old_inputs[key].grad is not None and new_inputs[key].grad is not None
        torch.testing.assert_close(old_inputs[key].grad, new_inputs[key].grad, rtol=0, atol=0)
    for name, value in old_model.state_dict().items():
        torch.testing.assert_close(value, new_model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("obstacle_channels,height,width", [(1, 48, 48), (4, 48, 48), (4, 40, 48)])
def test_declared_shapes_and_normalized_inputs(
    obstacle_channels: int, height: int, width: int
) -> None:
    space = observation_space(obstacle_channels, height, width)
    inputs = observation_batch(space)
    # Velocity beyond the nominal normalization scale remains valid and unclipped.
    inputs["robot_vel"][0] = torch.tensor([-2.0, 3.0])
    unchanged = {key: value.clone() for key, value in inputs.items()}
    validate_feature_extractor_inputs(inputs, space)
    model = CustomMultiInputExtractor(space, features_dim=17).eval()
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (2, 17)
    assert torch.isfinite(output).all()
    for key in inputs:
        torch.testing.assert_close(inputs[key], unchanged[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "violation,match",
    [
        ("missing", "Missing observation"),
        ("channel_last", "batched shape"),
        ("batch", "batch size"),
        ("empty", "at least one"),
        ("dtype", "float32"),
        ("nan", "finite"),
        ("image_range", r"normalized to \[0, 1\]"),
    ],
)
def test_preflight_rejects_invalid_input_contract(violation: str, match: str) -> None:
    space = observation_space()
    inputs = observation_batch(space)
    if violation == "missing":
        del inputs["robot_vel"]
    elif violation == "channel_last":
        inputs["track_image"] = inputs["track_image"].permute(0, 2, 3, 1)
    elif violation == "batch":
        inputs["robot_vel"] = inputs["robot_vel"][:1]
    elif violation == "empty":
        inputs = {key: value[:0] for key, value in inputs.items()}
    elif violation == "dtype":
        inputs["track_image"] = inputs["track_image"].to(torch.uint8)
    elif violation == "nan":
        inputs["robot_vel"][0, 0] = torch.nan
    elif violation == "image_range":
        inputs["obstacle_image"][0, 0, 0, 0] = 255.0
    with pytest.raises(ValueError, match=match):
        validate_feature_extractor_inputs(inputs, space)


def test_preflight_rejects_different_image_spatial_dimensions() -> None:
    space = observation_space()
    space.spaces["obstacle_image"] = spaces.Box(0.0, 1.0, (4, 40, 48), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial dimensions"):
        validate_feature_extractor_inputs(observation_batch(space), space)


def test_batchnorm_initialization_and_train_eval_state() -> None:
    space = observation_space()
    model = CustomMultiInputExtractor(space, features_dim=16)
    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert batchnorms
    for module in batchnorms:
        assert module.num_batches_tracked is not None
        assert module.num_batches_tracked.item() == 1
    model.train()
    model(observation_batch(space))
    for module in batchnorms:
        assert module.num_batches_tracked is not None
        assert module.num_batches_tracked.item() == 2

    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        first = model(observation_batch(space))
        second = model(observation_batch(space))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_state_dict_round_trip(tmp_path: Path) -> None:
    space = observation_space()
    original = CustomMultiInputExtractor(space, features_dim=16).eval()
    path = tmp_path / "encoder.pt"
    torch.save(original.state_dict(), path)
    restored = CustomMultiInputExtractor(space, features_dim=16).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            original(observation_batch(space)), restored(observation_batch(space)), rtol=0, atol=0
        )


class TinyObservationEnv(gym.Env):
    """Synthetic CPU wiring fixture, deliberately unrelated to driving dynamics."""

    observation_space = observation_space()
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "track_image": self.np_random.random((1, 48, 48), dtype=np.float32),
            "obstacle_image": self.np_random.random((4, 48, 48), dtype=np.float32),
            "robot_vel": self.np_random.uniform(-1.0, 1.0, (2,)).astype(np.float32),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.steps = 0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        reward = -float(np.square(action).sum())
        return self._observation(), reward, False, self.steps >= 4, {}


def make_ppo(
    env: TinyObservationEnv, extractor: type[BaseFeaturesExtractor] = CustomMultiInputExtractor
) -> PPO:
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": extractor,
            "features_extractor_kwargs": {"features_dim": 16},
            "net_arch": {"pi": [16], "vf": [16]},
        },
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        seed=61,
        device="cpu",
    )


def test_cpu_ppo_update_and_sb3_checkpoint_round_trip(tmp_path: Path) -> None:
    env = TinyObservationEnv()
    model = make_ppo(env)
    before = {
        name: parameter.detach().clone() for name, parameter in model.policy.named_parameters()
    }
    model.learn(total_timesteps=4)
    assert model.num_timesteps == 4
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.policy.named_parameters()
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    observation, _ = env.reset(seed=19)
    expected_action, _ = model.predict(observation, deterministic=True)
    checkpoint = tmp_path / "synthetic_ppo.zip"
    model.save(checkpoint)
    restored = PPO.load(checkpoint, device="cpu")
    actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_array_equal(expected_action, actual_action)
    assert isinstance(restored.policy.features_extractor, CustomMultiInputExtractor)
    env.close()


def test_old_sb3_class_reference_loads_through_compatibility_shim(tmp_path: Path) -> None:
    """An old dotted-class checkpoint resolves to the new code with identical actions."""
    env = TinyObservationEnv()
    observation, _ = env.reset(seed=29)
    checkpoint = tmp_path / "old_module_reference.zip"
    with pytest.MonkeyPatch.context() as patch:
        legacy = install_isolated_legacy_module(LEGACY_FIXTURE, patch)
        model = make_ppo(env, legacy.CustomMultiInputExtractor)
        assert model.policy.features_extractor.__class__.__module__ == LEGACY_MODULE_NAME
        expected_action, _ = model.predict(observation, deterministic=True)
        model.save(checkpoint)

    with pytest.MonkeyPatch.context() as patch:
        install_isolated_legacy_module(LEGACY_SHIM, patch)
        restored = PPO.load(checkpoint, device="cpu")
        assert isinstance(restored.policy.features_extractor, CustomMultiInputExtractor)
        actual_action, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_array_equal(expected_action, actual_action)
    env.close()


def test_legacy_leaf_exports_resolve_to_portable_classes() -> None:
    shim = load_module_directly(LEGACY_SHIM, "legacy_feature_extractor_aliases")
    assert shim.CustomMultiInputExtractor is CustomMultiInputExtractor
    assert shim.ResidualBlock.__module__ == "rc_car.models.feature_extractor"


def test_portable_import_does_not_import_isaac() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rc_car.models; "
            "assert not any(name.split('.')[0] in "
            "{'isaaclab', 'isaacsim', 'omni', 'car_training'} for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
