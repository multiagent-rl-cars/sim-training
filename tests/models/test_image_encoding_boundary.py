"""CPU image-boundary checks, not a simulator/rendering or driving claim.

Portable tests use the real task declarations and SB3 policies with a narrow
reference for pinned Lab's clone/clip/scale/cast boundary. Vendored, hash-checked
copies of the exact upstream manager/wrapper execute with fake environment
state on every checkout; these tests never import or start Isaac.
"""

import __future__

import ast
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.torch_layers import CombinedExtractor

from rc_car.models import CustomMultiInputExtractor
from rc_car.observation_contract import (
    POLICY_IMAGE_ENCODING,
    reject_legacy_car_checkpoint,
    require_policy_image_encoding,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "source/car_training/car_training/tasks/manager_based/car_training"
LAB_FIXTURES = Path(__file__).parent / "fixtures/isaaclab_2_3_0"
CASES = [("car_training_env_cfg.py", 1), ("multiagent_training_env_cfg.py", 4)]


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def terms(filename):
    """Read actual policy term clip/scale declarations without Isaac imports."""
    body = ast.parse((CONFIGS / filename).read_text()).body
    for name in ("ObservationsCfg", "PolicyCfg"):
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == name)
    result = {}
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if isinstance(node.targets[0], ast.Name):
                result[node.targets[0].id] = {
                    kw.arg: ast.literal_eval(kw.value)
                    for kw in node.value.keywords
                    if kw.arg in {"scale", "clip"}
                }
    assert set(result) == {"track_image", "obstacle_image", "robot_vel"}
    assert result["robot_vel"] == {}  # No new velocity transform.
    for name in ("track_image", "obstacle_image"):
        assert result[name] == {"clip": (0.0, 255.0), "scale": 255.0}
    return result


def helper_values(channels):
    # A line-shaped black region on white, with differently oriented obstacle channels.
    track = torch.ones((2, 1, 48, 48))
    track[:, :, :, 20:28] = 0
    obstacle = torch.ones((2, channels, 48, 48))
    for channel in range(channels):
        obstacle[:, channel, 4 + channel : 12 + channel, 8:16] = 0
    return {
        "track_image": track,
        "obstacle_image": obstacle,
        "robot_vel": torch.tensor([[-2.0, 3.0], [0.125, -0.75]]),
    }


def policy_observations(values, term_config):
    # Faithful narrow reference: pinned ObservationManager clones BEFORE in-place operations.
    result = {}
    for key, source in values.items():
        value = source.clone()
        cfg = term_config[key]
        if "clip" in cfg:
            value.clip_(*cfg["clip"])
        if "scale" in cfg:
            value.mul_(cfg["scale"])
        result[key] = value
    return result


def spaces_for(channels, *, wrapped=True):
    dtype = np.uint8 if wrapped else np.float32
    return gym.spaces.Dict(
        {
            "track_image": gym.spaces.Box(0, 255, (1, 48, 48), dtype),
            "obstacle_image": gym.spaces.Box(0, 255, (channels, 48, 48), dtype),
            "robot_vel": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        }
    )


@pytest.mark.parametrize("filename,channels", CASES)
def test_both_real_extractors_receive_binary_unit_pixels_without_shared_mutation(
    filename, channels
):
    values = helper_values(channels)
    unchanged = {key: value.clone() for key, value in values.items()}
    policy_obs = policy_observations(values, terms(filename))
    # Pinned Lab casts only image terms to uint8; SB3 handles their sole normalization.
    wrapped = {
        key: val.to(torch.uint8) if "image" in key else val for key, val in policy_obs.items()
    }
    extractor = CombinedExtractor if channels == 1 else CustomMultiInputExtractor
    torch.manual_seed(41)
    policy = MultiInputActorCriticPolicy(
        spaces_for(channels),
        gym.spaces.Box(-100, 100, (2,), np.float32),
        lambda _: 0.0003,
        features_extractor_class=extractor,
    )
    seen = []
    handle = policy.features_extractor.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0])
    )
    policy.extract_features(wrapped)
    handle.remove()
    assert policy.normalize_images is True
    for key, expected in values.items():
        torch.testing.assert_close(seen[0][key], expected, rtol=0, atol=0)
        torch.testing.assert_close(values[key], unchanged[key], rtol=0, atol=0)
    # No shape or initialization change: same spaces/seed construct identical state tensors.
    torch.manual_seed(41)
    reference = MultiInputActorCriticPolicy(
        spaces_for(channels),
        gym.spaces.Box(-100, 100, (2,), np.float32),
        lambda _: 0.0003,
        features_extractor_class=extractor,
    )
    assert policy.state_dict().keys() == reference.state_dict().keys()
    # BatchNorm running buffers may have advanced during the observed forward, not weights.
    for (name, value), (other_name, other) in zip(
        policy.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(value, other, rtol=0, atol=0)


class ImageEnvironment(gym.Env):
    def __init__(self, channels):
        self.observation_space = spaces_for(channels)
        self.action_space = gym.spaces.Box(-100, 100, (2,), np.float32)


@pytest.mark.parametrize("channels", [1, 4])
def test_old_checkpoint_rejected_and_new_marker_round_trips(channels, tmp_path):
    kwargs = {} if channels == 1 else {"features_extractor_class": CustomMultiInputExtractor}
    model = PPO(
        "MultiInputPolicy",
        ImageEnvironment(channels),
        n_steps=2,
        batch_size=2,
        policy_kwargs=kwargs,
        device="cpu",
    )
    old = tmp_path / "legacy.zip"
    model.save(old)
    loaded = PPO.load(old, device="cpu")
    with pytest.raises(ValueError, match="encoding mismatch"):
        require_policy_image_encoding(getattr(loaded, "rc_car_image_encoding", None))
    model.__dict__["rc_car_image_encoding"] = POLICY_IMAGE_ENCODING
    current = tmp_path / "current.zip"
    model.save(current)
    loaded = PPO.load(current, device="cpu")
    require_policy_image_encoding(
        getattr(loaded, "rc_car_image_encoding", None),
        normalize_images=loaded.policy.normalize_images,
    )
    with pytest.raises(ValueError, match="encoding mismatch"):
        require_policy_image_encoding(
            getattr(loaded, "rc_car_image_encoding", None), normalize_images=False
        )


def test_supported_evaluator_rejects_legacy_manifest_before_kit(tmp_path):
    source = ast.parse((ROOT / "scripts/training_experiment.py").read_text())
    guard = next(
        node
        for node in source.body
        if isinstance(node, ast.If) and "require_policy_image_encoding" in ast.unparse(node)
    )
    launcher = next(
        node
        for node in source.body
        if isinstance(node, ast.Assign) and ast.unparse(node).startswith("app = AppLauncher")
    )
    assert guard.lineno < launcher.lineno
    namespace = {
        "json": json,
        "args": SimpleNamespace(mode="end", run_dir=tmp_path),
        "require_policy_image_encoding": require_policy_image_encoding,
    }
    for manifest in ({"behavior_version": "astra-foundation-v2"}, {"policy_image_encoding": "old"}):
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="encoding mismatch"):
            exec(
                compile(ast.Module(body=[guard], type_ignores=[]), "evaluator_guard", "exec"),
                namespace,
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"policy_image_encoding": POLICY_IMAGE_ENCODING})
    )
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "evaluator_guard", "exec"), namespace)


@pytest.mark.parametrize(
    "filename", ["play.py", "train.py", "train_combined_res.py", "train_mlp.py"]
)
def test_each_legacy_entrypoint_guards_car_load_before_kit(filename):
    tree = ast.parse((ROOT / "scripts/sb3" / filename).read_text())
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "reject_legacy_car_checkpoint"
    )
    launcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node).startswith("app_launcher = AppLauncher")
    )
    assert guard.lineno < launcher.lineno
    for checkpoint in ("old.zip", None):
        arguments = SimpleNamespace(task="Basic-Car-Training-v0", checkpoint=checkpoint)
        scope = {
            "args_cli": arguments,
            "reject_legacy_car_checkpoint": reject_legacy_car_checkpoint,
        }
        code = compile(ast.Module(body=[guard], type_ignores=[]), "legacy_guard", "exec")
        if filename == "play.py" or checkpoint is not None:
            with pytest.raises(ValueError, match="unsupported"):
                exec(code, scope)
        else:
            exec(code, scope)
        arguments.task = "Isaac-Cartpole-v0"
        exec(code, scope)


@pytest.mark.parametrize(
    "encoding,normalize", [(None, True), ("old", True), (POLICY_IMAGE_ENCODING, False)]
)
def test_actual_evaluator_rejects_loaded_checkpoint_before_reset(encoding, normalize, tmp_path):
    tree = ast.parse((ROOT / "scripts/training_experiment.py").read_text())
    evaluate = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    # Execute the actual load/guard prefix, without the unrelated video import or simulator.
    prefix = []
    for node in evaluate.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "wrapped.seed"
        ):
            break
        if not isinstance(node, ast.Import):
            prefix.append(node)
    loaded = SimpleNamespace(
        rc_car_image_encoding=encoding, policy=SimpleNamespace(normalize_images=normalize)
    )
    scope = {
        "env": object(),
        "Sb3VecEnvWrapper": lambda _env: object(),
        "PPO": SimpleNamespace(load=lambda *_args, **_kwargs: loaded),
        "args": SimpleNamespace(run_dir=tmp_path, mode="end", device="cpu"),
        "require_policy_image_encoding": require_policy_image_encoding,
    }
    with pytest.raises(ValueError, match="encoding mismatch"):
        exec(
            compile(ast.Module(body=prefix, type_ignores=[]), "actual_evaluator_guard", "exec"),
            scope,
        )


def isolated_methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods: list[ast.stmt] = [
        n for n in body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods if isinstance(n, ast.FunctionDef)} == set(names)
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return {name: namespace[name] for name in names}


@pytest.mark.parametrize(
    "filename,sha256",
    [
        (
            "observation_manager.py",
            "468ed890539f2de29dc0c27024b0ad74667209b884e0431456e1455c860cafae",
        ),
        ("sb3.py", "fee16bf9acf58b802e9a2d483ac90769a661dc745f7d504ab222eeb2fec1d9e4"),
    ],
)
def test_upstream_lab_fixtures_are_exact_pinned_bytes(filename, sha256):
    assert hashlib.sha256((LAB_FIXTURES / filename).read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize("filename,channels", CASES)
def test_actual_pinned_lab_manager_and_reset_step_terminal_boundary(filename, channels):
    term_config, values = terms(filename), helper_values(channels)
    before = {key: value.clone() for key, value in values.items()}
    cfgs = [
        SimpleNamespace(
            func=lambda _env, key=key: values[key],
            params={},
            modifiers=None,
            noise=None,
            clip=term_config[key].get("clip"),
            scale=term_config[key].get("scale"),
            history_length=0,
        )
        for key in values
    ]
    manager_methods = isolated_methods(
        LAB_FIXTURES / "observation_manager.py",
        "ObservationManager",
        {"compute_group"},
        {
            "torch": torch,
            "noise": SimpleNamespace(
                NoiseCfg=type("NoiseCfg", (), {}), NoiseModelCfg=type("NoiseModelCfg", (), {})
            ),
        },
    )
    manager = SimpleNamespace(
        _group_obs_term_names={"policy": list(values)},
        _group_obs_term_cfgs={"policy": cfgs},
        _env=None,
        _group_obs_concatenate={"policy": False},
    )

    def observations():
        return {"policy": manager_methods["compute_group"](manager, "policy")}

    class VecEnvInit:
        def __init__(self, num_envs, observation_space, action_space):
            self.observation_space, self.action_space = observation_space, action_space

    wrapped_methods = isolated_methods(
        LAB_FIXTURES / "sb3.py",
        "Sb3VecEnvWrapper",
        {"_process_spaces", "_process_obs", "_process_extras", "reset", "step_async", "step_wait"},
        {
            "gym": gym,
            "np": np,
            "torch": torch,
            "VecEnv": VecEnvInit,
            "is_image_space": is_image_space,
            "is_image_space_channels_first": is_image_space_channels_first,
        },
    )
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Dict({"policy": spaces_for(channels, wrapped=False)}),
        single_action_space=gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        reset=lambda: (observations(), {}),
        step=lambda _: (
            observations(),
            torch.zeros(2),
            torch.tensor([True, False]),
            torch.zeros(2, dtype=torch.bool),
            {},
        ),
    )
    wrapper = SimpleNamespace(
        unwrapped=env,
        env=env,
        num_envs=2,
        sim_device="cpu",
        observation_processors={},
        fast_variant=True,
    )
    for name, method in wrapped_methods.items():
        setattr(wrapper, name, MethodType(method, wrapper))
    wrapper._process_spaces()
    initial = wrapper.reset()
    wrapper.step_async(np.zeros((2, 2), dtype=np.float32))
    after, _, _, infos = wrapper.step_wait()
    for observation in (initial, after):
        for key, expected in values.items():
            if "image" in key:
                assert observation[key].dtype == np.uint8
                np.testing.assert_array_equal(observation[key] / 255, expected.numpy())
            else:
                np.testing.assert_array_equal(observation[key], expected.numpy())
    # Test transport of Lab's existing terminal slot, not a fix to its reset semantics.
    for key in values:
        np.testing.assert_array_equal(infos[0]["terminal_observation"][key], after[key][0])
        torch.testing.assert_close(values[key], before[key], rtol=0, atol=0)
