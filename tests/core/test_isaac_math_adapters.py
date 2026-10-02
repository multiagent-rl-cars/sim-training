"""Exercise actual adapter function bodies without starting the Isaac application.

Importing the legacy package eagerly imports Isaac. Extracting only these narrow
function bodies lets the CPU suite catch adapter wiring/state-mutation mistakes.
This does not test Isaac imports, manager scheduling, or simulator integration;
those still require the GPU suite. Core imports are taken from the actual module.
"""

import __future__

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

MDP = (
    Path(__file__).resolve().parents[2]
    / "source/car_training/car_training/tasks/manager_based/car_training/mdp"
)


class QuietLogger:
    def debug_lazy(self, *args):
        pass

    def debug(self, *args):
        pass

    def warning(self, *args):
        pass


def adapter_functions(filename, names, *, class_name=None, **extra_globals):
    """Compile selected source functions with their real rc_car imports."""
    tree = ast.parse((MDP / filename).read_text())
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rc_car.")
    ]
    body = tree.body
    if class_name:
        body = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    isolated = ast.Module(body=[*imports, *functions], type_ignores=[])
    namespace = {
        "torch": torch,
        "get_logger": lambda _: QuietLogger(),
        **extra_globals,
    }
    exec(
        compile(isolated, str(MDP / filename), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


def test_repeated_velocity_reads_do_not_change_physics_or_next_actor_input():
    functions = adapter_functions(
        "observation.py", {"robot_vel", "aligned_robot_vel", "robot_heading"}
    )
    velocity = torch.tensor([[0.8, 0.4, 0.1], [-0.2, 0.5, 0.0]])
    original = velocity.clone()
    data = SimpleNamespace(root_com_lin_vel_w=velocity, heading_w=torch.zeros((2,)))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, unwrapped=SimpleNamespace())
    cfg = SimpleNamespace(name="robot")

    before = functions["aligned_robot_vel"](env, cfg, normalize=True)
    first = functions["robot_vel"](env, cfg)
    second = functions["robot_vel"](env, cfg)
    after = functions["aligned_robot_vel"](env, cfg, normalize=True)

    torch.testing.assert_close(first, torch.tensor([[0.8, -0.4], [-0.2, -0.5]]))
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(velocity, original, rtol=0, atol=0)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(functions["robot_heading"](env, cfg), torch.tensor([[1.0, 0.0]] * 2))


def test_action_adapter_preserves_targets_and_action_buffer():
    functions = adapter_functions(
        "actions.py", {"process_actions"}, class_name="ThrottleSteerAckermannAction"
    )
    driver = SimpleNamespace(
        min_throttle=10.0,
        max_throttle=30.0,
        max_steer=0.6,
        logger=QuietLogger(),
        action_buffer=torch.zeros((2, 20, 2)),
        buffer_count=torch.zeros((2,), dtype=torch.long),
        buffer_size=20,
    )
    actions = torch.tensor([[0.0, 0.0], [100.0, -100.0]])
    functions["process_actions"](driver, actions)
    expected = torch.tensor([[20.0, 0.0], [30.0, -0.6]])
    torch.testing.assert_close(driver._processed_actions, expected)
    torch.testing.assert_close(driver.action_buffer[:, -1], expected)
    torch.testing.assert_close(driver.buffer_count, torch.ones((2,), dtype=torch.long))
    torch.testing.assert_close(actions, torch.tensor([[0.0, 0.0], [100.0, -100.0]]))


def test_reward_adapters_preserve_gate_and_read_obstacles_once_per_call():
    reads = []

    def obstacles(env, cfg):
        reads.append(True)
        return torch.tensor([[[[1.0]]], [[[0.0]]]])

    functions = adapter_functions(
        "rewards.py",
        {"_reward_obstacle_present", "distance_from_center_reward", "turning_penalty"},
        MAX_DISTANCE_FROM_LINE=40,
        robot_pos=lambda *args, **kwargs: torch.tensor([[20.0, 0.0], [80.0, 0.0]]),
        find_closest_point=lambda *args: torch.zeros((2, 2)),
        aligned_other_agents_image=obstacles,
    )
    env = SimpleNamespace(
        track_points_tensor=torch.zeros((2, 1, 2)),
        track_points_lengths=torch.ones((2,), dtype=torch.long),
        action_manager=SimpleNamespace(
            _terms={
                "drive": SimpleNamespace(
                    _processed_actions=torch.tensor([[20.0, -0.3], [20.0, 0.6]]), max_steer=0.6
                )
            }
        ),
    )
    torch.testing.assert_close(
        functions["distance_from_center_reward"](env, None), torch.tensor([0.5, 0.0])
    )
    assert len(reads) == 1
    torch.testing.assert_close(functions["turning_penalty"](env, None), torch.tensor([-0.5, 0.0]))
    assert len(reads) == 2


def configured_terms(filename, class_path, namespace):
    """Evaluate real term declarations, replacing only Isaac config containers."""
    tree = ast.parse((MDP.parent / filename).read_text())
    body = tree.body
    for name in class_path:
        body = next(
            node.body for node in body if isinstance(node, ast.ClassDef) and node.name == name
        )
    scope = {
        **namespace,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "RewTerm": lambda **kwargs: SimpleNamespace(**kwargs),
        "ObsTerm": lambda **kwargs: SimpleNamespace(**kwargs),
        "DoneTerm": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    terms = {}
    for node in body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id not in (
            "RewTerm",
            "ObsTerm",
            "DoneTerm",
        ):
            continue
        assert isinstance(node.targets[0], ast.Name)
        terms[node.targets[0].id] = eval(
            compile(ast.Expression(node.value), str(MDP.parent / filename), "eval"), scope
        )
    return terms


@pytest.mark.parametrize("filename", ["car_training_env_cfg.py", "multiagent_training_env_cfg.py"])
def test_active_driving_reward_configuration_matches_actual_signatures(filename):
    functions = adapter_functions(
        "rewards.py",
        {
            "_reward_obstacle_present",
            "distance_from_center_reward",
            "turning_penalty",
            "velocity_reward",
            "remains_alive_reward",
            "multiagent_collision_penalty",
        },
    )
    terms = configured_terms(filename, ["RewardsCfg"], functions)
    assert len(terms) == (4 if filename == "car_training_env_cfg.py" else 5)
    for term in terms.values():
        inspect.signature(term.func).bind(object(), **getattr(term, "params", {}))


@pytest.mark.parametrize("enabled", [False, True])
def test_basic_reward_uses_static_obstacles_and_never_shared_agent_history(enabled):
    obstacle_reads = []

    def static_obstacles(env, cfg, obstacle_cfgs):
        obstacle_reads.append([entry.name for entry in obstacle_cfgs])
        return torch.tensor([[[[1.0]]], [[[0.0]]]])

    def unexpected_shared_history(*args, **kwargs):
        raise AssertionError("Basic reward must not read multi-agent history")

    functions = adapter_functions(
        "rewards.py",
        {
            "_reward_obstacle_present",
            "distance_from_center_reward",
            "turning_penalty",
            "velocity_reward",
            "remains_alive_reward",
        },
        MAX_DISTANCE_FROM_LINE=40,
        robot_pos=lambda *args, **kwargs: torch.tensor([[20.0, 0.0], [80.0, 0.0]]),
        find_closest_point=lambda *args: torch.zeros((2, 2)),
        aligned_obstacle_image=static_obstacles,
        aligned_other_agents_image=unexpected_shared_history,
    )
    terms = configured_terms("car_training_env_cfg.py", ["RewardsCfg"], functions)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        unwrapped=SimpleNamespace(enable_obstacles=enabled),
        track_points_tensor=torch.zeros((2, 1, 2)),
        track_points_lengths=torch.ones((2,), dtype=torch.long),
        action_manager=SimpleNamespace(
            _terms={
                "drive": SimpleNamespace(
                    _processed_actions=torch.tensor([[20.0, -0.3], [20.0, 0.6]]), max_steer=0.6
                )
            }
        ),
    )
    proximity = terms["distance_from_center_rew_term"]
    steering = terms["steer_penalty_term"]
    torch.testing.assert_close(
        proximity.func(env, **proximity.params),
        torch.tensor([0.5, 0.0 if enabled else -1.0]),
    )
    torch.testing.assert_close(
        steering.func(env, **steering.params),
        torch.tensor([-0.5, 0.0 if enabled else -1.0]),
    )
    assert obstacle_reads == [[f"obstacle_{index}" for index in range(4)]] * (2 if enabled else 0)
    assert proximity.weight == 3.0
    assert steering.weight == 0.5
    assert not hasattr(env, "_obstacle_image_stack")


def test_basic_track_crop_and_termination_use_each_arena_origin_and_dimensions():
    functions = adapter_functions(
        "observation.py",
        {
            "robot_pos",
            "robot_heading",
            "aligned_track_neighborhood_image",
            "aligned_obstacle_image",
            "aligned_robot_vel",
        },
        F=F,
        CROPPED_IMAGE_SIZE=48,
    )
    termination = adapter_functions(
        "termination.py",
        {"off_track_termination", "inside_obstacle_termination"},
        aligned_track_neighborhood_image=functions["aligned_track_neighborhood_image"],
    )
    observed = configured_terms(
        "car_training_env_cfg.py", ["ObservationsCfg", "PolicyCfg"], functions
    )["track_image"]
    ended = configured_terms("car_training_env_cfg.py", ["TerminationsCfg"], termination)[
        "out_of_bounds"
    ]
    assert observed.params["global_frame"] is False
    assert ended.params["global_frame"] is False
    origins = torch.tensor([[15.0, 15.0, 0.0], [-15.0, -15.0, 0.0]])
    root_positions = origins + torch.tensor([1.0, 2.0, 0.0])
    data = SimpleNamespace(root_link_pos_w=root_positions, heading_w=torch.zeros(2))

    class Scene(dict):
        env_origins = origins

    image = torch.ones((2, 256, 256, 3), dtype=torch.float32)
    # Local (1m, 2m) in each 10m arena is image (153.6px, 76.8px).
    image[:, 73:81, 150:158] = 0
    env = SimpleNamespace(
        scene=Scene(robot=SimpleNamespace(data=data)),
        cfg=SimpleNamespace(
            scene=SimpleNamespace(
                ground_plane=SimpleNamespace(
                    ground_plane_params=SimpleNamespace(
                        each_track_size=(10.0, 10.0), multi_agent_track_size=(40.0, 40.0)
                    )
                )
            )
        ),
        unwrapped=SimpleNamespace(
            device="cpu",
            num_envs=2,
            static_track_image=image,
            get_camera_dimensions=lambda: (256, 256),
        ),
    )
    local_pixels = functions["robot_pos"](
        env, observed.params["asset_cfg"], do_convert_coords=True, global_frame=False
    )
    torch.testing.assert_close(local_pixels, torch.tensor([[153.6, 76.8]] * 2))
    crops = observed.func(env, **observed.params)
    torch.testing.assert_close(crops[0], crops[1], rtol=0, atol=0)
    assert bool((crops[:, :, 12:36, 12:36] == 0).any())
    assert not bool(ended.func(env, **ended.params).any())
    # The preserved shared-world mapping would read entirely different pixels.
    old_global = functions["aligned_track_neighborhood_image"](env, observed.params["asset_cfg"])
    assert bool((old_global == 1).all())
    assert bool(termination["off_track_termination"](env, ended.params["asset_cfg"]).all())


def test_multiagent_track_and_termination_keep_global_defaults():
    functions = adapter_functions(
        "observation.py",
        {"aligned_track_neighborhood_image", "aligned_other_agents_image", "aligned_robot_vel"},
    )
    termination = adapter_functions(
        "termination.py", {"off_track_termination", "multiagent_collision_termination"}
    )
    track = configured_terms(
        "multiagent_training_env_cfg.py", ["ObservationsCfg", "PolicyCfg"], functions
    )["track_image"]
    ending = configured_terms("multiagent_training_env_cfg.py", ["TerminationsCfg"], termination)[
        "out_of_bounds"
    ]
    for term in (track, ending):
        bound = inspect.signature(term.func).bind(object(), **term.params)
        bound.apply_defaults()
        assert bound.arguments["global_frame"] is True
