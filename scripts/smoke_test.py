"""Bounded simulator check: inherited car task, finite observations, motion and PPO.

Use scripts/isaac.sh scripts/smoke_test.py --headless --enable_cameras --fixture.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Basic-Car-Training-v0", choices=["Basic-Car-Training-v0", "Multiagent-Car-Training-v0"])
parser.add_argument("--fixture", action="store_true", help="Use conservative physics/camera settings for generated test assets")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--train", action="store_true", help="Also run at least 64 PPO timesteps (rounded to a rollout) and save a checkpoint")
parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num_envs must be positive")
if not args.enable_cameras:
    parser.error("The inherited car task requires --enable_cameras")
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import car_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rc_car.lifecycle import finish_resources
from rc_car.observation_contract import (
    BEHAVIOR_VERSION, POLICY_IMAGE_ENCODING, policy_image_contract, policy_image_panel,
)


def check_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Non-finite simulator output"


def record_motion(env, steps, output):
    """Record the actual fixed-action motion check, including evidence on failure."""
    if steps < 1:
        raise ValueError("Motion check requires at least one step")
    import imageio.v2 as imageio
    robot = env.unwrapped.scene["robot"]
    start = robot.data.root_pos_w.clone()
    previous = start.clone()
    path_length = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    resets = 0
    max_displacement = 0.0
    frames = 0
    writer = imageio.get_writer(str(output / "smoke.mp4"), fps=1 / env.unwrapped.step_dt, codec="libx264")
    try:
        for _ in range(steps):
            with torch.no_grad():
                obs, reward, terminated, truncated, _ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
                check_finite(obs)
                check_finite(reward)
                check_finite(robot.data.root_pos_w)
                done = terminated | truncated
                delta = torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - previous[:, :2], dim=-1)
                path_length += torch.where(done, torch.zeros_like(delta), delta)
                previous = robot.data.root_pos_w.clone()
                resets += int(done.sum().item())
                max_displacement = max(max_displacement, float(torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - start[:, :2], dim=-1).max().item()))
                rgb = env.unwrapped.scene["overhead_camera"].data.output["rgb"][0].cpu().numpy()
                writer.append_data(rgb)
                frames += 1
    finally:
        finish_resources([("smoke video close", writer.close)], sys.exc_info()[1])
    return obs, {
        "max_xy_displacement_m": max_displacement,
        "resets": resets,
        "nonreset_path_length_m": path_length.tolist(),
        "videos": [{"path": "smoke.mp4", "expected_frames": frames}],
        "video_note": "Actual camera during fixed zero-action motion check, not policy training",
    }


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    if args.fixture:
        from fixture_config import configure_fixture
        configure_fixture(cfg, args.task)
    env = gym.make(args.task, cfg=cfg, task_name=args.task)
    try:
        obs, _ = env.reset(seed=42)
        check_finite(obs)
        args.output.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        rgb = env.unwrapped.scene["overhead_camera"].data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb).save(args.output / "camera_reset.png")
        # Policy terms are already float32 0/255; do not multiply by 255 again.
        policy_obs = obs["policy"]
        panel = policy_image_panel(policy_obs["track_image"][0].cpu().numpy(),
                                   policy_obs["obstacle_image"][0].cpu().numpy())
        Image.fromarray(panel).resize((panel.shape[1]*4, panel.shape[0]*4), Image.Resampling.NEAREST).save(args.output / "policy_inputs.png")
        robot = env.unwrapped.scene["robot"]
        obs, motion = record_motion(env, args.steps, args.output)
        print(f"Motion check: {motion}, joint velocities={robot.data.joint_vel.tolist()}", flush=True)
        assert min(motion["nonreset_path_length_m"]) > 0.001, "At least one vehicle did not move between resets"
        assert motion["resets"] < args.steps * args.num_envs, "Every step terminated; track/reset behavior is broken"
        args.output.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        rgb = env.unwrapped.scene["overhead_camera"].data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb).save(args.output / "camera.png")
        (args.output / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy2(Path(name), args.output / name)
        result = {"steps": args.steps, "num_envs": args.num_envs, "fixture": args.fixture,
                  **motion,
                  "behavior_version": BEHAVIOR_VERSION,
                  "policy_image_encoding": POLICY_IMAGE_ENCODING,
                  "policy_image_contract": policy_image_contract(),
                  "gpu": torch.cuda.get_device_name(), "ppo_timesteps": 0, "ppo_updates": 0, "task": args.task,
                  "ppo_update_counter_note": "SB3 PPO _n_updates counts optimization epochs, not minibatch optimizer steps"}
        result["observation_shapes"] = {k: list(v.shape) for k, v in obs["policy"].items()}
        manager = env.unwrapped.reward_manager
        result["last_reward_terms_before_dt"] = {
            name: manager._step_reward[:, i].tolist() for i, name in enumerate(manager.active_terms)
        }
        if args.train:
            from stable_baselines3 import PPO
            from isaaclab_rl.sb3 import Sb3VecEnvWrapper
            wrapped = Sb3VecEnvWrapper(env)
            policy_kwargs = {"net_arch": [32, 32]}
            if args.task == "Multiagent-Car-Training-v0":
                from car_training.tasks.manager_based.car_training.agents.feature_extractor import CustomMultiInputExtractor
                policy_kwargs["features_extractor_class"] = CustomMultiInputExtractor
            model = PPO("MultiInputPolicy", wrapped, n_steps=32, batch_size=32, n_epochs=1,
                        policy_kwargs=policy_kwargs, device=args.device, verbose=1, seed=42)
            model.rc_car_image_encoding = POLICY_IMAGE_ENCODING
            from stable_baselines3.common.logger import configure
            model.set_logger(configure(str(args.output / "ppo"), ["stdout", "csv"]))
            model.save(args.output / "initial")
            before = [p.detach().clone() for p in model.policy.parameters()]
            model.learn(total_timesteps=64)
            assert all(torch.isfinite(p).all() for p in model.policy.parameters()), "Non-finite PPO weights"
            changed = any(not torch.equal(old, new) for old, new in zip(before, model.policy.parameters()))
            assert changed, "PPO did not update policy weights"
            result["policy_weights_changed"] = changed
            assert model._n_updates > 0, "PPO did not perform an optimizer update"
            result["ppo_updates"] = model._n_updates
            from stable_baselines3.common.preprocessing import preprocess_obs
            raw = wrapped.reset()
            tensors, _ = model.policy.obs_to_tensor(raw)
            inputs = preprocess_obs(tensors, model.observation_space, normalize_images=model.policy.normalize_images)
            result["network_input_ranges"] = {k: [float(v.min()), float(v.max())] for k, v in inputs.items()}
            model.save(args.output / "fixture_ppo")
            model.save(args.output / "final")
            result["ppo_timesteps"] = model.num_timesteps
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("SMOKE TEST PASSED", json.dumps(result))
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except BaseException:
        # Kit shutdown may exit before Python prints an unwound exception.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit shutdown exits with status 0, so preserve failure for automation.
        os._exit(1)
    else:
        app.close()
