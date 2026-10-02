"""Bounded fixture training and matched, fresh-process evaluation videos.

Each train/eval invocation starts its own simulator. W&B is optional via config.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import asdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True, help="Generated recipe from config.yaml")
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--mode", choices=["train", "start", "end"], required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from rc_car.experiments import load_config, resolve_config, start_tracking, log_video
from rc_car.provenance import verify_experiment_identity
from rc_car.observation_contract import (
    BEHAVIOR_VERSION, POLICY_IMAGE_ENCODING, policy_image_contract,
    require_policy_image_encoding,
)
config = resolve_config(load_config(args.config))
if not args.enable_cameras:
    parser.error("--enable_cameras is required")
if args.mode == "train" and (args.run_dir / "manifest.json").exists():
    parser.error("Training run already exists; choose a fresh --run-dir")
if config["num_envs"] < 2 or config["total_timesteps"] < 1 or config["video_stride"] < 1:
    parser.error("Invalid experiment dimensions")
# Fail before starting Kit if the copied tree no longer matches its source identity.
# This covers listed source files only, not assets or installed runtime packages.
repo_root = Path(__file__).resolve().parents[1]
run_identity = verify_experiment_identity(repo_root, args.run_dir, args.mode, config)
if args.mode != "train":
    require_policy_image_encoding(json.loads((args.run_dir / "manifest.json").read_text()).get("policy_image_encoding"))
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import car_training.tasks  # noqa: F401
from car_training.tasks.manager_based.car_training.agents.feature_extractor import CustomMultiInputExtractor
from fixture_config import configure_fixture
from rc_car.telemetry import RewardRollup, count_termination_causes
from rc_car.lifecycle import best_effort_cleanup, finish_resources


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def event(name, **fields):
    row = {"schema_version": 1, "run_id": args.run_dir.name,
           "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "phase": args.mode, "event": name, **fields}
    with (args.run_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row), flush=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reset_counts(env):
    manager = env.unwrapped.termination_manager
    # get_term retains the last episode cause; mask it by current dones.
    # Pinned Lab 2.3 keeps only the last configured matching term per agent.
    # These are reported causes, not a reconstruction of every physical cause.
    return count_termination_causes(
        {name: manager.get_term(name) for name in manager.active_terms}, manager.dones
    )


class Metrics(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.wall_start = time.monotonic()
        self.last_checkpoint = 0
        self.terminations = {k: 0 for k in env.unwrapped.termination_manager.active_terms}
        self.rewards = []
        self.episode_returns = np.zeros(config["num_envs"])
        self.episode_lengths = np.zeros(config["num_envs"], dtype=int)
        self.completed_returns = []
        self.completed_lengths = []
        self.video_writer = None
        self.video_frames = 0
        self.next_video = 0
        self.reward_rollup = RewardRollup(
            env.unwrapped.reward_manager.active_terms, config["num_envs"]
        )

    def close_video(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            if tracking is not None:
                log_video(tracking, self.video_path, timesteps=self.num_timesteps)

    def _on_training_end(self):
        finish_resources([
            ("video close", self.close_video),
            ("partial rollout", self.flush_partial_rollout),
        ])

    def flush_partial_rollout(self, reason="training_end"):
        # A time budget can stop collection part-way through a rollout. Preserve
        # those measurements, but do not claim that PPO optimized those samples.
        if self.reward_rollup.steps:
            event("partial_rollout", agent_timesteps=self.num_timesteps,
                  reward_terms_rollout=self.reward_rollup.snapshot(), optimized=False,
                  reason=reason)
            self.reward_rollup.clear()

    def _on_step(self):
        elapsed = time.monotonic() - self.wall_start
        if config.get("training_video_interval_s"):
            if self.video_writer is None and elapsed >= self.next_video:
                import imageio.v2 as imageio
                self.video_path = args.run_dir / f"train_{self.num_timesteps:09d}.mp4"
                self.video_writer = imageio.get_writer(str(self.video_path), fps=1 / self.env.unwrapped.step_dt / config["video_stride"], codec="libx264")
                self.video_frames = 0
                self.next_video = elapsed + config["training_video_interval_s"]
            if self.video_writer is not None and self.n_calls % config["video_stride"] == 0:
                rgb = self.env.unwrapped.scene["overhead_camera"].data.output["rgb"][0].cpu().numpy()
                self.video_writer.append_data(rgb)
                self.video_frames += 1
                if self.video_frames >= config.get("training_video_frames", 90):
                    self.close_video()
        rewards = np.asarray(self.locals["rewards"])
        if not np.isfinite(rewards).all():
            raise RuntimeError("Non-finite training reward")
        self.reward_rollup.add(
            self.env.unwrapped.reward_manager._step_reward, self.env.unwrapped.step_dt
        )
        self.episode_returns += rewards
        self.episode_lengths += 1
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.completed_returns.append(float(self.episode_returns[i]))
                self.completed_lengths.append(int(self.episode_lengths[i]))
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        for key, count in reset_counts(self.env).items():
            self.terminations[key] += count
        self.rewards.append(float(rewards.mean()))
        return elapsed < config.get("max_training_seconds", float("inf"))

    def _on_rollout_end(self):
        # Termination causes may overlap on the same transition; completed
        # episodes are counted independently from the done vector above.
        elapsed = time.monotonic() - self.wall_start
        manager = self.env.unwrapped.reward_manager
        terms = {name: float(manager._step_reward[:, i].mean().item()) for i, name in enumerate(manager.active_terms)}
        reward_summary = self.reward_rollup.snapshot()
        for name, value in reward_summary["weighted_mean_before_dt"].items():
            self.logger.record(f"task/reward_mean_before_dt/{name}", value)
        for name, value in reward_summary["contribution_mean_per_agent_step"].items():
            self.logger.record(f"task/reward_contribution_mean/{name}", value)
        for name, count in self.terminations.items():
            self.logger.record(f"task/{name}_total", count)
        self.logger.record("task/mean_reward", float(np.mean(self.rewards)))
        self.logger.record("task/completed_episodes", len(self.completed_lengths))
        if tracking is not None:
            tracking.log({"training/wall_seconds": elapsed, "training/agent_timesteps": self.num_timesteps,
                          "training/mean_reward": float(np.mean(self.rewards)),
                          "training/completed_episodes": len(self.completed_lengths),
                          **{f"terminations/{k}": v for k, v in self.terminations.items()},
                          **{k: float(v) for k, v in self.logger.name_to_value.items() if np.isscalar(v)}})
        if self.completed_lengths:
            self.logger.record("task/mean_episode_length_steps", float(np.mean(self.completed_lengths)))
            self.logger.record("task/mean_episode_return", float(np.mean(self.completed_returns)))
        event("rollout", agent_timesteps=self.num_timesteps,
              wall_seconds=elapsed, aggregate_agent_steps_per_second=self.num_timesteps / elapsed,
              mean_reward=float(np.mean(self.rewards)), reward_terms_last_step_before_dt=terms,
              reward_terms_rollout=reward_summary,
              terminations_total=self.terminations, completed_episodes=len(self.completed_lengths))
        self.rewards.clear()
        self.reward_rollup.clear()
        # Checkpoint after the preceding completed PPO updates, at this rollout boundary.
        if self.num_timesteps - self.last_checkpoint >= config["checkpoint_interval"]:
            path = args.run_dir / f"checkpoint_{self.num_timesteps}"
            self.model.save(path)
            self.last_checkpoint = self.num_timesteps
            event("checkpoint", agent_timesteps=self.num_timesteps, path=path.name + ".zip",
                  note="rollout boundary, before this rollout optimization")


def train(env):
    dump_yaml(str(args.run_dir / "resolved_env.yaml"), env.unwrapped.cfg)
    asset_root = Path(os.environ["CAR_TRAINING_ASSET_ROOT"])
    manifest = {
        "schema_version": 1, "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "gpu": torch.cuda.get_device_name(),
        "control_dt_seconds": env.unwrapped.step_dt,
        "behavior_version": BEHAVIOR_VERSION,
        "policy_image_encoding": POLICY_IMAGE_ENCODING,
        "policy_image_contract": policy_image_contract(),
        "behavior_changes": [
            "robot_vel no longer mutates the simulator velocity tensor",
            "Policy image terms scale binary helper pixels by 255 before SB3 uint8 conversion and normalization; CNN white pixels are now 1.0 instead of 1/255, while black pixels remain 0",
        ] + (
            [
                "Basic rewards use static-obstacle data, skipping the obstacle gate when disabled, instead of shared-agent history",
                "Basic steering reward accepts the configured multi_obstacle_cfgs argument",
                "Basic track observations and off-track checks use local arena coordinates and dimensions",
            ] if config["task"] == "Basic-Car-Training-v0" else []
        ),
        "termination_telemetry_note": "Isaac Lab 2.3 retains the last configured matching cause per done agent, not all simultaneous causes",
        "lock_sha256": sha256(Path("uv.lock")),
        "assets": {str(p.relative_to(asset_root)): sha256(p) for p in sorted(asset_root.rglob("*.usd"))},
        "source": run_identity.source,
        "isaaclab_runtime": asdict(run_identity.isaaclab_runtime),
    }
    write_json(args.run_dir / "manifest.json", manifest)
    shutil.copy2(args.config, args.run_dir / "requested_config.json")
    write_json(args.run_dir / "config.json", config)
    shutil.copy2("uv.lock", args.run_dir / "uv.lock")
    (args.run_dir / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    wrapped = Sb3VecEnvWrapper(env)
    policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh}
    if config["task"] == "Multiagent-Car-Training-v0":
        policy_kwargs["features_extractor_class"] = CustomMultiInputExtractor
    model = PPO("MultiInputPolicy", wrapped, seed=config["train_seed"], device=args.device,
                policy_kwargs=policy_kwargs,
                verbose=1, **config["ppo"])
    model.rc_car_image_encoding = POLICY_IMAGE_ENCODING
    model.set_logger(configure(str(args.run_dir / "metrics"), ["stdout", "csv", "tensorboard"]))
    model.save(args.run_dir / "initial")
    event("training_started", target_agent_timesteps=config["total_timesteps"])
    callback = Metrics(env)
    start = time.monotonic()
    try:
        model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    except BaseException as error:
        # A media/upload error must not prevent the checkpoint attempt or mask
        # the original failure. SB3 does not call on_training_end on exceptions.
        best_effort_cleanup(error, [
            ("interrupted checkpoint", lambda: model.save(args.run_dir / "interrupted")),
            ("partial rollout", lambda: callback.flush_partial_rollout(reason="exception")),
            ("video close", callback.close_video),
        ])
        raise
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    model.save(args.run_dir / "final")
    summary = {"agent_timesteps": model.num_timesteps, "wall_seconds": time.monotonic() - start,
               "completed_episodes": len(callback.completed_lengths),
               "terminations": callback.terminations, "ppo_updates": model._n_updates,
               "ppo_update_counter_note": "SB3 PPO _n_updates counts optimization epochs, not minibatch optimizer steps",
               "initial_checkpoint_sha256": sha256(args.run_dir / "initial.zip"),
               "final_checkpoint_sha256": sha256(args.run_dir / "final.zip")}
    write_json(args.run_dir / "training_summary.json", summary)
    event("training_finished", **summary)
    if tracking is not None:
        tracking.summary.update(summary)
        import wandb
        artifact = wandb.Artifact(args.run_dir.name + "-checkpoints", type="model")
        for name in ("initial.zip", "final.zip", "manifest.json", "config.json", "resolved_env.yaml"):
            artifact.add_file(str(args.run_dir / name))
        tracking.log_artifact(artifact)


def evaluate(env):
    import imageio.v2 as imageio
    wrapped = Sb3VecEnvWrapper(env)
    checkpoint = args.run_dir / ("initial.zip" if args.mode == "start" else "final.zip")
    model = PPO.load(checkpoint, device=args.device)
    require_policy_image_encoding(getattr(model, "rc_car_image_encoding", None),
                                  normalize_images=model.policy.normalize_images)
    wrapped.seed(config["eval_seed"])
    obs = wrapped.reset()
    robot = env.unwrapped.scene["robot"]
    initial_state = {"root_state": robot.data.root_state_w.cpu().tolist(),
                     "joint_pos": robot.data.joint_pos.cpu().tolist(),
                     "joint_vel": robot.data.joint_vel.cpu().tolist()}
    write_json(args.run_dir / f"{args.mode}_initial_state.json", initial_state)
    writer = imageio.get_writer(str(args.run_dir / f"{args.mode}.mp4"),
                               fps=1 / env.unwrapped.step_dt / config["video_stride"],
                               codec="libx264", quality=8, macro_block_size=16)
    returns = np.zeros(config["num_envs"])
    terminations = {k: 0 for k in env.unwrapped.termination_manager.active_terms}
    distances = np.zeros(config["num_envs"])
    previous = robot.data.root_pos_w[:, :2].clone()
    origins = env.unwrapped.scene.env_origins[:, :2]
    angular_progress = np.zeros(config["num_envs"])
    dones_count = 0
    event("evaluation_started", checkpoint=checkpoint.name, seed=config["eval_seed"], deterministic=True)
    try:
        with (args.run_dir / f"{args.mode}_trajectory.jsonl").open("w") as trajectory:
            for step in range(config["eval_steps"]):
                actions, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = wrapped.step(actions)
                if not np.isfinite(rewards).all():
                    raise RuntimeError("Non-finite evaluation rewards")
                returns += rewards
                positions = robot.data.root_pos_w[:, :2].clone()
                delta = torch.linalg.vector_norm(positions - previous, dim=1).cpu().numpy()
                distances += np.where(dones, 0, delta)
                before_angle = torch.atan2(previous[:, 1] - origins[:, 1], previous[:, 0] - origins[:, 0])
                after_angle = torch.atan2(positions[:, 1] - origins[:, 1], positions[:, 0] - origins[:, 0])
                angle_delta = torch.atan2(torch.sin(after_angle - before_angle), torch.cos(after_angle - before_angle)).cpu().numpy()
                angular_progress += np.where(dones, 0, angle_delta)
                previous = positions
                dones_count += int(dones.sum())
                for key, count in reset_counts(env).items():
                    terminations[key] += count
                row = {"control_step": step, "sim_time_s": (step + 1) * env.unwrapped.step_dt,
                       "positions_xy": positions.cpu().tolist(), "actions": actions.tolist(),
                       "rewards": rewards.tolist(), "dones": dones.tolist()}
                trajectory.write(json.dumps(row, allow_nan=False) + "\n")
                if step % config["video_stride"] == 0:
                    rgb = env.unwrapped.scene["overhead_camera"].data.output["rgb"][0].cpu().numpy()
                    canvas = Image.new("RGB", (512, 608), "#141820")
                    canvas.paste(Image.fromarray(rgb), (0, 48))
                    draw = ImageDraw.Draw(canvas)
                    actual_steps = json.loads((args.run_dir / "training_summary.json").read_text())["agent_timesteps"]
                    label = "START - untrained policy" if args.mode == "start" else f"END - {actual_steps:,} agent steps"
                    draw.text((12, 8), label, fill="white")
                    draw.text((12, 25), f"Seed {config['eval_seed']} | {config['num_envs']} envs | generated fixture", fill="#bdc7d7")
                    draw.text((12, 569), f"Sim time {row['sim_time_s']:.2f}s | resets {dones_count} | distance {distances.sum():.2f}m", fill="white")
                    draw.text((12, 586), "Real-time playback; teleports are episode resets", fill="#bdc7d7")
                    writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    # A done transition may have more than one cause (e.g. collision + off-track).
    summary = {"phase": args.mode, "seed": config["eval_seed"], "deterministic": True,
               "control_steps": config["eval_steps"], "sim_seconds": config["eval_steps"] * env.unwrapped.step_dt,
               "returns_per_car": returns.tolist(), "mean_return": float(returns.mean()),
               "resets": dones_count, "terminations": terminations,
               "nonreset_distance_per_car_m": distances.tolist(), "nonreset_distance_total_m": float(distances.sum()),
               "fixture_signed_revolutions_per_car": (angular_progress / (2 * np.pi)).tolist(),
               "route_metric_note": "Net angular progress around generated circle; resets excluded; not uninterrupted completed laps",
               "checkpoint_sha256": sha256(checkpoint)}
    write_json(args.run_dir / f"{args.mode}_evaluation.json", summary)
    event("evaluation_finished", **summary)


def main():
    global tracking
    torch.set_num_threads(config["torch_threads"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(config["task"], device=args.device, num_envs=config["num_envs"])
    cfg.seed = config["train_seed"] if args.mode == "train" else config["eval_seed"]
    configure_fixture(cfg, config["task"])
    kwargs = {} if config["task"] == "Multiagent-Car-Training-v0" else {"enable_obstacles": config.get("enable_obstacles", True)}
    env = gym.make(config["task"], cfg=cfg, task_name=config["task"], **kwargs)
    try:
        if args.mode == "train":
            if config.get("wandb"):
                tracking = start_tracking(config, args.run_dir)
            train(env)
        else:
            evaluate(env)
    finally:
        original_error = sys.exc_info()[1]
        cleanup = []
        if tracking is not None:
            cleanup.append(("tracking finish", lambda: tracking.finish(
                exit_code=1 if original_error is not None else 0
            )))
        cleanup.append(("environment close", env.close))
        finish_resources(cleanup, original_error)


tracking = None
try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
