# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import numpy as np
import cv2
import logging
import torch.nn.functional as F
from rc_car.core.rewards import legacy_line_proximity, legacy_steering_penalty

from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_closest_point
from car_training.tasks.manager_based.car_training.mdp.observation import (
    robot_pos,
    aligned_robot_vel,
    robot_vel,
    aligned_obstacle_image,
    aligned_other_agents_image
)
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import (
    GpuSafeLogger,
)

_LOGGER: Optional[GpuSafeLogger] = None

MAX_DISTANCE_FROM_LINE = 40



def get_logger(env: ManagerBasedRLEnv) -> GpuSafeLogger:
    """
    Get a child logger of the project logger for the car_training.mdp.observation module.

    :param env: The environment instance containing the base logger.
    :type env: ManagerBasedRLEnv

    :return: Child logger for the observation module.
    :rtype: GpuSafeLogger
    """
    global _LOGGER  # creating a cache to avoid repeated calls
    if _LOGGER is None:
        base = env.unwrapped.logger
        # car_training.mdp.observation
        short_name = ".".join(__name__.split(".")[-2:])
        _LOGGER = base.getChild(short_name)
    return _LOGGER

def _reward_obstacle_present(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    multi_obstacle_cfgs: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    """Choose the task's existing obstacle observation without mixing arenas.

    Basic tasks explicitly supply static obstacle configurations. Shared-track
    tasks omit them and retain their historical other-agent/history observation.
    This routing fix does not change the shared-track history or ego semantics.
    """
    if multi_obstacle_cfgs is not None:
        if not env.unwrapped.enable_obstacles:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        obstacle_imgs = aligned_obstacle_image(env, robot_cfg, multi_obstacle_cfgs)
    else:
        obstacle_imgs = aligned_other_agents_image(env, robot_cfg)
    return torch.any(obstacle_imgs < 0.12, dim=tuple(range(1, obstacle_imgs.ndim)))


def distance_from_center_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    global_frame: bool = False,
    multi_obstacle_cfgs: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    """
    Reward for being close to the center of the track.

    :param env: The environment instance containing the track and robot information.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot to extract its position.
    :type robot_cfg: SceneEntityCfg

    :return: Distance from center reward.
    :rtype: torch.Tensor
    """
    car_centers = robot_pos(env, robot_cfg, do_convert_coords=True, global_frame=global_frame)
    track_points = env.track_points_tensor  # (num_envs, num_track_points, 2)
    track_points_lengths = env.track_points_lengths  # (num_envs,)

    closest_track_points = find_closest_point(car_centers, track_points, track_points_lengths)  # (num_envs, 2)
    distances = torch.norm(car_centers - closest_track_points, dim=1)  #
    has_obstacle_pixel = _reward_obstacle_present(env, robot_cfg, multi_obstacle_cfgs)
    return legacy_line_proximity(
        distances, has_obstacle_pixel, distance_scale_px=MAX_DISTANCE_FROM_LINE
    )

def velocity_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for maintaining a high speed.

    :param env: The environment instance containing the robot information.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot to extract its velocity.
    :type robot_cfg: SceneEntityCfg

    :return: Velocity reward.
    :rtype: torch.Tensor
    """
    car_vels = aligned_robot_vel(env, robot_cfg, normalize=True)
    car_speeds = torch.norm(car_vels, dim=1)
    normalized_speeds = car_speeds
    return normalized_speeds

def remains_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns a constant reward for remaining alive.

    :return: Constant reward value.
    :rtype: torch.Tensor
    """
    return torch.ones((env.num_envs,), device=env.device)

def steering_jerk_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for smooth steering by penalizing large changes in steering angle.

    :return: Smooth steering reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    logger.debug("Calculating steering jerk penalty...")

    drive = env.action_manager._terms["drive"]

    # Need to create a buffer to average second derivitive over multiple points
    # Convolution kernal is [1,-2,1]
    def d2_filter(buffer):
        # Kernel: [-1, 2, -1] 
        kernel = torch.tensor([1.0, -2.0, 1.0], device=buffer.device)
        kernel = kernel.view(1, 1, -1)  
        buffer_input = buffer.unsqueeze(1)
        result = F.conv1d(buffer_input, kernel, padding=0)  # (num_envs, 1, buffer_size-2)
        return result.squeeze(1)  # (num_envs, buffer_size-2) 

    steer_buffer = drive.action_buffer[:, :, 1]
    
    min_buffer_size = 3
    valid_envs = drive.buffer_count >= min_buffer_size
    
    if not valid_envs.any():
        # Not enough history yet, return zero penalty
        logger.debug("Insufficient buffer history for steering jerk penalty.")
        return torch.zeros((env.num_envs,), device=env.device)
    # dt is optional; keep if you want rate-invariance
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    #Updated j with filter 
    j = d2_filter(steer_buffer)/(dt*dt)

    mask = torch.zeros_like(j, dtype=torch.bool)
    for env_idx in range(env.num_envs):
        count = drive.buffer_count[env_idx].item()
        if count >= 3:
            num_valid = count - 2
            mask[env_idx, -num_valid:] = True

    j_sqr_mask = (j ** 2) * mask.float()
    num_valid = mask.sum(dim=1).float().clamp(min=1.0)
    j_rms = torch.sqrt(j_sqr_mask.sum(dim=1) / num_valid)

    penalty = -1 * (j_rms)

    logger.debug_lazy("Steering angles prev2: %s", lambda: steer_buffer)
    logger.debug_lazy("Decimation: %s", lambda: env.cfg.decimation)
    logger.debug_lazy("Sim dt: %s", lambda: env.cfg.sim.dt)
    logger.debug_lazy("Control timestep dt: %s", lambda: dt)
    logger.debug_lazy("Steering jerk: %s", lambda: j)
    logger.debug_lazy("Smooth steering reward: %s", lambda: penalty)

    return penalty


def turning_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    multi_obstacle_cfgs: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    """
    Penalty for large steering angles, encouraging the reset agent to take smoother turns.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param weight: The weight of the turning penalty.
    :type weight: float

    :return: Turning penalty for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating turning penalty...")

    drive = env.action_manager._terms["drive"]
    steer = drive._processed_actions[:, 1]  # (N,)

    has_obstacle_pixel = _reward_obstacle_present(env, robot_cfg, multi_obstacle_cfgs)

    penalty = legacy_steering_penalty(
        steer, has_obstacle_pixel, max_steering_rad=drive.max_steer
    )

    logger.debug_lazy("Steering angles for turning penalty: %s", lambda: steer)
    logger.debug_lazy("Turning penalty: %s", lambda: penalty)

    return penalty

def collision_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, obstacle_cfg: SceneEntityCfg, collision_threshold: float = 20.0) -> torch.Tensor:
    """
    Collision penalty function that currently returns zero for all environments.

    :return: Collision penalty (currently zero).
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    logger.debug("Calculating collision penalty...")

    # Get positions of robot and obstacle
    robot_positions = robot_pos(env, asset_cfg, do_convert_coords=True)  # (N, 2)
    obstacle_positions = robot_pos(env, obstacle_cfg, do_convert_coords=True)  # (N, 2)

    # Compute distance between robot and obstacle
    distances = torch.norm(robot_positions - obstacle_positions, dim=1)  # (N,)

    # Penalize distcances below the collision threshold + 5 pixels to reduce sparsity of the reward signal

    # penalty = torch.zeros_like(distances)
    # penalty_mask = distances < (collision_threshold + 5.0)
    # penalty[penalty_mask] = 1.0 - (distances[penalty_mask] / (collision_threshold + 5.0))

    # Old binary penalty (works still, other one untested)
    collision_flags = distances < collision_threshold
    penalty = torch.where(collision_flags, torch.tensor(1.0, device=env.device), torch.tensor(0.0, device=env.device))

    logger.debug_lazy("Collision penalty: %s", lambda: penalty)

    penalty = -1 * penalty
    return penalty 

def multiagent_collision_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, collision_threshold: float = 20.0) -> torch.Tensor:
    """
    Variable penalty based on proximity to other agents, encouraging the agent to avoid collisions with other agents.

    :param env: The environment instance containing the robot information.
    :type env: ManagerBasedRLEnv
    :param asset_cfg: Configuration for the robot to extract its position.
    :type asset_cfg: SceneEntityCfg
    :param collision_threshold: Distance threshold for considering a collision.
    :type collision_threshold: float
    :return: Collision penalty for each agent.
    :rtype: torch.Tensor

    """
    logger = get_logger(env)

    # this includes all positions of all other agents, so we can check distance to all of them at once
    robot_positions = robot_pos(env, asset_cfg, do_convert_coords=True, global_frame=True)  # (N, 2)

    num_agents = robot_positions.shape[0]
    if num_agents < 2:
        return torch.zeros(num_agents, dtype=torch.float, device=robot_positions.device)

    # Pairwise distances between all agents: (N, N)
    pairwise_dist = torch.cdist(robot_positions, robot_positions, p=2)

    # Ignore self-distances by setting diagonal to +inf
    pairwise_dist.fill_diagonal_(float("inf"))

    # Collision if an agent is within threshold of any other agent
    collision_flags = torch.any(pairwise_dist < collision_threshold, dim=1)

    penalty = torch.where(collision_flags, 1, 0).float()

    logger.debug_lazy("Multi-agent collision penalty: %s", lambda: penalty)
    penalty = -1 * penalty
    
    return penalty


############ FOR RESET AGENT ############

def reset_agent_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Total reward for the reset agent, which can be different from the main agent's reward.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param robot_cfg: Configuration for the robot.
    :type robot_cfg: SceneEntityCfg

    :return: Total reward for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating reset agent total reward...")

    car_pos: torch.Tensor = env.unwrapped.current_observation["pos"]  # (N, 2)
    destination_pos: torch.Tensor = env.unwrapped.current_observation["destination"]  # (N, 2)
    # Distance from each car to its own destination (diagonal)
    per_env_distance = torch.norm(car_pos - destination_pos, dim=1)  # (N,)

    prev_dist = env.unwrapped.previous_distance_to_goal
    skip = env.unwrapped.skip_distance_reward_next

    distance_reward = prev_dist - per_env_distance

    # skip distance reward for envs that just reached the goal to avoid giving a large positive reward on the same step as reaching the goal
    distance_reward = torch.where(skip, torch.zeros_like(distance_reward), distance_reward)

    env.unwrapped.previous_distance_to_goal = per_env_distance.clone()

    combined_flags = env.unwrapped.reached_goal_flags | env.unwrapped.out_of_bounds_flags
    env.unwrapped.skip_distance_reward_next = combined_flags.clone()

    logger.debug_lazy("Per-env distance to destination: %s", lambda: per_env_distance)
    logger.debug_lazy("Reset agent reward: %s", lambda: distance_reward)

    return distance_reward

def combined_heading_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor: 
    """
    Combined reward for the reset agent that takes into account both distance to the destination and heading alignment.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv

    :return: Combined reward for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating combined heading and distance reward...")

    distance_reward = reset_agent_distance_reward(env)
    heading_reward = heading_alignment_reward(env)

    combined_reward = distance_reward * heading_reward

    logger.debug_lazy("Combined reward: %s", lambda: combined_reward)

    return combined_reward

def get_terminal_bonus(env: ManagerBasedRLEnv, bonus: float) -> torch.Tensor:
    """
    Bonus reward given at the end of an episode, for example when the agent reaches the goal.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param bonus: The bonus reward to give.
    :type bonus: float

    :return: Terminal bonus reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating terminal bonus...")

    reached_goal_flags = env.unwrapped.reached_goal_flags  # (N,), this is set in the termination function and stored in the env

    # Give bonus to envs that are done
    rewards = torch.where(reached_goal_flags, torch.tensor(bonus, device=env.device), torch.zeros_like(reached_goal_flags, dtype=torch.float))

    logger.debug_lazy("Reached goal flags: %s", lambda: reached_goal_flags)
    logger.debug_lazy("Terminal bonus rewards: %s", lambda: rewards)

    if reached_goal_flags.any():
        logger.debug_lazy("Some envs reached the goal! Reached goal flags: %s", lambda: reached_goal_flags)
    return rewards


def heading_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for aligning the car's heading with the direction to the goal.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv

    :return: Heading alignment reward.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating heading alignment reward...")

    # since direction to goal is in the car's local frame, we take the dot product of it with vertical
    direction_to_goal = env.unwrapped.current_observation["direction_to_goal"]  # (N, 2)
    vertical_vec = torch.tensor([0.0, -1.0], device=env.device)  # (2,) using [0, -1] because the camera vertical axis points negative
    # print(f"Heading vec: {heading_vec}, Direction to goal: {direction_to_goal}")

    # Take dot product of the two vectors
    alignment = F.cosine_similarity(direction_to_goal, vertical_vec.unsqueeze(0), dim=1)  # (N,)

    # normalize the alignment to be between 0 and 1 instead of -1 and 1
    # alignment = 0.5 * (alignment + 1.0)

    logger.debug_lazy("Heading alignment reward: %s", lambda: alignment)
    # print(f"Heading alignment reward: {alignment}")

    return alignment

def reset_agent_velocity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for the reset agent based on its velocity, encouraging it to move towards the goal.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param max_speed: The maximum speed of the agent for normalization.
    :type max_speed: float
    :param asset_cfg: The configuration of the asset for which the velocity is being calculated.
    :type asset_cfg: SceneEntityCfg

    :return: Velocity reward for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating reset agent velocity reward...")

    # Assuming speed is normalized in observation
    car_speeds = env.unwrapped.current_observation["speed"]  # (N,)

    # if inside goal, make velocity reward so that higher vel gives negative rewrad to encourage the agent to slow down once it reaches the goal
    inside_goal_flags = env.unwrapped.inside_goal_flags  # (N,), this is set in the termination function and stored in the env
    velocity_reward = torch.where(inside_goal_flags, -car_speeds, car_speeds)

    return velocity_reward

def reset_agent_turning_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalty for large steering angles, encouraging the reset agent to take smoother turns.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param weight: The weight of the turning penalty.
    :type weight: float

    :return: Turning penalty for the reset agent.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    logger.debug("Calculating turning penalty...")

    drive = env.action_manager._terms["drive"]
    steer = drive._processed_actions[:, 1]  # (N,)

    penalty = -1 * torch.abs(steer) / drive.max_steer # divide by max_steer to normalize

    logger.debug_lazy("Steering angles for turning penalty: %s", lambda: steer)
    logger.debug_lazy("Turning penalty: %s", lambda: penalty)

    return penalty
