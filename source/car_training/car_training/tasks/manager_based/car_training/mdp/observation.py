from __future__ import annotations


from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import logging

import cv2
import math
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from isaaclab.assets import RigidObject
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_closest_point
from rc_car.core.geometry import (
    convert_coordinates as convert_coords,
    heading_to_image_vector,
    legacy_aligned_velocity,
    world_velocity_to_image,
)
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import GpuSafeLogger
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CROPPED_IMAGE_SIZE = 48 # Size of the cropped track neighborhood image
OBSTACLE_CROPPED_IMAGE_SIZE = 96 # Size of the cropped obstacle image
FRONT_STEER_NAMES = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]


_LOGGER: Optional[GpuSafeLogger] = None

def get_logger(env: ManagerBasedRLEnv) -> GpuSafeLogger:
    """
    Get a child logger of the project logger for the car_training.mdp.observation module.

    :param env: The environment instance containing the base logger.
    :type env: ManagerBasedRLEnv
    
    :return: Child logger for the observation module.
    :rtype: GpuSafeLogger
    """
    global _LOGGER # creating a cache to avoid repeated calls
    if _LOGGER is None:
        base = env.unwrapped.logger
        short_name = ".".join(__name__.split(".")[-2:]) # car_training.mdp.observation
        _LOGGER = base.getChild(short_name)
    return _LOGGER


def robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the (x, y) velocity of the car.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The velocity vector of the car.
    """
    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]

    # Return an independent image-frame velocity tensor. The previous slice
    # followed by an in-place sign flip mutated the simulator's cached state.
    relative_vel = world_velocity_to_image(asset.data.root_com_lin_vel_w)

    logger.debug_lazy("Car velocities: %s", lambda: relative_vel)
    return relative_vel

def robot_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, normalize: bool = False) -> torch.Tensor:
    """
    Calculate the speed of the car.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.

    Returns:
        torch.Tensor: The speed of the car.
    """
    logger = get_logger(env)

    velocity = robot_vel(env, asset_cfg)  # (N, 2)
    speed = torch.norm(velocity, dim=1)  # (N,)

    if hasattr(env.unwrapped, 'action_driver'):
        driver = env.unwrapped.action_driver
        max_throttle = driver.max_throttle
        if max_throttle != 30:
            logger.warning(f"Unexpected throttle limits for speed normalization: max_throttle={max_throttle}. Speed normalization may be incorrect.")

    if normalize:
        max_speed = 0.8  # 0.8 m/s corresponds to 30 rad/s for inside the action space
        speed = speed / max_speed

    return speed

def robot_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:

    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_heading = asset.data.heading_w

    logger.debug_lazy("Car heading: %s", lambda: robot_heading)

    # make a unit vector
    robot_heading = heading_to_image_vector(robot_heading)
    return robot_heading

def robot_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, do_convert_coords: bool = False, global_frame: bool = False) -> torch.Tensor:
    """
    Get the position of the car relative to the environment or convert to image coordinates.

    :param do_convert_coords: Whether to convert the coordinates to image space.
    :type do_convert_coords: bool
    :return: The (x, y) position of the car.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    asset: RigidObject = env.scene[asset_cfg.name]
    if global_frame:
        relative_pos = asset.data.root_link_pos_w  # Global position in world coordinates
        track_size = env.cfg.scene.ground_plane.ground_plane_params.multi_agent_track_size
    else:
        relative_pos = asset.data.root_link_pos_w - env.scene.env_origins  # Relative to environment origin
        track_size = env.cfg.scene.ground_plane.ground_plane_params.each_track_size
    logger.debug_lazy("Car world positions: %s", lambda: relative_pos)

    
    camera_width, camera_height = env.unwrapped.get_camera_dimensions()
    logger.debug(f"Track size: {track_size}, Camera width: {camera_width}, Camera height: {camera_height}")

    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, track_size, (camera_width, camera_height))
        logger.debug_lazy("Car image positions: %s", lambda: relative_pos)
    return relative_pos[:, :2]  # Return x, y position

def single_obstacle_pos(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg, do_convert_coords: bool = False) -> torch.Tensor:
    """
    Get the position of the obstacle relative to the environment or convert to image coordinates.

    :param do_convert_coords: Whether to convert the coordinates to image space.
    :type do_convert_coords: bool
    :return: The (x, y) position of the obstacle.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)

    asset: RigidObject = env.scene[obstacle_cfg.name]
    relative_pos = asset.data.root_link_pos_w - env.scene.env_origins  # Relative to environment origin
    logger.debug_lazy("Obstacle world positions: %s", lambda: relative_pos)

    track_size = env.cfg.scene.ground_plane.ground_plane_params.each_track_size
    camera_width, camera_height = env.unwrapped.get_camera_dimensions()
    logger.debug(f"Track size: {track_size}, Camera width: {camera_width}, Camera height: {camera_height}")

    if do_convert_coords:
        relative_pos = convert_coords(relative_pos, track_size, (camera_width, camera_height))
        logger.debug_lazy("Obstacle image positions: %s", lambda: relative_pos)
    return relative_pos[:, :2]  # Return x, y position

def multi_obstacles_pos(env: ManagerBasedRLEnv, obstacle_cfgs: list[SceneEntityCfg], do_convert_coords: bool = False) -> torch.Tensor:
    """
    Get the positions of multiple obstacles relative to the environment or convert to image coordinates.

    :param do_convert_coords: Whether to convert the coordinates to image space.
    :type do_convert_coords: bool
    :return: The (x, y) positions of the obstacles, shape (N, num_obstacles, 2).
    :rtype: torch.Tensor
    """
    positions = []
    for obstacle_cfg in obstacle_cfgs:
        xy_pos = single_obstacle_pos(env, obstacle_cfg, do_convert_coords=do_convert_coords)  # (N, 2)
        positions.append(xy_pos)
    
    positions = torch.stack(positions, dim=1)  # (N, num_obstacles, 2)

    return positions


def single_obstacle_heading(env: ManagerBasedRLEnv, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:

    logger = get_logger(env)
    asset: RigidObject = env.scene[obstacle_cfg.name]
    obstacle_heading = asset.data.heading_w

    logger.debug_lazy("Obstacle heading: %s", lambda: obstacle_heading)

    # make a unit vector
    obstacle_heading = heading_to_image_vector(obstacle_heading)
    return obstacle_heading

def multi_obstacles_heading(env: ManagerBasedRLEnv, obstacle_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    headings = []
    for obstacle_cfg in obstacle_cfgs:
        heading = single_obstacle_heading(env, obstacle_cfg)  # (N, 2)
        headings.append(heading)
    
    headings = torch.stack(headings, dim=1)  # (N, num_obstacles, 2)

    return headings

def aligned_track_neighborhood_image(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hide_previous_track: bool = False,
    global_frame: bool = True,
) -> torch.Tensor:
    """
    Get the track neighborhood image rotated and aligned with the car's heading.
    The car's heading will always point towards the top of the image.
    Shared-track callers retain global coordinates by default. Independent Basic
    arenas explicitly pass global_frame=False to subtract each arena's origin
    and use its local track dimensions.

    Returns:
        torch.Tensor: Aligned grayscale images of shape (N, 1, crop, crop).
    """
    logger = get_logger(env)

    # TODO: remove attribute check - fix in env setup since this is not on gpu
    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        pass
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        logger.warning("static track image not found, returning dummy gray image")
        return torch.zeros((num_envs, 1, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), device=env.unwrapped.device, dtype=torch.float32)

    device = env.unwrapped.device

    imgs = env.unwrapped.static_track_image
    N, H, W, _ = imgs.shape
    logger.debug(f"Static track image shape: number of envs={N}, H={H}, W={W}")

    # Convert to float in [0, 1] and NCHW
    if imgs.dtype != torch.float32:
        imgs = imgs.float() / 255.0
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    car_xy = robot_pos(env, asset_cfg, do_convert_coords=True, global_frame=global_frame)
    logger.debug_lazy("Car positions (image coords): %s", lambda: car_xy)

    # Normalize to [-1, 1] for grid_sample
    cx = 2.0 * car_xy[:, 0] / (W - 1) - 1.0
    cy = 2.0 * car_xy[:, 1] / (H - 1) - 1.0

    lin = torch.linspace(-1, 1, CROPPED_IMAGE_SIZE, device=device)
    dy, dx = torch.meshgrid(lin, lin, indexing="ij")

    # Scale offsets to image size
    dx = dx[None] * (CROPPED_IMAGE_SIZE / W)
    dy = dy[None] * (CROPPED_IMAGE_SIZE / H)
    logger.debug(f"Crop offsets dx shape: {dx.shape}, dy shape: {dy.shape}")

    # Shift grid to each car center
    grid = torch.stack(
        (
            cx[:, None, None] + dx,
            cy[:, None, None] + dy,
        ),
        dim=-1,
    )  # (N, crop, crop, 2)

    # Get car headings and compute rotation angles
    headings = robot_heading(env, asset_cfg)  # (N, 2)
    angles = torch.atan2(-headings[:, 0], headings[:, 1]) + torch.pi  # (N,) Negative x to match image coords

    cos = torch.cos(angles).view(-1, 1, 1, 1)  # (N,1,1,1)
    sin = torch.sin(angles).view(-1, 1, 1, 1)  # (N,1,1,1)

    center = torch.stack((cx, cy), dim=-1).view(-1, 1, 1, 2)  # (N,1,1,2)
    rel = grid - center  # (N,H,W,2)
    rel_x = rel[..., 0:1]
    rel_y = rel[..., 1:2]

    rot_x = cos * rel_x - sin * rel_y
    rot_y = sin * rel_x + cos * rel_y

    grid = torch.cat((rot_x, rot_y), dim=-1) + center  # (N,H,W,2)

    # Sample with border padding
    crops = F.grid_sample(
        imgs,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (N, 3, crop, crop)

    # Convert to grayscale
    gray_crops = 0.2989 * crops[:, 0:1, :, :] + 0.5870 * crops[:, 1:2, :, :] + 0.1140 * crops[:, 2:3, :, :]  # (N, 1, crop, crop)

    gray_crops = (gray_crops > 0.5).float()
    # log the shape of gray crops

    # If hide_previous_track is true, the bottom half of the image is whited out
    if hide_previous_track:
        gray_crops[:, :, CROPPED_IMAGE_SIZE//2:, :] = 1.0

    logger.debug("Obtained track neighborhood images.")
    logger.debug_lazy("Track neighborhood image shape: %s", lambda: list(gray_crops.shape))

    logger.debug_lazy("Track neighborhood image min/max: %s", lambda: list((gray_crops.min().item(), gray_crops.max().item())))

    # print image shape for debugging
    # print(f"track neighborhood image shape: {gray_crops.shape}")

    #save debug image
    # debug_img = gray_crops[0, 0].cpu().numpy() * 255
    # cv2.imwrite("debug_track_neighborhood_image.png", debug_img)

    # draw the robot position on the static track image
    # static_track_image = env.unwrapped.static_track_image
    # for i in range(N):
    #     img = static_track_image[i].cpu().numpy()
    #     x = int(car_xy[i, 0].item())
    #     y = int(car_xy[i, 1].item())
    #     cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    #     cv2.imwrite(f"debug_static_track_image_{i}.png", img)

    return gray_crops # (N, 1, crop, crop)

def aligned_obstacle_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, multi_obstacle_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """
    Get the aligned obstacle image of shape (N, 1, crop, crop).

    This function generates an image of the obstacles in the environment, aligned with the car's heading. 
    The obstacles are represented as black trapezoids on a white background. The resulting image is cropped and rotated to match the car's perspective.

    Returns:
        torch.Tensor: Aligned obstacle images of shape (N, 1, crop, crop).
    """

    logger = get_logger(env)

    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        pass
    else:
        # return a dummy image if static image is not available
        num_envs = env.unwrapped.num_envs
        logger.warning("static track image not found, returning dummy gray image")
        return torch.zeros((num_envs, 1, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), device=env.unwrapped.device, dtype=torch.float32)

    device = env.unwrapped.device
    N = env.unwrapped.num_envs
    crop = OBSTACLE_CROPPED_IMAGE_SIZE

    # Image size from static map
    imgs = env.unwrapped.static_track_image
    N, H, W, _ = imgs.shape

    # blank obstacle map in same resolution as track image
    if not hasattr(env.unwrapped, 'obstacle_map'):
        env.unwrapped.obstacle_map = torch.ones((N, 1, H, W), device=device)
    obstacle_map = env.unwrapped.obstacle_map

    if not env.unwrapped.enable_obstacles:
        # if obstacles are disabled, return a white image
        return torch.ones((N, 1, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), device=device)

    robot_xy = robot_pos(env, asset_cfg, do_convert_coords=True)
    robot_headings = robot_heading(env, asset_cfg)

    # save images in obstacle map
    # for i in range(N):
    #     cv2.imwrite(f"obstacle_map_{i}.png", obstacle_map[i, 0].cpu().numpy() * 255)

    # ----- SAME ROTATION / CROPPING AS TRACK FUNCTION -----

    cx = 2.0 * robot_xy[:, 0] / (W - 1) - 1.0
    cy = 2.0 * robot_xy[:, 1] / (H - 1) - 1.0

    lin = torch.linspace(-1, 1, crop, device=device)
    dy, dx = torch.meshgrid(lin, lin, indexing="ij")

    dx = dx[None] * (crop / W)
    dy = dy[None] * (crop / H)

    grid = torch.stack(
        (
            cx[:, None, None] + dx,
            cy[:, None, None] + dy,
        ),
        dim=-1,
    )

    headings = robot_headings
    angles = torch.atan2(-headings[:, 0], headings[:, 1]) + torch.pi

    cos = torch.cos(angles).view(-1,1,1,1)
    sin = torch.sin(angles).view(-1,1,1,1)

    center = torch.stack((cx, cy), dim=-1).view(-1,1,1,2)
    rel = grid - center

    rel_x = rel[...,0:1]
    rel_y = rel[...,1:2]

    rot_x = cos * rel_x - sin * rel_y
    rot_y = sin * rel_x + cos * rel_y

    grid = torch.cat((rot_x, rot_y), dim=-1) + center

    crops = F.grid_sample(
        obstacle_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    logger.debug("Obtained aligned obstacle images.")

    # resize the crops to the desired dimension 
    if CROPPED_IMAGE_SIZE != OBSTACLE_CROPPED_IMAGE_SIZE:
        crops = F.interpolate(crops, 
                            size=(CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), 
                            mode='area')
        
    crops = (crops > 0.5).float() # binarize the image
    # save debug image
    # debug_img = crops[0, 0].cpu().numpy() * 255
    # cv2.imwrite("debug_aligned_obstacle_image.png", debug_img)

    return crops


def generate_agents_map(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Generate ONE shared map containing all agents as black trapezoids.
    GPU-friendly PyTorch version (no cv2 / no numpy).

    Returns:
        torch.Tensor: shape (1, 1, H_map, W_map), float32
                     white background = 1.0, agents = 0.0
    """
    device = env.unwrapped.device
    dtype = torch.float32

    N = env.unwrapped.num_envs
    H, W = env.unwrapped.static_track_image.shape[1:3]

    agents_map = torch.ones((1, 1, H, W), device=device, dtype=dtype)

    # Agent poses in global/image coordinates
    robot_xy = robot_pos(env, asset_cfg, do_convert_coords=True, global_frame=True).to(device=device, dtype=dtype)
    robot_headings = robot_heading(env, asset_cfg).to(device=device, dtype=dtype)

    # -------------------------
    # Trapezoid geometry
    # -------------------------
    heading_offset_rad = math.pi / 2.0

    size_scale = 0.75
    top_w = 5.0 * size_scale
    bottom_w = 15.0 * size_scale
    length = 10.0 * size_scale
    h = length / 2.0

    top_half = top_w / 2.0
    bottom_half = bottom_w / 2.0

    # A square patch large enough to contain any rotated trapezoid
    patch_radius = int(math.ceil(math.sqrt(bottom_half**2 + h**2))) + 2
    patch_size = 2 * patch_radius + 1

    # Local integer offsets for the patch
    offs = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=dtype)
    dy_patch, dx_patch = torch.meshgrid(offs, offs, indexing="ij")   # shape (P, P)

    # Expand to all agents: shape (N, P, P)
    cx = robot_xy[:, 0].view(N, 1, 1)   # x = column
    cy = robot_xy[:, 1].view(N, 1, 1)   # y = row

    x_global = cx + dx_patch.unsqueeze(0)
    y_global = cy + dy_patch.unsqueeze(0)

    # Pixel indices for writing into image
    x_idx = torch.round(x_global).long()
    y_idx = torch.round(y_global).long()

    in_bounds = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)

    # -------------------------
    # Compute local coordinates
    # -------------------------
    theta = torch.atan2(robot_headings[:, 1], robot_headings[:, 0]) + heading_offset_rad
    c = torch.cos(theta).view(N, 1, 1)
    s = torch.sin(theta).view(N, 1, 1)

    # Relative coordinates in global image frame
    dx = x_global - cx
    dy = y_global - cy

    # Inverse rotation: global -> local trapezoid frame
    # local y points along vehicle length
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy

    # -------------------------
    # Analytic trapezoid test
    # -------------------------
    # Width changes linearly from top to bottom as y goes from -h to +h
    t = (y_local + h) / length
    half_width = top_half + t * (bottom_half - top_half)

    inside = (
        (y_local >= -h) &
        (y_local <= h) &
        (torch.abs(x_local) <= half_width)
    )

    mask = inside & in_bounds

    # -------------------------
    # Paint all trapezoids black
    # -------------------------
    flat_mask = mask.reshape(-1)
    flat_y = y_idx.reshape(-1)[flat_mask]
    flat_x = x_idx.reshape(-1)[flat_mask]

    agents_map[0, 0, flat_y, flat_x] = 0.0

    # save debug image
    # debug_img = agents_map[0, 0].cpu().numpy() * 255
    # cv2.imwrite("debug_agents_map.png", debug_img)

    return agents_map


def aligned_other_agents_image(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Generate aligned image of other agents in frame, using one shared agent map.

    Returns:
        torch.Tensor: (N, 1, crop, crop)
    """
    logger = get_logger(env)

    if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
        pass
    else:
        # return a dummy stack of images if static image is not available
        num_envs = env.unwrapped.num_envs
        dummy_img = torch.ones((num_envs, 1, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE))

        stack_size = 4
        return dummy_img.repeat(1, stack_size, 1, 1)
    
    device = env.unwrapped.device
    N = env.unwrapped.num_envs
    crop = OBSTACLE_CROPPED_IMAGE_SIZE

    # base tile dimensions
    _, H, W, _ = env.unwrapped.static_track_image.shape

    # shared map with all agents
    agents_map = generate_agents_map(env, asset_cfg)
    map_h, map_w = agents_map.shape[-2], agents_map.shape[-1]

    # centers for each agent in shared-map pixel coordinates
    robot_xy = robot_pos(env, asset_cfg, do_convert_coords=True, global_frame=True)
    robot_headings = robot_heading(env, asset_cfg)
    center_x_px = robot_xy[:, 0]
    center_y_px = robot_xy[:, 1]

    # normalize centers to [-1, 1] wrt shared map size
    cx = 2.0 * center_x_px / (map_w - 1) - 1.0
    cy = 2.0 * center_y_px / (map_h - 1) - 1.0

    lin = torch.linspace(-1, 1, crop, device=device)
    dy, dx = torch.meshgrid(lin, lin, indexing="ij")
    dx = dx[None] * (crop / map_w)
    dy = dy[None] * (crop / map_h)

    grid = torch.stack((cx[:, None, None] + dx, cy[:, None, None] + dy), dim=-1)

    angles = torch.atan2(-robot_headings[:, 0], robot_headings[:, 1]) + torch.pi
    cos = torch.cos(angles).view(-1, 1, 1, 1)
    sin = torch.sin(angles).view(-1, 1, 1, 1)

    center = torch.stack((cx, cy), dim=-1).view(-1, 1, 1, 2)
    rel = grid - center
    rel_x = rel[..., 0:1]
    rel_y = rel[..., 1:2]

    rot_x = cos * rel_x - sin * rel_y
    rot_y = sin * rel_x + cos * rel_y
    grid = torch.cat((rot_x, rot_y), dim=-1) + center

    # reuse one map for all N crops
    agents_map_batched = agents_map.expand(N, -1, -1, -1)
    crops = F.grid_sample(
        agents_map_batched,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # resize the crops to the desired dimension 
    if CROPPED_IMAGE_SIZE != OBSTACLE_CROPPED_IMAGE_SIZE:
        crops = F.interpolate(crops, 
                            size=(CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE), 
                            mode='area')

    crops = (crops > 0.5).float()

    # Here is where we initialize the stacked obstacle images
    if not hasattr(env, "_obstacle_image_stack"):
        stack_size = 4
        num_envs, _, H, W = crops.shape
        env._stack_size = stack_size
        env._obstacle_image_stack = crops.repeat(1, stack_size, 1, 1)
        env._obstacle_stack_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    if not hasattr(env, "stack_step_counters"):
        env.stack_step_counters = torch.zeros(num_envs, dtype=torch.int32, device=device)

    stack = env._obstacle_image_stack

    # find which envs need initialization (first time or after reset)
    to_init = ~env._obstacle_stack_initialized

    if to_init.any():
        stack[to_init] = crops[to_init].repeat(1, env._stack_size, 1, 1)
        env._obstacle_stack_initialized[to_init] = True
        env.stack_step_counters[to_init] = 0

    already_init = env._obstacle_stack_initialized & (~to_init)

    # sample only when counter reaches frequency
    to_update = already_init & (env.stack_step_counters >= env.stack_frequency_steps)

    if to_update.any():
        stack[to_update, 1:] = stack[to_update, :-1].clone()
        stack[to_update, 0] = crops[to_update, 0]
        env.stack_step_counters[to_update] = 0

    # envs that were initialized but not updated this step
    not_updated = already_init & (~to_update)
    if not_updated.any():
        env.stack_step_counters[not_updated] += 1
    
    # save all stacked images for debugging for env 0
    # for j in range(env._stack_size):
    #     debug_img = stack[0, j].cpu().numpy() * 255
    #     cv2.imwrite(f"debug_other_agents_stack_env{0}_frame{j}.png", debug_img)
    
    return stack


def aligned_robot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, normalize: bool) -> torch.Tensor:
    """
    Calculate the car's velocity in the car's local frame, aligned with its heading.

    Args:
        env: The environment instance.
        asset_cfg: The configuration of the car asset.
        normalize: Whether to normalize the velocity.

    Returns:
        torch.Tensor: The velocity vector of the car in its local frame.
    """
    logger = get_logger(env)
    asset: RigidObject = env.scene[asset_cfg.name]

    if hasattr(env.unwrapped, 'action_driver'):
        driver = env.unwrapped.action_driver
        max_throttle = driver.max_throttle
        if max_throttle != 30:
            logger.warning(f"Unexpected throttle limits for speed normalization: max_throttle={max_throttle}. Speed normalization may be incorrect.")
    local_vel = legacy_aligned_velocity(
        asset.data.root_com_lin_vel_w,
        asset.data.heading_w,
        normalization_speed_mps=0.8 if normalize else None,
    )

    logger.debug_lazy("Car local frame velocities: %s", lambda: local_vel)
    return local_vel



def front_wheels_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Get the front wheels steering angle of the car.

    
    """
    logger = get_logger(env)
    # robot: Articulation = env.scene[asset_cfg.asset_name]
    robot = env.unwrapped.scene["robot"]

    front_ids, _ = robot.find_joints(FRONT_STEER_NAMES)

    front_right_angle = robot.data.joint_pos[:, front_ids][:, 0] # Num_envs x 1
    front_left_angle = robot.data.joint_pos[:, front_ids][:, 1] # Num_envs x 1

    logger.debug_lazy("Front right wheel angle: %s", lambda: front_right_angle)
    logger.debug_lazy("Front left wheel angle: %s", lambda: front_left_angle)

    return front_right_angle, front_left_angle


# Below are methods for reset agent training

def get_reset_agent_observation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the observation for the reset agent, which consists of the car's position and heading frame coordinates

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param robot_asset_cfg: The configuration of the robot asset.
    :type robot_asset_cfg: SceneEntityCfg

    :return: A tensor containing the car's (x, y) position and heading angle.
    :rtype: torch.Tensor
    """
    logger = get_logger(env)
    image_width, image_height = env.unwrapped.get_camera_dimensions()

    pos = robot_pos(env, asset_cfg, do_convert_coords=True)  # (N, 2)
    heading = robot_heading(env, asset_cfg)  # (N, 2)
    heading_angle_to_vertical = -torch.atan2(heading[:, 0], -heading[:, 1]) 
    
    speed = robot_speed(env, asset_cfg, normalize=True)  # (N,)

    if not hasattr(env.unwrapped, 'destination_pos'):
        destination = torch.zeros((env.unwrapped.num_envs, 2), device=pos.device)  # (N, 2)
    else:
        destination = env.unwrapped.destination_pos  # (N, 2)


    vec_to_goal = destination - pos  # (N, 2)
    vec_to_goal = rotate_vector(vec_to_goal, heading_angle_to_vertical) 

    distance_to_goal = torch.norm(vec_to_goal, dim=1, keepdim=True)  # (N, 1)
    direction_to_goal = vec_to_goal / (distance_to_goal + 1e-8)  # (N, 2)

    distance_to_goal_normalized = distance_to_goal / torch.norm(torch.tensor([float(image_width), float(image_height)], device=distance_to_goal.device))  # (N, 1)

    angle_of_vec_to_goal_with_vertical = torch.atan2(direction_to_goal[:, 0], -direction_to_goal[:, 1])  # (N,)

    if hasattr(env.unwrapped, 'yellow_walls_tensor') and env.unwrapped.yellow_walls_tensor.numel() > 0:
        closest_yellow_wall_point = find_closest_point(pos, env.unwrapped.yellow_walls_tensor, env.unwrapped.yellow_walls_lengths)  # (N, 2)
        vec_to_wall = closest_yellow_wall_point - pos  # (N, 2)
        vec_to_wall = rotate_vector(vec_to_wall, heading_angle_to_vertical)
        distance_to_wall = torch.norm(vec_to_wall, dim=1, keepdim=True)  # (N, 1)
        distance_to_wall_normalized = distance_to_wall / torch.norm(torch.tensor([float(image_width), float(image_height)], device=distance_to_wall.device))  # (N, 1)
        vec_to_wall_normalized = vec_to_wall / (torch.norm(vec_to_wall, dim=1, keepdim=True) + 1e-8)  # (N, 2)
    else:
        vec_to_wall_normalized = torch.zeros((pos.shape[0], 2), device=pos.device)  # (N, 2)
        distance_to_wall_normalized = torch.zeros((pos.shape[0], 1), device=pos.device)  # (N, 1)
    
    observation = torch.cat((speed.unsqueeze(-1), distance_to_goal_normalized, direction_to_goal, angle_of_vec_to_goal_with_vertical.unsqueeze(-1), vec_to_wall_normalized, distance_to_wall_normalized), dim=-1)  # (N, 2 + 2 + 2 + 2 + 1 + 2 + 1) = (N, 12)

    logger.debug_lazy("Reset agent observation: %s", lambda: observation)

    # try:
    #     print(f"episode reward sum: ", env.reward_manager._episode_sums)
    #     # reward_sum = 0
    #     # for reward_term in env.reward_manager._episode_sums.keys():
    #     #     reward_sum += env.reward_manager._episode_sums[reward_term]
    #     # print(f"episode reward sum by summing terms: ", reward_sum)
    #     # print(f"bonus reward: ", env.reward_manager._episode_sums["terminal_bonus"])
    # except Exception as e:
    #     pass
    
    # # for debugging, get the camera image
    # if hasattr(env.unwrapped, 'static_track_image') and env.unwrapped.static_track_image is not None:
    #     for env_index in range(env.unwrapped.num_envs):

    #         camera_img = env.unwrapped.static_track_image  # (N, H, W, 3)
    #         env_image = camera_img[env_index].cpu().numpy()

    #         # draw position and destination on the image for debugging
    #         pos_coords = (int(pos[env_index, 0].item()), int(pos[env_index, 1].item()))
    #         dest_coords = (int(destination[env_index, 0].item()), int(destination[env_index, 1].item()))
    #         heading_vector = (int(heading[env_index, 0].item() * 10), int(heading[env_index, 1].item() * 10)) # scale for visibility
    #         cv2.arrowedLine(env_image, pos_coords, (pos_coords[0] + heading_vector[0], pos_coords[1] + heading_vector[1]), color=(255, 0, 0), thickness=2) # blue for heading
            
    #         # draw position and destination on the image for debugging
    #         cv2.circle(env_image, pos_coords, radius=5, color=(0, 255, 0), thickness=2)  # green for position
    #         cv2.circle(env_image, dest_coords, radius=5, color=(0, 0, 255), thickness=2)  # red for destination

    #         # draw direction to goal
    #         direction_vector = (int(direction_to_goal[env_index, 0].item() * 10), int(direction_to_goal[env_index, 1].item() * 10))
    #         cv2.arrowedLine(env_image, pos_coords, (pos_coords[0] + direction_vector[0], pos_coords[1] + direction_vector[1]), color=(255, 255, 0), thickness=2) # cyan for direction to goal
            
    #         # write distance to goal on the image
    #         cv2.putText(env_image, f"{distance_to_goal[env_index].item():.2f}", (pos_coords[0] + 10, pos_coords[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #         cv2.imwrite(f"debug_reset_agent_observation_{env_index}.png", env_image)

    # make a white image of the same size as the image and draw the vec to goal on it for debugging
    # for env_index in range(env.unwrapped.num_envs):
    #     debug_img = np.ones((384, 512, 3), dtype=np.uint8) * 255
        
    #     # draw direction to goal
    #     direction_vector = (int(direction_to_goal[env_index, 0].item() * 10), int(direction_to_goal[env_index, 1].item() * 10))
    #     cv2.arrowedLine(debug_img, (256, 192), (256 + direction_vector[0], 192 + direction_vector[1]), color=(255, 0, 0), thickness=2) # blue for direction to goal

    #     direction_vector_wall = (int(vec_to_wall_normalized[env_index, 0].item() * 10), int(vec_to_wall_normalized[env_index, 1].item() * 10))
    #     cv2.arrowedLine(debug_img, (256, 192), (256 + direction_vector_wall[0], 192 + direction_vector_wall[1]), color=(0, 255, 0), thickness=2) # green for direction to wall

    #     # write speed on the image
    #     cv2.putText(debug_img, f"Speed: {observation[env_index, 0].item():.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     # write distance to goal on the image
    #     cv2.putText(debug_img, f"Distance to goal: {observation[env_index, 1].item():.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     # write angle to goal on the image
    #     cv2.putText(debug_img, f"Angle to goal: {observation[env_index, 4].item():.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     # write distance to wall on the image
    #     cv2.putText(debug_img, f"Distance to wall: {distance_to_wall_normalized[env_index, 0].item():.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     cv2.imwrite(f"debug_reset_agent_observation_vec_to_goal_{env_index}.png", debug_img)

    env.unwrapped.current_observation = {
        "pos": pos.clone(),
        "heading": heading.clone(),
        "speed": speed.clone(),
        "destination": destination.clone(),
        "distance_to_goal": distance_to_goal.clone(),
        "distance_to_goal_normalized": distance_to_goal_normalized.clone(),
        "direction_to_goal": direction_to_goal.clone(),
        "distance_to_wall_normalized": distance_to_wall_normalized.clone(),
        "angle_of_vec_to_goal_with_vertical": angle_of_vec_to_goal_with_vertical.clone(),
    }
    return observation

def angle_between(a: torch.Tensor, b: torch.Tensor):
    # a, b shape: (N, 2)

    dot = (a * b).sum(dim=1)                 # (N,)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]  # scalar 2D cross product (N,)

    angle = torch.atan2(torch.abs(cross), dot)  # radians in [0, pi]
    return angle


def rotate_vector(vec: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 2D vector by a given angle.

    Args:
        vec: A tensor of shape (N, 2) representing the vectors to rotate.
        angle: A tensor of shape (N,) representing the rotation angles in radians.

    Returns:
        A tensor of shape (N, 2) representing the rotated vectors.
    """
    cos = torch.cos(angle).view(-1, 1)  # (N, 1)
    sin = torch.sin(angle).view(-1, 1)  # (N, 1)

    rot_matrix = torch.cat(
        [
            torch.cat([cos, -sin], dim=-1).unsqueeze(-2),  # (N, 1, 2)
            torch.cat([sin, cos], dim=-1).unsqueeze(-2),   # (N, 1, 2)
        ],
        dim=-2,
    )  # (N, 2, 2)

    rotated_vec = torch.bmm(rot_matrix, vec.unsqueeze(-1)).squeeze(-1)  # (N, 2)
    return rotated_vec
