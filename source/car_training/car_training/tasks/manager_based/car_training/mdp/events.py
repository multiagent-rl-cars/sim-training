from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import logging
import cv2
import torch
import math
import numpy as np

from PIL import Image
from pxr import UsdGeom
from isaacsim.core.utils.prims import get_prim_at_path

from isaaclab.envs.mdp import reset_scene_to_default
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords_list



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import GpuSafeLogger

from car_training.tasks.manager_based.car_training.mdp.observation import multi_obstacles_pos, multi_obstacles_heading

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



def set_robot_visibility(env: ManagerBasedRLEnv, visible: bool):
    """
    Set visibility for all robot prims following naming convention:
    Robot, Robot2, Robot3, ...
    """

    logger = get_logger(env)

    for env_path in env.scene.env_prim_paths:
        env_prim = get_prim_at_path(env_path)

        # iterate over all children under this env
        for child in env_prim.GetChildren():
            name = child.GetName()

            # match Robot, Robot2, Robot3, ...
            if name.startswith("Robot"):
                logger.info(f"{'Showing' if visible else 'Hiding'} robot prim at path: {child.GetPath()}")

                imageable = UsdGeom.Imageable(child)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()



def init_static_track_image(env: ManagerBasedRLEnv, camera_name: str = "overhead_camera", num_warmup_steps: int = 10):
    """
    Initialize and capture a static track image from a specified camera in the environment. This method should be called once for all. 
    The function temporarily hides the robot(s) in the environment to ensure they do not appear in the captured image.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param camera_name: The name of the camera from which to capture the static track image
    :type camera_name: str
    :param num_warmup_steps: The number of warmup steps to render before capturing
    :type num_warmup_steps: int
    :return: The captured static track image as a tensor of shape (num_envs, H, W, 3).
    :rtype: torch.Tensor
    """


    logger = get_logger(env)
    logger.info("Initializing static track image from camera: %s", camera_name)

    # hide robots so they don't appear
    set_robot_visibility(env, visible=False)

    logger.debug("Rendering simulation for %d warmup steps to capture static track image.", num_warmup_steps)
    # Render some new frames (avoiding using step)
    for _ in range(num_warmup_steps):
        # env.step(actions)
        # update scene buffers (including sensors) if available
        if hasattr(env.scene, "update"):
            env.scene.update(dt=0.0)

        # render a frame so RTX sensors/cameras refresh
        if hasattr(env.sim, "render"):
            env.sim.render()

    rgb = env.scene[camera_name].data.output["rgb"]  # (num_envs, H, W, 3)

    # for debugging: save images
    # for i in range(rgb.shape[0]):
    #     img = rgb[i].cpu().numpy()
    #     Image.fromarray(img).save(f"{camera_name}_hidden_robot_{i}.png")

    static_track_image = rgb.clone()

    # show robots back
    set_robot_visibility(env, visible=True)

    return static_track_image



def reset_robot_root_random(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
    z=0.05,
    yaw_noise_std=0.2,     # radians, e.g. 0.2 ~= 11.5 degrees
    lookahead=5,           # how many points ahead to compute tangent
    xy_rand_radius=0.05,   # random offset radius in XY plane
    top_half_only=False,      # whether to only sample from the top half of the track (for obstacles)
):
    """
    Reset the robot's root position and orientation to a random location along the track. The robot 
    is positioned at a random track point with a small random offset in the XY plane and a yaw orientation aligned 
    with the track direction, optionally with some noise.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param env_ids: The IDs of the environments to reset.
    :type env_ids: torch.Tensor
    :param asset_cfg: The configuration of the robot asset.
    :type asset_cfg: SceneEntityCfg
    :param z: The Z-coordinate (height) at which to place the robot.
    :type z: float
    :param yaw_noise_std: Standard deviation of the yaw noise to add to the robot's orientation (in radians).
    :type yaw_noise_std: float
    :param lookahead: Number of track points ahead to compute the tangent direction for yaw alignment.
    :type lookahead: int
    :param xy_rand_radius: Radius for random XY offset from the track point in meters.
    :type xy_rand_radius: float
    
    """

    if hasattr(env.unwrapped, "track_points_world_by_track") and len(env.unwrapped.track_points_world_by_track) > 0:
        return reset_robot_root_random_multiagent(
            env,
            env_ids,
            asset_cfg,
            z=z,
            yaw_noise_std=yaw_noise_std,
            lookahead=lookahead,
            xy_rand_radius=xy_rand_radius,
            top_half_only=top_half_only,
        )

    logger = get_logger(env)

    robot = env.scene[asset_cfg.name]
    device = env.device
    n = env_ids.numel()

    tp_local = env.unwrapped.track_points_world  # local-to-env points (x,y)
    if len(tp_local) == 0:
        reset_scene_to_default(env, env_ids)
        return

    origins = env.scene.env_origins

    pos = torch.empty((n, 3), device=device, dtype=torch.float32)
    yaw = torch.empty((n,), device=device, dtype=torch.float32)

    for k, eid in enumerate(env_ids.tolist()):
        pts = tp_local[eid]
        # if top half only, exclude points with y < 0
        if top_half_only:
            pts = [point for point in pts if float(point[1]) >= 0.0]
        if len(pts) < 2:
            raise RuntimeError("Not enough track points to reset robot position.")
        else:
            L = len(pts)
            j = int(torch.randint(0, L, (1,), device=device).item())

            # current point
            x0, y0 = pts[j]

            # lookahead point (wrap around)
            j2 = (j + lookahead) % L
            x1, y1 = pts[j2]

            # tangent direction
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)

            logger.debug("Resetting robot in env %d to track point %d (lookahead %d to point %d) with delta (%.3f, %.3f).", eid, j, lookahead, j2, dx, dy)

            # if degenerate (same point), try a different lookahead
            if dx == 0.0 and dy == 0.0:
                j2 = (j + 1) % L
                x1, y1 = pts[j2]
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)

            yaw_k = math.atan2(dy, dx)

            # add small noise around track direction so car isn't perfectly aligned
            if yaw_noise_std and yaw_noise_std > 0:
                yaw_k += float(torch.randn((), device=device) * yaw_noise_std)
            
            # 50 % of the time, flip direction (180 deg)
            if torch.rand((), device=device) < 0.5:
                yaw_k += math.pi

        logger.debug("reset position before random shift: (%.3f, %.3f).", x0, y0)
        # add random shift to x,y position within a radius
        angle = torch.rand((), device=device) * 2 * math.pi
        distance = torch.rand((), device=device) * xy_rand_radius
        shift_x = float(distance) * math.cos(float(angle))
        shift_y = float(distance) * math.sin(float(angle))
        x0 += shift_x
        y0 += shift_y
        logger.debug("Applying random shift (%.3f, %.3f) to reset position.", shift_x, shift_y)

        # local -> world
        pos[k, 0] = float(x0) + float(origins[eid, 0])
        pos[k, 1] = float(y0) + float(origins[eid, 1])
        pos[k, 2] = float(z) + float(origins[eid, 2])

        yaw[k] = yaw_k

        logger.debug("Robot reset position for env %d: (%.3f, %.3f, %.3f).", eid, pos[k,0], pos[k,1], pos[k,2])

    # yaw -> quaternion (w,x,y,z)
    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)

    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def reset_robot_root_random_multiagent(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
    z: float = 0.05,
    yaw_noise_std: float = 0,
    lookahead: int = 10,
    xy_rand_radius: float = 0,
    top_half_only: bool = False,
    min_spawn_distance: float = 3,
    track_groups_attr: str = "track_points_world_by_track",
    track_ids_attr: str = "multiagent_track_ids",
    max_tries: int = 1000,
):
    """
    Reset a multiagent robot onto one of several isolated loop tracks.

    Each environment keeps a fixed track assignment, and reset candidates are rejected if they
    spawn too close to other cars on the same track.
    """

    logger = get_logger(env)

    robot = env.scene[asset_cfg.name]
    device = env.device
    n = env_ids.numel()

    track_groups = getattr(env.unwrapped, track_groups_attr, None)
    if not track_groups:
        return reset_robot_root_random(
            env,
            env_ids,
            asset_cfg,
            z=z,
            yaw_noise_std=yaw_noise_std,
            lookahead=lookahead,
            xy_rand_radius=xy_rand_radius,
            top_half_only=top_half_only,
        )

    per_env_track_groups = (
        len(track_groups) == env.num_envs
        and len(track_groups) > 0
        and isinstance(track_groups[0], (list, tuple))
        and len(track_groups[0]) > 0
        and isinstance(track_groups[0][0], (list, tuple, np.ndarray, torch.Tensor))
    )

    reference_groups = track_groups[0] if per_env_track_groups else track_groups
    num_tracks = len(reference_groups)

    track_ids = getattr(env.unwrapped, track_ids_attr, None)
    if track_ids is None or len(track_ids) != env.num_envs:
        track_ids = torch.arange(env.num_envs, device=device, dtype=torch.long) % num_tracks
        setattr(env.unwrapped, track_ids_attr, track_ids)
    else:
        track_ids = torch.as_tensor(track_ids, device=device, dtype=torch.long)

    pos = torch.empty((n, 3), device=device, dtype=torch.float32)
    yaw = torch.empty((n,), device=device, dtype=torch.float32)

    reserved_xy: dict[int, list[tuple[float, float]]] = {track_id: [] for track_id in range(num_tracks)}

    if hasattr(robot.data, "root_pos_w"):
        current_xy = robot.data.root_pos_w[:, :2].detach()
        reset_set = set(env_ids.tolist())
        for eid in range(env.num_envs):
            if eid in reset_set:
                continue
            track_id = int(track_ids[eid].item())
            if 0 <= track_id < num_tracks:
                reserved_xy[track_id].append((float(current_xy[eid, 0]), float(current_xy[eid, 1])))

    for k, eid in enumerate(env_ids.tolist()):
        track_id = int(track_ids[eid].item())
        current_track_groups = track_groups[eid] if per_env_track_groups else track_groups
        pts = current_track_groups[track_id]

        if len(pts) < 2:
            raise RuntimeError(f"Not enough track points to reset robot position for track {track_id}.")

        if top_half_only:
            pts = [point for point in pts if float(point[1]) >= 0.0]
            if len(pts) < 2:
                raise RuntimeError(f"Not enough top-half track points to reset robot position for track {track_id}.")

        chosen = None
        debug_info = None
        for _ in range(max_tries):
            L = len(pts)
            j = int(torch.randint(0, L, (1,), device=device).item())

            x0, y0 = pts[j]

            j2 = (j + lookahead) % L
            x1, y1 = pts[j2]
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)
            if dx == 0.0 and dy == 0.0:
                j2 = (j + 1) % L
                x1, y1 = pts[j2]
                dx = float(x1) - float(x0)
                dy = float(y1) - float(y0)

            yaw_k = math.atan2(dy, dx)
            if yaw_noise_std and yaw_noise_std > 0:
                yaw_k += float(torch.randn((), device=device) * yaw_noise_std)
            if torch.rand((), device=device) < 0.5:
                yaw_k += math.pi

            angle = torch.rand((), device=device) * 2 * math.pi
            distance = torch.rand((), device=device) * xy_rand_radius
            shift_x = float(distance) * math.cos(float(angle))
            shift_y = float(distance) * math.sin(float(angle))

            candidate_x = float(x0) + shift_x
            candidate_y = float(y0) + shift_y

            ok = True
            for px, py in reserved_xy[track_id]:
                if math.hypot(candidate_x - px, candidate_y - py) < min_spawn_distance:
                    ok = False
                    break

            if ok:
                chosen = (candidate_x, candidate_y, yaw_k)
                debug_info = (j, j2, dx, dy, shift_x, shift_y)
                reserved_xy[track_id].append((candidate_x, candidate_y))
                break

        if chosen is None:
            raise RuntimeError(
                f"Could not find a collision-free spawn point for env {eid} on track {track_id}."
            )

        x, y, yaw_k = chosen
        j, j2, dx, dy, shift_x, shift_y = debug_info
        logger.debug(
            "Resetting multiagent robot in env %d on track %d to track point %d (lookahead %d to point %d) with delta (%.3f, %.3f) and shift (%.3f, %.3f).",
            eid, track_id, j, lookahead, j2, dx, dy, shift_x, shift_y,
        )

        pos[k, 0] = x
        pos[k, 1] = y
        pos[k, 2] = float(z)
        yaw[k] = yaw_k

        logger.debug(
            "Multiagent robot reset position for env %d: (%.3f, %.3f, %.3f).",
            eid, pos[k, 0], pos[k, 1], pos[k, 2],
        )

    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)

    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)



def reset_obstacles_random(
    env,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    z: float = 0.06,
    xy_rand_radius: float = 0.03,
    top_half_only: bool = True,
    min_separation: float = 1.0,   # keep obstacles apart
):
    """
    Place multiple cube obstacles at random track locations.
    Assumes env.track_points_world has already been initialized.
    """
    device = env.device
    origins = env.scene.env_origins

    if len(env.track_points_world) == 0:
        return

    num_obs = len(obstacle_names)

    # collect poses for each obstacle separately because each asset writes separately
    obstacle_positions = {
        name: torch.zeros((len(env_ids), 3), device=device, dtype=torch.float32)
        for name in obstacle_names
    }
    obstacle_quats = {
        name: torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(len(env_ids), 1)
        for name in obstacle_names
    }

    for k, eid in enumerate(env_ids.tolist()):
        pts = env.track_points_world[eid]

        if top_half_only:
            pts = [p for p in pts if float(p[1]) >= 0.0]

        if len(pts) < num_obs:
            raise RuntimeError(f"Not enough track points in env {eid} to place {num_obs} obstacles.")

        chosen_positions = []

        max_tries = 500
        tries = 0
        while len(chosen_positions) < num_obs and tries < max_tries:
            tries += 1

            j = int(torch.randint(0, len(pts), (1,), device=device).item())
            x0, y0 = pts[j]

            angle = torch.rand((), device=device) * 2 * math.pi
            distance = torch.rand((), device=device) * xy_rand_radius
            x = float(x0) + float(distance) * math.cos(float(angle))
            y = float(y0) + float(distance) * math.sin(float(angle))

            ok = True
            for (px, py) in chosen_positions:
                if math.hypot(x - px, y - py) < min_separation:
                    ok = False
                    break

            if ok:
                chosen_positions.append((x, y))

        if len(chosen_positions) < num_obs:
            raise RuntimeError(f"Could not place {num_obs} separated obstacles in env {eid}.")

        for obs_idx, name in enumerate(obstacle_names):
            x, y = chosen_positions[obs_idx]
            obstacle_positions[name][k, 0] = x + float(origins[eid, 0])
            obstacle_positions[name][k, 1] = y + float(origins[eid, 1])
            obstacle_positions[name][k, 2] = z + float(origins[eid, 2])

    # write poses to sim
    for name in obstacle_names:
        obstacle = env.scene[name]
        pose = torch.cat([obstacle_positions[name], obstacle_quats[name]], dim=-1)
        if env.unwrapped.enable_obstacles:
            obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        else:
            obstacle.write_root_pose_to_sim(torch.zeros_like(pose), env_ids=env_ids)  # move to origin if obstacles are disabled
        obstacle.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids=env_ids)

def generate_obstacles_map(
    env,
    env_ids: torch.Tensor,
    multi_obstacle_cfgs: list[SceneEntityCfg],
    side_len: float = 4.0,
):
    """
    Generate a binary obstacle map on GPU for rotated square obstacles.

    Output convention:
        1.0 = free space
        0.0 = obstacle

    Assumptions:
        - multi_obstacles_pos(..., do_convert_coords=True) returns image-plane coordinates
          in pixel units, shape either:
              (num_envs, num_obstacles, 2)
          or
              (len(env_ids), num_obstacles, 2)
        - multi_obstacles_heading(...) returns heading vectors, shape matching above:
              (..., num_obstacles, 2)
        - obstacle_map should have shape (num_envs, 1, H, W)
    """
    device = env.device
    half = side_len / 2.0

    # --------------------------------------------------
    # Get obstacle centers + headings
    # --------------------------------------------------
    multi_obstacle_xy = multi_obstacles_pos(env, multi_obstacle_cfgs, do_convert_coords=True)
    multi_obstacle_headings = multi_obstacles_heading(env, multi_obstacle_cfgs)

    # Decide whether helper returns all envs or only requested env_ids
    returns_all_envs = (multi_obstacle_xy.shape[0] == env.num_envs)

    if returns_all_envs:
        xy = multi_obstacle_xy[env_ids]                # (B, M, 2)
        headings = multi_obstacle_headings[env_ids]   # (B, M, 2)
    else:
        xy = multi_obstacle_xy                        # (B, M, 2)
        headings = multi_obstacle_headings            # (B, M, 2)

    B, M, _ = xy.shape

    # --------------------------------------------------
    # Allocate obstacle map once
    # --------------------------------------------------
    if not hasattr(env.unwrapped, "obstacle_map"):
        camera_width, camera_height = env.unwrapped.get_camera_dimensions()
        env.unwrapped.obstacle_map = torch.ones(
            (env.num_envs, 1, camera_height, camera_width),
            device=device,
            dtype=torch.float32,
        )

    _, _, H, W = env.unwrapped.obstacle_map.shape

    # --------------------------------------------------
    # Cache pixel grid on GPU
    # grid_x, grid_y shapes: (H, W)
    # --------------------------------------------------
    grid_cache_ok = (
        hasattr(env.unwrapped, "_obstacle_grid_x")
        and hasattr(env.unwrapped, "_obstacle_grid_y")
        and env.unwrapped._obstacle_grid_x.shape == (H, W)
        and env.unwrapped._obstacle_grid_y.shape == (H, W)
        and env.unwrapped._obstacle_grid_x.device == device
        and env.unwrapped._obstacle_grid_y.device == device
    )

    if not grid_cache_ok:
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # both (H, W)
        env.unwrapped._obstacle_grid_x = grid_x
        env.unwrapped._obstacle_grid_y = grid_y

    grid_x = env.unwrapped._obstacle_grid_x
    grid_y = env.unwrapped._obstacle_grid_y

    # --------------------------------------------------
    # Compute heading angle for each obstacle
    # headings[..., 0] = hx, headings[..., 1] = hy
    # theta shape: (B, M)
    # --------------------------------------------------
    theta = torch.atan2(headings[..., 1], headings[..., 0])
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # --------------------------------------------------
    # Broadcast everything:
    #
    # xy[..., 0] -> cx shape (B, M, 1, 1)
    # xy[..., 1] -> cy shape (B, M, 1, 1)
    # grid_x/y    ->     shape (1, 1, H, W)
    #
    # world/image relative coordinates:
    # dx = x - cx
    # dy = y - cy
    #
    # rotate by -theta into obstacle local frame:
    # x_local =  cos(theta)*dx + sin(theta)*dy
    # y_local = -sin(theta)*dx + cos(theta)*dy
    # --------------------------------------------------
    cx = xy[..., 0].unsqueeze(-1).unsqueeze(-1)   # (B, M, 1, 1)
    cy = xy[..., 1].unsqueeze(-1).unsqueeze(-1)   # (B, M, 1, 1)

    cos_t = cos_t.unsqueeze(-1).unsqueeze(-1)     # (B, M, 1, 1)
    sin_t = sin_t.unsqueeze(-1).unsqueeze(-1)     # (B, M, 1, 1)

    gx = grid_x.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
    gy = grid_y.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)

    dx = gx - cx                                  # (B, M, H, W)
    dy = gy - cy                                  # (B, M, H, W)

    x_local =  cos_t * dx + sin_t * dy
    y_local = -sin_t * dx + cos_t * dy

    # Inside rotated square
    inside = (x_local.abs() <= half) & (y_local.abs() <= half)   # (B, M, H, W)

    # Any obstacle occupies the pixel
    occupied = inside.any(dim=1)                                  # (B, H, W)

    batch_map = torch.ones((B, H, W), device=device, dtype=torch.float32)
    batch_map[occupied] = 0.0

    env.unwrapped.obstacle_map[env_ids, 0] = batch_map

def reset_stacked_obstacles_images(env, env_ids: torch.Tensor):
    env.unwrapped._obstacle_stack_initialized[env_ids] = False
    env.unwrapped.stack_step_counters[env_ids] = 0

def choose_rand_point(pts) -> list[float, float]:
    """
    Choose a random point from a list of points.

    :param pts: A list of points, where each point is a tuple (x, y).
    :type pts: list[tuple[float, float]]

    :return: A randomly chosen point from the list.
    :rtype: list[float, float]
    """
    if len(pts) == 0:
        raise ValueError("No points available to choose from.")

    
    idx = torch.randint(0, len(pts), (1,)).item()
    return pts[idx]


##### ---------- Below are functions for Reset Agent Training ---------- #####
def reset_destination_pos(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    Reset the destination position for the specified environments to a random point on the track.

    :param env: The environment instance.
    :type env: ManagerBasedRLEnv
    :param env_ids: The IDs of the environments to reset.
    :type env_ids: torch.Tensor
    """
    logger = get_logger(env)

    tp_frame = env.unwrapped.track_points  # in to coordinates of the frame
    if len(tp_frame) == 0:
        logger.warning("No track points available to reset destination position.")
        return

    for eid in env_ids.tolist():
        pts = tp_frame[eid]
        if len(pts) == 0:
            logger.warning("No track points available for env %d to reset destination position.", eid)
            continue
        
        random_point_on_track = choose_rand_point(pts)
        env.unwrapped.destination_pos[eid] = torch.tensor(random_point_on_track, device=env.device)
        env.unwrapped.destination_yaw_deg[eid] = np.random.uniform(-180, 180)  # random yaw between -180 and 180 degrees
    # convert destination_pos to world coordinates for setting goal markers
    camera_width, camera_height = env.get_camera_dimensions()
    track_dimensions: tuple = env.scene.cfg.ground_plane.ground_plane_params.each_track_size
    env.unwrapped.destination_pos_world = convert_coords_list(env.destination_pos.cpu().numpy(), track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
    env.unwrapped.set_goal_markers(env.destination_pos_world, yaw_deg_tensor=env.destination_yaw_deg) # set goal marker positions and orientations

def reset_robot_root_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
    position,
    yaw,
    zero_velocity: bool = True,
):
    """
    Reset the robot root to a specified position and yaw.

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The environment instance.
    env_ids : torch.Tensor
        Environment IDs to reset, shape (N,).
    asset_cfg : SceneEntityCfg
        Robot asset config.
    position : torch.Tensor | list | tuple
        Either:
        - single position (3,) applied to all env_ids
        - per-env positions (N, 3)
    yaw : float | torch.Tensor
        Either:
        - single yaw scalar applied to all env_ids
        - per-env yaw tensor of shape (N,)
    zero_velocity : bool
        If True, resets root linear/angular velocity to zero.
    """
    robot = env.scene[asset_cfg.name]
    device = env.device
    n = env_ids.numel()

    # ----- position handling -----
    pos = torch.as_tensor(position, device=device, dtype=torch.float32)

    if pos.ndim == 1:
        if pos.numel() != 3:
            raise ValueError(f"Single position must have shape (3,), got {tuple(pos.shape)}")
        pos = pos.unsqueeze(0).repeat(n, 1)
    elif pos.ndim == 2:
        if pos.shape != (n, 3):
            raise ValueError(f"Per-env positions must have shape ({n}, 3), got {tuple(pos.shape)}")
    else:
        raise ValueError(f"Position must have shape (3,) or ({n}, 3), got {tuple(pos.shape)}")

    # ----- yaw handling -----
    yaw_t = torch.as_tensor(yaw, device=device, dtype=torch.float32)

    if yaw_t.ndim == 0:
        yaw_t = yaw_t.repeat(n)
    elif yaw_t.ndim == 1:
        if yaw_t.shape != (n,):
            raise ValueError(f"Per-env yaw must have shape ({n},), got {tuple(yaw_t.shape)}")
    else:
        raise ValueError(f"Yaw must be scalar or shape ({n},), got {tuple(yaw_t.shape)}")

    # ----- yaw -> quaternion (w, x, y, z) -----
    half = 0.5 * yaw_t
    qw = torch.cos(half)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    qz = torch.sin(half)
    quat = torch.stack([qw, qx, qy, qz], dim=-1)  # (N, 4)

    # convert position from local to world coordinates
    origins = env.scene.env_origins[env_ids]  # (N, 3)
    pos += origins

    # ----- write pose -----
    root_pose = torch.cat([pos, quat], dim=-1)  # (N, 7)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    # ----- optionally zero velocities -----
    if zero_velocity:
        robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)

def reset_robot_root_random_deterministic_mix(
        env, 
        env_ids: torch.Tensor,
        asset_cfg
):
    """
    Reset robot position 50% of time randomly on the track and 50% of time to a corner.
    """
    possible_positions_yaws = []
    max_extent = 4.5
    points = np.linspace(-max_extent, max_extent, num=10)  # 10 points along each edge
    for x in points:
        possible_positions_yaws.append(((x, max_extent, 0.05), torch.pi/2))   # top edge
        possible_positions_yaws.append(((x, -max_extent, 0.05), 3*torch.pi/2))  # bottom edge
    for y in points:
        possible_positions_yaws.append(((-max_extent, y, 0.05), torch.pi))  # left edge
        possible_positions_yaws.append(((max_extent, y, 0.05), 2*torch.pi))   # right edge

    possible_positions_yaws.append(((-max_extent, max_extent, 0.05), torch.pi))
    possible_positions_yaws.append(((max_extent, max_extent, 0.05), 2*torch.pi))
    possible_positions_yaws.append(((-max_extent, -max_extent, 0.05), torch.pi))
    possible_positions_yaws.append(((max_extent, -max_extent, 0.05), 2*torch.pi))

    random_number = torch.rand((), device=env.device)
    if random_number < 0.7:  # 70% random on track, 30% corners
        reset_robot_root_random(env, env_ids, asset_cfg, yaw_noise_std=0.6, xy_rand_radius=0.0)
    else:

        position_yaw = possible_positions_yaws[torch.randint(0, len(possible_positions_yaws), (1,)).item()]
        position, yaw = position_yaw
        reset_robot_root_pose(env, env_ids, asset_cfg, position=position, yaw=yaw)