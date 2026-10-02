from __future__ import annotations

import math
from collections.abc import Sequence
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

ASSET_FOLDER = Path(__file__).resolve().parents[0] / "assets"
    
import logging
import cv2
from car_training.tasks.manager_based.car_training.multiagent_training_env_cfg import OverheadCameraCfg
from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image, reset_obstacles_random, generate_obstacles_map, reset_robot_root_random_multiagent
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_all_tracks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from car_training.tasks.manager_based.car_training.multiagent_training_env_cfg import MultiAgentEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger, GpuSafeLogger
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords_list

class MultiAgentEnv(ManagerBasedRLEnv):
    cfg: MultiAgentEnvCfg

    def __init__(self, cfg: MultiAgentEnvCfg, render_mode: str | None = None, task_name: str = "Unknown", console_level: int | None = logging.INFO,
        file_level: int | None = logging.INFO, enable_cameras: bool = True, enable_obstacles: bool = True, **kwargs):
        
        self.task_name = task_name
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        self.enable_cameras = enable_cameras
        if self.enable_cameras and getattr(cfg.scene, "overhead_camera", None) is None:
            cfg.scene.overhead_camera = OverheadCameraCfg()
        if self.enable_cameras:
            # All cameras observe the same world; env slots are cars, not arenas.
            cfg.scene.env_spacing = 0.0

        self.enable_obstacles = enable_obstacles

        super().__init__(cfg, render_mode, **kwargs)

        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset
        self.track_points = [] # will be initialized on first reset
        self.track_points_by_track = [] # per-env list of isolated track point groups
        self.track_points_tensor = torch.empty((self.num_envs, 0, 2), device=self.device) # tensor version of track points for easier processing in rewards and termination, to be filled after reset

        self.track_points_world = [] # track points in world coordinates, to be filled after reset
        self.track_points_world_by_track = [] # world-space isolated track point groups
        self.multiagent_track_ids = None # fixed track assignment for each environment

        self.action_driver: ThrottleSteerAckermannAction = self.action_manager._terms["drive"] # this is a ThrottleSteerAckermannAction

        self.in_obstacle_radius_counter = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # to keep track of whether each robot is currently within the obstacle proximity radius
        self.in_obstacle_time_s = 5
        self.in_obstacle_time_threshold = int(self.in_obstacle_time_s / self.step_dt) 

        # for stacking obstacle images 
        self._obstacle_stack_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stack_frequency_s = 0.3
        self.stack_frequency_steps = int(self.stack_frequency_s / self.step_dt)
        self.stack_step_counters = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def reset(self, seed = None, env_ids = None, options = None) -> tuple[dict[str, torch.Tensor], dict]:
        """
        This is the reset method for the MultiAgentEnv class. This method is called only once at the beginning 
        of the training and is called explicitly by the user through `env.reset()`. This is not the same as the 
        reset method used at the end of each episode, which is handled internally by the environment.
        
        :param seed: Optional seed for the environment reset.
        :type seed: Optional[int]
        :param env_ids: Optional environment IDs to reset.
        :type env_ids: Optional[Sequence[int]]
        :param options: Optional additional options for the reset.
        :type options: Optional[dict]
        """
        # TODO: Adapt to new multiagent setup
        # This function is called once at the beginning

        obs, info = super().reset(seed, env_ids, options)

        # We perform the usual reset plus an initialization of the static track image if needed
        self.__init_static_track_image_if_needed()
        self.__init_track_points_if_needed()

        if self.multiagent_track_ids is None and len(self.track_points_world_by_track) > 0:
            first_track_group_set = self.track_points_world_by_track[0]
            num_tracks = len(first_track_group_set)
            self.multiagent_track_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % num_tracks

        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(self.track_points_world_by_track) > 0:
            reset_robot_root_random_multiagent(
                self,
                env_ids=env_ids,
                asset_cfg=SceneEntityCfg("robot"),
                **{k: v for k, v in self.cfg.events.reset_robot_position_random_marl.params.items() if k != "asset_cfg"},
            )
        
        N, H, W, _ = self.static_track_image.shape
        device = self.static_track_image.device
        self.obstacle_map = torch.ones((N, 1, H, W), device=device)
        # generate_obstacles_map(self, env_ids=env_ids, multi_obstacle_cfgs=[SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")],)
        
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, info


    def __init_static_track_image_if_needed(self) -> None:
        """
        Initialize the static track image if it hasn't been initialized yet.

        This method checks if the static track image has been initialized. If not, it calls the
        `init_static_track_image` function to set it up using the overhead camera.
        
        """
        if not self._static_track_inited:
            if self.enable_cameras:
                captured = init_static_track_image(self, camera_name="overhead_camera")
                # This is one shared arena. Use one map for every car so small
                # camera/render differences cannot create inconsistent track IDs.
                self.static_track_image = captured[:1].expand(self.num_envs, -1, -1, -1)
                self._static_track_inited = True
                self.logger.info("Static track image initialized.")

                # save static track image as a .pt file
                # torch.save(self.static_track_image.cpu(), f"{ASSET_FOLDER}/scene_assets/multiagent_static_track_image_isolated.pt")
            else:
                # use torch.load to load pre-saved static track image
                self.static_track_image: torch.Tensor = torch.load(f"{ASSET_FOLDER}/scene_assets/multiagent_static_track_image_isolated.pt").to(self.device)
                if self.num_envs <= self.static_track_image.shape[0]:
                    # only take into account the first N environments
                    self.static_track_image = self.static_track_image[:self.num_envs]
                else:
                    self.logger.warning("Number of environments is greater than the number of pre-saved static track images. Reusing images.")
                    # repeat the images to match the number of environments
                    repeats = math.ceil(self.num_envs / self.static_track_image.shape[0])
                    self.static_track_image = self.static_track_image.repeat(repeats, 1, 1, 1)[:self.num_envs]

                self._static_track_inited = True
                self.logger.info("Static track image loaded from file.")


    def __init_track_points_if_needed(self) -> None:
        """
        Initialize the track points if they haven't been initialized yet.

        This method checks if the track points have been initialized. If not, it processes the
        static track image to extract the track points.
        
        """
        if len(self.track_points) == 0:
            for index in range(self.num_envs):
                image_np = self.static_track_image[index].detach().cpu().numpy()
                track_groups = self._extract_track_groups_from_image(image_np, debug=index<=30, image_idx=index)

                if len(track_groups) == 0:
                    track_points = find_all_tracks(image_np, debug=index<=30, image_idx=index)
                    track_groups = [np.asarray(track_points)]

                flat_track_points = np.concatenate(track_groups, axis=0)
                self.track_points.append(flat_track_points)
                self.track_points_by_track.append(track_groups)

                # below is needed for random resetting
                camera_width, camera_height = self.get_camera_dimensions()
                track_dimensions: tuple = self.scene.cfg.ground_plane.ground_plane_params.multi_agent_track_size
                # convert to world coordinates
                world_points = convert_coords_list(flat_track_points, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
                self.track_points_world.append(world_points)

                world_groups = [convert_coords_list(track_group, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True) for track_group in track_groups]
                self.track_points_world_by_track.append(world_groups)
            
            seqs = [torch.as_tensor(tp, device=self.device, dtype=torch.float32) for tp in self.track_points]
            self.track_points_tensor = pad_sequence(seqs, batch_first=True, padding_value=0.0)
            self.track_points_lengths = torch.tensor([len(tp) for tp in self.track_points], device=self.device)
            self.logger.info("Track points initialized from static track image.")


    def _extract_track_groups_from_image(self, image_np, debug: bool = False, image_idx: int = 0):
        """Extract separate closed-loop track contours from a static camera image."""
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        binary = cv2.inRange(hsv, lower_black, upper_black)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        track_groups = []
        for contour in contours:
            track_points = contour.squeeze()
            if track_points.ndim == 1:
                continue
            if len(track_points) < 2:
                continue
            track_groups.append(np.asarray(track_points))

        track_groups.sort(key=lambda pts: len(pts), reverse=True)

        if debug:
            debug_image = image_np.copy()
            for pts in track_groups:
                for point in pts:
                    cv2.circle(debug_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            # cv2.imwrite(f"{ASSET_FOLDER}/scene_assets/debug_multiagent_tracks_{image_idx}.png", debug_image)

        return track_groups
            

    def get_camera_dimensions(self) -> tuple[int, int]:
        """
        Get the dimensions of the overhead camera.

        :return: A tuple containing the width and height of the overhead camera.
        :rtype: tuple[int, int]
        """
        if self.enable_cameras:
            camera_cfg = self.scene.cfg.overhead_camera
            return camera_cfg.width, camera_cfg.height
        else:
            if self.static_track_image is None:
                return 2048, 2048
            else:
                return self.static_track_image.shape[2], self.static_track_image.shape[1]

