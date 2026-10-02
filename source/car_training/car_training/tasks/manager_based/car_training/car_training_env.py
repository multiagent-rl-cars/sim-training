from __future__ import annotations

import math
from collections.abc import Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

ASSET_FOLDER = Path(__file__).resolve().parents[0] / "assets"
    
import logging
import cv2
from car_training.tasks.manager_based.car_training.car_training_env_cfg import OverheadCameraCfg
from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image, reset_obstacles_random, generate_obstacles_map
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_track_points_from_image
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from car_training.tasks.manager_based.car_training.car_training_env_cfg import CarTrainingEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger, GpuSafeLogger
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords_list

class CarTrainingEnv(ManagerBasedRLEnv):
    cfg: CarTrainingEnvCfg

    def __init__(self, cfg: CarTrainingEnvCfg, render_mode: str | None = None, task_name: str = "Unknown", console_level: int | None = logging.INFO,
        file_level: int | None = logging.INFO, enable_cameras: bool = True, enable_obstacles: bool = True, **kwargs):
        
        self.task_name = task_name
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        self.enable_cameras = enable_cameras
        if self.enable_cameras and getattr(cfg.scene, "overhead_camera", None) is None:
            cfg.scene.overhead_camera = OverheadCameraCfg()

        self.enable_obstacles = enable_obstacles

        super().__init__(cfg, render_mode, **kwargs)

        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset
        self.track_points = [] # will be initialized on first reset
        self.track_points_tensor = torch.empty((self.num_envs, 0, 2), device=self.device) # tensor version of track points for easier processing in rewards and termination, to be filled after reset

        self.track_points_world = [] # track points in world coordinates, to be filled after reset

        self.action_driver: ThrottleSteerAckermannAction = self.action_manager._terms["drive"] # this is a ThrottleSteerAckermannAction

        self.in_obstacle_radius_counter = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # to keep track of whether each robot is currently within the obstacle proximity radius
        self.in_obstacle_time_s = 5
        self.in_obstacle_time_threshold = int(self.in_obstacle_time_s / self.step_dt) 

    def reset(self, seed = None, env_ids = None, options = None) -> tuple[dict[str, torch.Tensor], dict]:
        """
        This is the reset method for the CarTrainingEnv class. This method is called only once at the beginning 
        of the training and is called explicitly by the user through `env.reset()`. This is not the same as the 
        reset method used at the end of each episode, which is handled internally by the environment.
        
        :param seed: Optional seed for the environment reset.
        :type seed: Optional[int]
        :param env_ids: Optional environment IDs to reset.
        :type env_ids: Optional[Sequence[int]]
        :param options: Optional additional options for the reset.
        :type options: Optional[dict]
        """
        # This function is called once at the beginning

        obs, info = super().reset(seed, env_ids, options)

        # We perform the usual reset plus an initialization of the static track image if needed
        self.__init_static_track_image_if_needed()
        self.__init_track_points_if_needed()

        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        reset_obstacles_random(self, env_ids=env_ids, obstacle_names=["obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"],
                               top_half_only=False,)
        
        N, H, W, _ = self.static_track_image.shape
        device = self.static_track_image.device
        self.obstacle_map = torch.ones((N, 1, H, W), device=device)
        generate_obstacles_map(self, env_ids=env_ids, multi_obstacle_cfgs=[SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")],)
        
        # Track and obstacle maps are initialized after the base reset; return
        # observations from the completed reset, not its placeholder images.
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
                self.static_track_image = init_static_track_image(self, camera_name="overhead_camera")
                self._static_track_inited = True
                self.logger.info("Static track image initialized.")

                # save static track image as a .pt file
                # torch.save(self.static_track_image.cpu(), f"{ASSET_FOLDER}/scene_assets/static_track_image_12_env.pt")
            else:
                # use torch.load to load pre-saved static track image
                self.static_track_image: torch.Tensor = torch.load(f"{ASSET_FOLDER}/scene_assets/static_track_image_12_env.pt").to(self.device)
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
                track_points = find_track_points_from_image(image_np, debug=index<=30, image_idx=index)
                self.track_points.append(track_points)

                # below is needed for random resetting
                camera_width, camera_height = self.get_camera_dimensions()
                track_dimensions: tuple = self.scene.cfg.ground_plane.ground_plane_params.each_track_size
                # convert to world coordinates
                world_points = convert_coords_list(track_points, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
                self.track_points_world.append(world_points)
            
            seqs = [torch.as_tensor(tp, device=self.device, dtype=torch.float32) for tp in self.track_points]
            self.track_points_tensor = pad_sequence(seqs, batch_first=True, padding_value=0.0)
            self.track_points_lengths = torch.tensor([len(tp) for tp in self.track_points], device=self.device)
            self.logger.info("Track points initialized from static track image.")
            

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
                print("Warning: Static track image is not initialized. Returning default camera dimensions (512, 386).")
                return 512, 386
            else:
                return self.static_track_image.shape[2], self.static_track_image.shape[1]


