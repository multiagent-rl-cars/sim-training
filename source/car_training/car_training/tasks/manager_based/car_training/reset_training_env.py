from __future__ import annotations

import math
from collections.abc import Sequence
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import omni
from pxr import UsdGeom, Gf
import numpy as np

ASSET_FOLDER = Path(__file__).resolve().parents[0] / "assets"
    
import logging
import cv2
from car_training.tasks.manager_based.car_training.car_training_env_cfg import OverheadCameraCfg
from car_training.tasks.manager_based.car_training.mdp.events import init_static_track_image
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import find_track_points_from_image, find_yellow_walls_from_image
from isaaclab.envs import ManagerBasedRLEnv
from car_training.tasks.manager_based.car_training.reset_training_env_cfg import CarTrainingEnvCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import setup_logger, GpuSafeLogger
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.mdp.utils.helpers import convert_coords_list
from car_training.tasks.manager_based.car_training.mdp.events import choose_rand_point

class CarResetEnv(ManagerBasedRLEnv):
    cfg: CarTrainingEnvCfg

    def __init__(self, cfg: CarTrainingEnvCfg, render_mode: str | None = None, task_name: str = "Unknown", console_level: int | None = logging.INFO,
        file_level: int | None = logging.INFO, enable_cameras: bool = True, **kwargs):
        
        self.task_name = task_name
        logger = setup_logger(task_name=task_name, name="car_training", console_level=console_level, file_level=file_level) # I am setting up the logger before calling the super constructor to allow for logging in observations and actions
        self.logger: GpuSafeLogger = GpuSafeLogger(logger)

        self.enable_cameras = enable_cameras
        if self.enable_cameras:
            cfg.scene.overhead_camera = OverheadCameraCfg()

        self.current_observation = None

        super().__init__(cfg, render_mode, **kwargs)
        self.ensure_goal_markers(self.num_envs)

        self.destination_pos = torch.zeros((self.num_envs, 2), device=self.device) # this will be updated at each reset based on the track points
        self.destination_pos_world = [] # this will be updated at each reset based on the track points, in world coordinates
        self.destination_yaw_deg = torch.zeros(self.num_envs, device=self.device) # this will be updated at each reset based on the track points, in world coordinates
        self._static_track_inited = False
        self.static_track_image = None # will be initialized on first reset
        self.track_points = [] # will be initialized on first reset

        self.track_points_world = [] # track points in world coordinates, to be filled after reset

        self.action_driver: ThrottleSteerAckermannAction = self.action_manager._terms["drive"] # this is a ThrottleSteerAckermannAction
        self.goal_distance_threshold = 20 # threshold is number of pixels
        self.reached_goal_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # to keep track of which envs have reached the goal
        self.out_of_bounds_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # to keep track of which envs are out of bounds
        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.device) # to keep track of the previous distance to goal for each env, used in reward calculation
        self.skip_distance_reward_next = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # flag to indicate whether to skip distance reward for the next step for each env, used to avoid giving distance reward on the same step as reaching the goal

        self.goal_hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # to keep track of how many consecutive steps each env has been holding at the goal, used for termination
        self.goal_hold_time_s = 5
        self.goal_hold_steps_required = int(round(self.goal_hold_time_s / self.step_dt))

        self.previous_observation = None # to keep track of the previous observation for each env.

        self.inside_goal_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # to keep track of which envs are currently inside the goal region, used for rewards and logging

        self.yellow_walls_tensor = torch.empty((self.num_envs, 0, 2), device=self.device) 
        self.yellow_walls_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        obs, info = super().reset(seed, env_ids, options)

        # We perform the usual reset plus an initialization of the static track image if needed
        self.__init_static_track_image_if_needed()
        self.__init_track_points_if_needed()
        self.__init_yellow_walls_points()
        
        return obs, info


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

                # save static track image as a .pt file
                # torch.save(self.static_track_image.cpu(), f"{ASSET_FOLDER}/scene_assets/reset_agent_static_track_image_12_env.pt")
                
                self.logger.info("Static track image initialized.")
            else:
                # use torch.load to load pre-saved static track image
                self.static_track_image: torch.Tensor = torch.load(f"{ASSET_FOLDER}/scene_assets/reset_agent_static_track_image_12_env.pt").to(self.device)
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
                random_point_on_track = choose_rand_point(track_points)
                self.destination_pos[index] = torch.tensor(random_point_on_track, device=self.device)
                self.destination_yaw_deg[index] = np.random.uniform(-180, 180)  # random yaw between -180 and 180 degrees

                # below is needed for random resetting
                camera_width, camera_height = self.get_camera_dimensions()
                track_dimensions: tuple = self.scene.cfg.ground_plane.ground_plane_params.each_track_size
                # convert to world coordinates
                world_points = convert_coords_list(track_points, track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
                self.track_points_world.append(world_points)
            
            self.destination_pos_world = convert_coords_list(self.destination_pos.cpu().numpy(), track_dimensions, (camera_width, camera_height), image_to_isaaclab=True)
            self.set_goal_markers(self.destination_pos_world, yaw_deg_tensor=self.destination_yaw_deg) # set initial goal marker positions and orientations

    def __init_yellow_walls_points(self) -> None:
        yellow_walls_list = []
        for index in range(self.num_envs):
                image_np = self.static_track_image[index].detach().cpu().numpy()
                # change to rgb
                image_np = cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB)
                yellow_walls_points = find_yellow_walls_from_image(image_np, debug=index<=30, image_idx=index)
                yellow_walls_list.append(yellow_walls_points)

        seqs = [torch.as_tensor(tp, device=self.device, dtype=torch.float32) for tp in yellow_walls_list]
        self.yellow_walls_tensor = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        self.yellow_walls_lengths = torch.tensor([len(tp) for tp in yellow_walls_list], device=self.device)
        self.logger.info("Yellow walls points initialized from static track image.")

    def ensure_goal_markers(self, num_envs: int, radius: float = 0.1):
        """
        Creates a visual marker in the simulator for the goal position of each environment. This is only useful for debugging and visualization purposes.
        
        :param num_envs: The number of environments to create goal markers for.
        :type num_envs: int
        :param radius: The radius of the goal marker spheres.
        :type radius: float
        """
        stage = omni.usd.get_context().get_stage()
        for i in range(num_envs):
            path = f"/World/envs/env_{i}/GoalMarker"
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                sphere = UsdGeom.Sphere.Define(stage, path)
                sphere.GetRadiusAttr().Set(radius)
                UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # red
                # make sure it has a translate op
                UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp()
            # ---- Add arrow under the marker ----
            arrow_path = f"{path}/Arrow"
            arrow_prim = stage.GetPrimAtPath(arrow_path)
            if not arrow_prim or not arrow_prim.IsValid():
                arrow_prim = self._define_arrow(stage, arrow_path, color=(0.0, 1.0, 0.0))  # green arrow

                arrow_xf = UsdGeom.Xformable(arrow_prim)

                # Make sure arrow has ops we can drive later
                # (translate + orient; you can also add scale if you want)
                arrow_xf.AddTranslateOp()  # local offset relative to marker if desired
                arrow_xf.AddOrientOp()
        
    
    def set_goal_markers(self, goal_pos_w, yaw_deg_tensor: torch.Tensor):
        stage = omni.usd.get_context().get_stage()
        num_envs = len(goal_pos_w)

        for i in range(num_envs):
            marker_path = f"/World/envs/env_{i}/GoalMarker"
            marker_prim = stage.GetPrimAtPath(marker_path)
            if not marker_prim or not marker_prim.IsValid():
                continue

            # Move the marker sphere
            marker_xf = UsdGeom.Xformable(marker_prim)
            marker_t = self.get_or_create_translate_op(marker_xf)
            p = goal_pos_w[i]
            marker_t.Set(Gf.Vec3d(float(p[0]), float(p[1]), 0.1))

            # Arrow
            arrow_path = f"{marker_path}/Arrow"
            arrow_prim = stage.GetPrimAtPath(arrow_path)
            if not arrow_prim or not arrow_prim.IsValid():
                continue

            arrow_xf = UsdGeom.Xformable(arrow_prim)
            arrow_t = self.get_or_create_translate_op(arrow_xf)          # local offset
            orient_op = self.get_or_create_orient_op_float(arrow_xf)     # existing or created

            arrow_t.Set(Gf.Vec3d(0.0, 0.0, 0.15))  # lift arrow above sphere/ground

            yaw_deg =  float(yaw_deg_tensor[i])  # convert from tensor to float
            q_d = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
            orient_op.Set(self.quatd_to_quatf(q_d))

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
            # return self.static_track_image.shape[2], self.static_track_image.shape[1]
            return 512, 384

    def _define_arrow(self, stage, arrow_xf_path: str,
                  shaft_len=0.45, shaft_r=0.03,
                  head_len=0.15, head_r=0.06,
                  color=(0.0, 1.0, 0.0)):
        """
        Creates:
        arrow_xf_path (Xform)
            ├─ Shaft (Cylinder)
            └─ Head  (Cone)
        Oriented by default along +X (so yaw about Z is intuitive for 2D).
        """

        arrow_xf = UsdGeom.Xform.Define(stage, arrow_xf_path)
        arrow_prim = arrow_xf.GetPrim()

        # Shaft
        shaft_path = f"{arrow_xf_path}/Shaft"
        shaft = UsdGeom.Cylinder.Define(stage, shaft_path)
        shaft.GetRadiusAttr().Set(shaft_r)
        shaft.GetHeightAttr().Set(shaft_len)
        UsdGeom.Gprim(shaft.GetPrim()).CreateDisplayColorAttr().Set([color])

        # Head
        head_path = f"{arrow_xf_path}/Head"
        head = UsdGeom.Cone.Define(stage, head_path)
        head.GetRadiusAttr().Set(head_r)
        head.GetHeightAttr().Set(head_len)
        UsdGeom.Gprim(head.GetPrim()).CreateDisplayColorAttr().Set([color])

        shaft_xf = UsdGeom.Xformable(shaft.GetPrim())
        head_xf  = UsdGeom.Xformable(head.GetPrim())

        # Rotate child shapes so local +Z becomes +X (rotate +90° about Y)
        rot_d = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat()   # Quatd
        rot_f = self.quatd_to_quatf(rot_d)  # Quatf

        # IMPORTANT: create orient ops as float precision
        shaft_xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(rot_f)
        head_xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(rot_f)

        # Translate them along +X (since after rotation they point along +X)
        shaft_xf.AddTranslateOp().Set(Gf.Vec3d(shaft_len * 0.5, 0.0, 0.0))
        head_xf.AddTranslateOp().Set(Gf.Vec3d(shaft_len + head_len * 0.5, 0.0, 0.0))

        return arrow_prim

    def quatd_to_quatf(self, qd: Gf.Quatd) -> Gf.Quatf:
        """Convert a Gf.Quatd to Gf.Quatf (float), handling Vec3d -> Vec3f."""
        im_d = qd.GetImaginary()  # usually Gf.Vec3d
        im_f = Gf.Vec3f(float(im_d[0]), float(im_d[1]), float(im_d[2]))
        return Gf.Quatf(float(qd.GetReal()), im_f)

    def get_or_create_translate_op(self, xf: UsdGeom.Xformable) -> UsdGeom.XformOp:
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xf.AddTranslateOp()

    def get_or_create_orient_op_float(self, xf: UsdGeom.Xformable) -> UsdGeom.XformOp:
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                return op
        # Only create if missing
        return xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)


