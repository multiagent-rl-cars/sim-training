# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg as ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

from . import mdp
from car_training.tasks.manager_based.car_training.mdp.rewards import distance_from_center_reward, steering_jerk_penalty, velocity_reward, remains_alive_reward, turning_penalty, collision_penalty
from car_training.tasks.manager_based.car_training.mdp.observation import  (
    aligned_track_neighborhood_image,
    robot_vel,
    aligned_robot_vel,
    aligned_other_agents_image,
    aligned_obstacle_image
)
from car_training.tasks.manager_based.car_training.mdp.termination import off_track_termination, collision_termination, inside_obstacle_termination
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.configs.leatherback_params import LeatherbackParamsCfg
from car_training.tasks.manager_based.car_training.configs.ttr_params import TTRNanoParamsCfg
from car_training.tasks.manager_based.car_training.configs.plane_track_params import GroundPlaceCfg, TrackParamsCfg
from car_training.tasks.manager_based.car_training.configs.overhead_camera_params import OverheadCameraParamsCfg
from car_training.tasks.manager_based.car_training.mdp.events import reset_robot_root_random, reset_obstacles_random, generate_obstacles_map

import os
from pathlib import Path

# Override explicitly for generated fixtures or an external original asset bundle.
ASSET_ROOT = Path(os.environ.get("CAR_TRAINING_ASSET_ROOT", Path(__file__).parent / "assets")).expanduser().resolve()
leatherback_usd_path = str(ASSET_ROOT / "robot_assets/leatherback_simple_better.usd")
ttr_nano_path = str(ASSET_ROOT / "robot_assets/ttr_nano_v1.usd")
TRACK_USD_PATHS = [str(ASSET_ROOT / f"scene_assets/tracks_thickness_random/track_{i}.usd") for i in range(12)]



## 
# Leatherback Articulation Config
##
leather_back_params = LeatherbackParamsCfg()

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=leatherback_usd_path,                       
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=leather_back_params.rigid_props.rigid_body_enabled,
            max_linear_velocity=leather_back_params.rigid_props.max_linear_velocity,
            max_angular_velocity=leather_back_params.rigid_props.max_angular_velocity,
            max_depenetration_velocity=leather_back_params.rigid_props.max_depenetration_velocity,
            enable_gyroscopic_forces=leather_back_params.rigid_props.enable_gyroscopic_forces,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=leather_back_params.articulation_props.enabled_self_collisions,
            solver_position_iteration_count=leather_back_params.articulation_props.solver_position_iteration_count,
            solver_velocity_iteration_count=leather_back_params.articulation_props.solver_velocity_iteration_count,
            sleep_threshold=leather_back_params.articulation_props.sleep_threshold,
            stabilization_threshold=leather_back_params.articulation_props.stabilization_threshold,
        ),),

    init_state= ArticulationCfg.InitialStateCfg(
        pos=leather_back_params.init_state.pos,
        rot=leather_back_params.init_state.rot,
        joint_pos=leather_back_params.init_state.joint_pos,
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=leather_back_params.actuators.throttle_joint_names_expr,
            effort_limit=leather_back_params.actuators.throttle_effort_limit,
            velocity_limit=leather_back_params.actuators.throttle_velocity_limit,
            stiffness=leather_back_params.actuators.throttle_stiffness,
            damping=leather_back_params.actuators.throttle_damping,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=leather_back_params.actuators.steering_joint_names_expr,
            effort_limit=leather_back_params.actuators.steering_effort_limit,
            velocity_limit=leather_back_params.actuators.steering_velocity_limit,
            stiffness=leather_back_params.actuators.steering_stiffness,
            damping=leather_back_params.actuators.steering_damping,
        ),
    },
    
)

##
# Scene definition
##


@configclass
class PlaneCfg(AssetBaseCfg):
    """Ground Plane configuration."""
    ground_plane_params = GroundPlaceCfg()

    prim_path="/World/Plane"
    spawn: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg(
        size=ground_plane_params.size,
        visible=ground_plane_params.visible,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


@configclass
class TrackCfg(AssetBaseCfg):
    """Per-env track configuration."""
    track_params = TrackParamsCfg()

    prim_path: str = "{ENV_REGEX_NS}/Track"
    print("track prim path:", prim_path)
    spawn: sim_utils.MultiUsdFileCfg = sim_utils.MultiUsdFileCfg(
        usd_path=TRACK_USD_PATHS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=track_params.rigid_props.rigid_body_enabled,
            max_linear_velocity=track_params.rigid_props.max_linear_velocity,
            max_angular_velocity=track_params.rigid_props.max_angular_velocity,
            max_depenetration_velocity=track_params.rigid_props.max_depenetration_velocity,
            enable_gyroscopic_forces=track_params.rigid_props.enable_gyroscopic_forces,
        ),
        random_choice=False,  # env_id % N assignment
    )

@configclass
class OverheadCameraCfg(TiledCameraCfg):
    """Configuration for the per-env overhead camera."""
    camera_params = OverheadCameraParamsCfg()
    
    prim_path: str = "{ENV_REGEX_NS}/OverheadCamera"
    offset: TiledCameraCfg.OffsetCfg = TiledCameraCfg.OffsetCfg(pos=camera_params.pose.pos, rot=camera_params.pose.rot, convention=camera_params.pose.convention)
    data_types: list = camera_params.image.data_types
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=camera_params.optics.focal_length,
        focus_distance=camera_params.optics.focus_distance,
        horizontal_aperture=camera_params.optics.horizontal_aperture,
        clipping_range=camera_params.optics.clipping_range,
        f_stop=camera_params.optics.fstop,
    )
    width: int = camera_params.image.width
    height: int = camera_params.image.height

def make_cube_obstacle(prim_path: str, size=(0.12, 0.12, 0.12), color=(1.0, 0.0, 0.0)) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # obstacle stays fixed unless you move it manually
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -10.0),   # hide them initially if you want
        ),
    )

@configclass
class CarTrainingSceneCfg(InteractiveSceneCfg):
    """Configuration for a driving scene."""
    replicate_physics: bool = False # for the tracks which are different we need to add this

    # ground plane
    ground_plane: PlaneCfg = PlaneCfg()

    # track
    track: TrackCfg = TrackCfg()

    # robot
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=LEATHERBACK_CFG.spawn.replace(
            activate_contact_sensors=True,
        ),
    )
    # robot_2: ArticulationCfg = LEATHERBACK_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Robot2",
    #     spawn=LEATHERBACK_CFG.spawn.replace(
    #         activate_contact_sensors=True,
    #     ),
    # )

    # overhead camera
    # overhead_camera: OverheadCameraCfg | None = None

    robot_chassis_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis",
        history_length=3, 
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    obstacle_0 = make_cube_obstacle("{ENV_REGEX_NS}/Obstacle_0", color=(1.0, 0.0, 0.0))
    obstacle_1 = make_cube_obstacle("{ENV_REGEX_NS}/Obstacle_1", color=(1.0, 0.0, 0.0))
    obstacle_2 = make_cube_obstacle("{ENV_REGEX_NS}/Obstacle_2", color=(1.0, 0.0, 0.0))
    obstacle_3 = make_cube_obstacle("{ENV_REGEX_NS}/Obstacle_3", color=(1.0, 0.0, 0.0))

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    drive = ActionTermCfg(
        class_type=ThrottleSteerAckermannAction,
        asset_name="robot"
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        track_image = ObsTerm(func=aligned_track_neighborhood_image,
                        params={"asset_cfg": SceneEntityCfg("robot"), "global_frame": False},
                        clip=(0.0, 255.0), scale=255.0,)
        obstacle_image = ObsTerm(func=aligned_obstacle_image, # needs robot and obstacle cfg to find the relative pose and then align the image accordingly
                        params={"asset_cfg": SceneEntityCfg("robot"), "multi_obstacle_cfgs": [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")]}, 
                        clip=(0.0, 255.0), scale=255.0,)
        robot_vel = ObsTerm(func=aligned_robot_vel,
                                params={"asset_cfg": SceneEntityCfg("robot"), "normalize": True},
                                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_obstacles_position = EventTerm(func=reset_obstacles_random, mode="reset", params={"obstacle_names": ["obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"],
                                                                                            "z": 0.06, "top_half_only": False})
    generating_obs_map = EventTerm(func=generate_obstacles_map, mode="reset", params={"multi_obstacle_cfgs": [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")]})
    # reset for obstacle car
    # reset_robot_2_position = EventTerm(
    #     func=reset_robot_root_random,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_2"),
    #         "z": 0.05,
    #         "yaw_noise_std": 0.0, # radians, e.g. 0.2 ~= 11.5 degrees
    #         "lookahead": 5,       # how many points ahead to compute tangent
    #         "xy_rand_radius": 0.5
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Get total reward from mdp.rewards.get_total_reward
    # total_reward = RewTerm(
    #     func=get_total_reward,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "max_speed": 1,    # TODO: Replace with actual max speed
    #     },
    #     weight=1.0,
    # )

    distance_from_center_rew_term = RewTerm(
        func=distance_from_center_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "multi_obstacle_cfgs": [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")],
        },
        weight=3.0,
    )

    velocity_rew_term = RewTerm(
        func=velocity_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot")
        },
        weight=1.5,
    )

    remain_alive_rew_term = RewTerm(
        func=remains_alive_reward,
        weight=0.2
        )
    
    steer_penalty_term = RewTerm(
        func=turning_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "multi_obstacle_cfgs": [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")],
        },
        weight=0.5
    )

    # obstacle_collision_penalty = RewTerm(
    #     func=collision_penalty,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "obstacle_cfg": SceneEntityCfg("obstacle_car")},
    #     weight=0.5
    # )
        
    # steering_jerk_penalty_term = RewTerm(
    #     func=steering_jerk_penalty,
    #     weight=0.003
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Out of bounds termination
    out_of_bounds = DoneTerm(func=off_track_termination, 
                             params={"asset_cfg": SceneEntityCfg("robot"), "global_frame": False},)

    # collision_chassis = DoneTerm(func=mdp.terminations.illegal_contact,
    #                      params={
    #                          "threshold": 0.05,
    #                          "sensor_cfg": SceneEntityCfg("robot_chassis_contact_sensor"),
    #                      })

    inside_obstacle_term = DoneTerm(func=inside_obstacle_termination,
                            params={"asset_cfg": SceneEntityCfg("robot"), 
                                    "multi_obstacle_cfgs": [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2"), SceneEntityCfg("obstacle_3")],
                                    })
    
    # time_out_term = DoneTerm(func=mdp.time_out)



@configclass
class CommandCfg:
    """No command configuration for the MDP."""
    pass


@configclass
class CurriculumCfg:
    """Curriculum configuration for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class CarTrainingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CarTrainingSceneCfg = CarTrainingSceneCfg(num_envs=1, env_spacing=15.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # no command config
    commands: CommandCfg = CommandCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 # How many physics substeps the same action is applied for
        self.episode_length_s = 150
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 40.0)
        # simulation settings
        self.sim.dt = 0.05
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        # assign tracks to envs
