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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg

from . import mdp
from car_training.tasks.manager_based.car_training.mdp.rewards import reset_agent_distance_reward, get_terminal_bonus, heading_alignment_reward, reset_agent_velocity_reward, reset_agent_turning_penalty, combined_heading_distance_reward
from car_training.tasks.manager_based.car_training.mdp.observation import get_reset_agent_observation
from car_training.tasks.manager_based.car_training.mdp.termination import out_of_view_termination, reached_goal_termination
from car_training.tasks.manager_based.car_training.mdp.actions import ThrottleSteerAckermannAction
from car_training.tasks.manager_based.car_training.configs.leatherback_params import LeatherbackParamsCfg
from car_training.tasks.manager_based.car_training.configs.ttr_params import TTRNanoParamsCfg
from car_training.tasks.manager_based.car_training.configs.plane_track_params import GroundPlaceCfg, TrackParamsCfg
from car_training.tasks.manager_based.car_training.configs.overhead_camera_params import OverheadCameraParamsCfg
from car_training.tasks.manager_based.car_training.mdp.events import reset_robot_root_random, reset_robot_root_pose, reset_robot_root_random_deterministic_mix

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
        pos=leather_back_params.init_state.pos_reset_agent, # I am using the reset agent position here
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

ttr_nano_params = TTRNanoParamsCfg()

TTR_NANO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=ttr_nano_path,                       
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=ttr_nano_params.rigid_props.rigid_body_enabled,
            max_linear_velocity=ttr_nano_params.rigid_props.max_linear_velocity,
            max_angular_velocity=ttr_nano_params.rigid_props.max_angular_velocity,
            max_depenetration_velocity=ttr_nano_params.rigid_props.max_depenetration_velocity,
            enable_gyroscopic_forces=ttr_nano_params.rigid_props.enable_gyroscopic_forces,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=ttr_nano_params.articulation_props.enabled_self_collisions,
            solver_position_iteration_count=ttr_nano_params.articulation_props.solver_position_iteration_count,
            solver_velocity_iteration_count=ttr_nano_params.articulation_props.solver_velocity_iteration_count,
            sleep_threshold=ttr_nano_params.articulation_props.sleep_threshold,
            stabilization_threshold=ttr_nano_params.articulation_props.stabilization_threshold,
        ),),

    init_state= ArticulationCfg.InitialStateCfg(
        pos=ttr_nano_params.init_state.pos,
        rot=ttr_nano_params.init_state.rot,
        joint_pos=ttr_nano_params.init_state.joint_pos,
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=ttr_nano_params.actuators.throttle_joint_names_expr,
            effort_limit=ttr_nano_params.actuators.throttle_effort_limit,
            velocity_limit=ttr_nano_params.actuators.throttle_velocity_limit,
            stiffness=ttr_nano_params.actuators.throttle_stiffness,
            damping=ttr_nano_params.actuators.throttle_damping,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=ttr_nano_params.actuators.steering_joint_names_expr,
            effort_limit=ttr_nano_params.actuators.steering_effort_limit,
            velocity_limit=ttr_nano_params.actuators.steering_velocity_limit,
            stiffness=ttr_nano_params.actuators.steering_stiffness,
            damping=ttr_nano_params.actuators.steering_damping,
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

def make_wall(prim_path: str, size_xyz, pos_xyz) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size_xyz,  # full extents (x, y, z) in meters
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # immovable "static" wall
                disable_gravity=True,
            ),
            # optional visual
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz),
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
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # overhead camera
    # overhead_camera: OverheadCameraCfg | None = None

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    wall_left: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/WallLeft")
    wall_right: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/WallRight")
    wall_bottom: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/WallBottom")
    wall_top: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/WallTop")

    

    def __post_init__(self):
        super().__post_init__()

        xmin, xmax, ymin, ymax = -5.0, 5.0, -5.0, 5.0

        # tunables
        wall_height: float = 0.8
        wall_thickness: float = 0.02
        wall_margin: float = 0.0

        m = wall_margin
        t = wall_thickness
        h = wall_height

        x_len = (xmax - xmin) + 2.0 * m
        y_len = (ymax - ymin) + 2.0 * m
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        zc = 0.5 * h

        # Assign fully-defined configs (now the scene can spawn them)
        self.wall_left = make_wall("{ENV_REGEX_NS}/WallLeft",   (t, y_len + t, h), (xmin - m - 0.5*t, cy, zc))
        self.wall_right = make_wall("{ENV_REGEX_NS}/WallRight", (t, y_len + t, h), (xmax + m + 0.5*t, cy, zc))
        self.wall_bottom = make_wall("{ENV_REGEX_NS}/WallBottom", (x_len + t, t, h), (cx, ymin - m - 0.5*t, zc))
        self.wall_top = make_wall("{ENV_REGEX_NS}/WallTop",       (x_len + t, t, h), (cx, ymax + m + 0.5*t, zc))

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
        observation_vector = ObsTerm(func=get_reset_agent_observation, params={"asset_cfg": SceneEntityCfg("robot"),},)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # reset_robot_position = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot_position = EventTerm(
    #     func=reset_robot_root_random,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "z": 0.05,
    #         "yaw_noise_std": 0.6, # radians, e.g. 0.2 ~= 11.5 degrees
    #         "lookahead": 5,       # how many points ahead to compute tangent
    #         "xy_rand_radius": 0.0
    #     },
    # )

    reset_robot_position = EventTerm(
        func=reset_robot_root_random_deterministic_mix,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Get total reward from mdp.rewards.get_total_reward
    distance_reward = RewTerm(
        func=reset_agent_distance_reward,
        weight=3.0,
    )

    terminal_bonus = RewTerm(
        func=get_terminal_bonus,
        params={"bonus": 10.0,},
        weight=5.0,
    )

    heading_alignment_rew = RewTerm(
        func=heading_alignment_reward,
        weight=1.0,
    )

    velocity_reward_term = RewTerm(
        func=reset_agent_velocity_reward,
        weight=0.5,
    )

    turning_penalty_term = RewTerm(
        func=reset_agent_turning_penalty,
        weight=2.0,
    )

    # distance_to_wall_penalty_term = RewTerm(
    #     func=distance_to_wall_penalty,
    #     weight=1.0,
    # )

    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Reached goal termination
    reached_goal_term = DoneTerm(func=reached_goal_termination)
    
    out_of_view_term = DoneTerm(func=out_of_view_termination,
                             params={"asset_cfg": SceneEntityCfg("robot"),},)
        
    time_out_term = DoneTerm(func=mdp.time_out)



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
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 40.0)
        # simulation settings
        self.sim.dt = 0.05
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
