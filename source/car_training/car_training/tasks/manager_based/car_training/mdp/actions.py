from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.managers import ActionTerm

from isaaclab.managers import SceneEntityCfg, ActionTermCfg
from car_training.tasks.manager_based.car_training.mdp.utils.training_logging import GpuSafeLogger
from rc_car.core.actions import legacy_action_targets

import csv
import os



REAR_WHEEL_NAMES = ["Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"]
FRONT_WHEEL_NAMES = ["Wheel__Knuckle__Front_Right", "Wheel__Knuckle__Front_Left"]
FRONT_STEER_NAMES = ["Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"]

TTR_NANO_REAR_WHEEL_NAMES = ["move_BL", "move_BR"]
TTR_NANO_FRONT_WHEEL_NAMES = ["move_FL", "move_FR"]
TTR_NANO_FRONT_STEER_NAMES = ["turn_L", "turn_R"]


MAX_STEER = 0.6  # rad
MAX_THROTTLE_DRIVING_AGENT = 30
MIN_THROTTLE_DRIVING_AGENT = 10

MAX_THROTTLE_RESET_AGENT = 30
MIN_THROTTLE_RESET_AGENT = -15

def ackermann_to_wheel_angles(delta_center: torch.Tensor,
                             wheelbase: float,
                             trackwidth: float,
                             eps: float = 1e-6):
    """
    Convert a center steering angle (bicycle model) to left/right steering angles.
    delta_center: (N,) in radians
    returns: delta_left (N,), delta_right (N,)
    """
    # tan(delta) can blow up near pi/2; clamp a bit
    tan_delta = torch.tan(delta_center).clamp(min=-1e3, max=1e3)
    R = wheelbase / (tan_delta + eps)  # signed turning radius

    # If delta ~ 0 => R huge => left/right ~ delta_center (approximately)
    # Compute inner/outer wheel angles
    # delta_left is "inner" when turning left (delta_center > 0)
    R_abs = torch.abs(R) + eps

    # Use formulas: tan(delta_inner)=L/(R - W/2), tan(delta_outer)=L/(R + W/2)
    tan_left  = wheelbase / (R - trackwidth * 0.5 + eps)
    tan_right = wheelbase / (R + trackwidth * 0.5 + eps)

    # When turning right, the "inner" is right; but the formulas with signed R already handle it.
    delta_left  = torch.atan(tan_left)
    delta_right = torch.atan(tan_right)

    return delta_left, delta_right


class ThrottleSteerAckermannAction(ActionTerm):
    """
    Policy gives (N,2): [throttle_cmd, steer_cmd]
    We apply:
      - throttle_cmd -> both rear wheel joint velocity targets
      - steer_cmd    -> ackermann left/right -> both front steering joints position targets
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.logger: GpuSafeLogger = env.unwrapped.logger.getChild(self.__class__.__name__)

        # grab articulation handle
        self.robot = env.scene[cfg.asset_name]

        # cache joint indices
        self.rear_ids, _ = self.robot.find_joints(REAR_WHEEL_NAMES)
        self.front_ids, _ = self.robot.find_joints(FRONT_STEER_NAMES)

        self.logger.debug(f"Rear wheel joint IDs: {self.rear_ids}")
        self.logger.debug(f"Front steer joint IDs: {self.front_ids}")

        # Isaac Lab expects action dim known
        self._action_dim = 2
        self.logger.debug(f"Action dimension: {self._action_dim}")

        self._action_curr = torch.full((env.num_envs, self._action_dim), float('nan'), device=env.device) # current action
        self._action_prev = torch.full((env.num_envs, self._action_dim), float('nan'), device=env.device) # previous action
        self._action_prev2 = torch.full((env.num_envs, self._action_dim), float('nan'), device=env.device) # action before previous

        self.buffer_size = 20
        self.action_buffer = torch.zeros(
            (env.num_envs, self.buffer_size, self._action_dim),
            device=env.device
        )
        self.buffer_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        self.max_steer = MAX_STEER

        if env.unwrapped.task_name == "Basic-Car-Training-v0" or env.unwrapped.task_name == "Multiagent-Car-Training-v0":
            self.max_throttle = MAX_THROTTLE_DRIVING_AGENT
            self.min_throttle = MIN_THROTTLE_DRIVING_AGENT
        elif env.unwrapped.task_name == "Reset-Agent":
            self.max_throttle = MAX_THROTTLE_RESET_AGENT
            self.min_throttle = MIN_THROTTLE_RESET_AGENT
        else:
            raise ValueError(f"Unknown task name {env.unwrapped.task_name} for setting throttle limits")
        

    @property
    def action_dim(self):
        return self._action_dim

    def reset(self, env_ids=None):
        pass

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def action_prev(self) -> torch.Tensor | None:
        return self._action_prev
    
    @property
    def action_prev2(self) -> torch.Tensor | None:
        return self._action_prev2
    
    @property
    def action_curr(self) -> torch.Tensor | None:
        return self._action_curr
    
    @action_prev.setter
    def action_prev(self, value: torch.Tensor):
        self._action_prev = value

    @action_prev2.setter
    def action_prev2(self, value: torch.Tensor):
        self._action_prev2 = value

    @action_curr.setter
    def action_curr(self, value: torch.Tensor):
        self._action_curr = value

    def process_actions(self, actions: torch.Tensor):
        """
        actions: (num_envs, 2) coming from env.step(actions).
        Store raw + processed (scaled/clipped).
        """
        self._raw_actions = actions

        self._processed_actions = legacy_action_targets(
            actions,
            min_wheel_speed_rad_s=self.min_throttle,
            max_wheel_speed_rad_s=self.max_throttle,
            max_steering_rad=self.max_steer,
        )
        throttle = self._processed_actions[:, 0]
        self.logger.debug_lazy("Processed throttle commands: %s", lambda: throttle)

        steer = self._processed_actions[:, 1]
        self.logger.debug_lazy("Processed steer commands: %s", lambda: steer)
        
        # self.record_actions_to_file(throttle, steer)

        self.action_buffer = torch.roll(self.action_buffer, shifts=-1, dims=1)
        self.action_buffer[:, -1, :] = self._processed_actions
        self.buffer_count = torch.clamp(self.buffer_count + 1, max=self.buffer_size)

    def apply_actions(self):
        # actions: (num_envs, 2) = [throttle, steer]
        self.logger.debug("In apply_actions()")
        a = self._processed_actions

        self._action_curr = a.clone()  # store current action
        
        throttle = a[:, 0]
        steer = a[:, 1]
        # Rear wheel velocities (same for both rear wheels)
        rear_vel = torch.stack([throttle, throttle], dim=-1)

        # Front steering angles (same for both front joints)
        front_pos = torch.stack([steer, steer], dim=-1)  # order matches your front joint list

        self.logger.debug_lazy("Rear wheel velocity targets: %s", lambda: rear_vel)
        self.logger.debug_lazy("Front steering position targets: %s", lambda: front_pos)

        # Apply
        self.robot.set_joint_velocity_target(rear_vel, joint_ids=self.rear_ids)
        self.robot.set_joint_position_target(front_pos, joint_ids=self.front_ids)

    def reset_history(self, env_ids: list[int] | None) -> None:
        """
        Reset the action history for smooth steering penalty calculation. This is needed on env reset.

        :param env_ids: IDs of the environments to reset.
        :type env_ids: list[int] | None

        """

        self._action_curr[env_ids] = torch.full((len(env_ids), self.action_dim), float('nan'), device=self._action_curr.device)
        self._action_prev[env_ids] = torch.full((len(env_ids), self.action_dim), float('nan'), device=self._action_prev.device)
        self._action_prev2[env_ids] = torch.full((len(env_ids), self.action_dim), float('nan'), device=self._action_prev2.device)  

        self.action_buffer[env_ids] = 0.0
        self.buffer_count[env_ids] = 0
        
    def reset(self, env_ids: list[int] | None) -> None:
        """
        Reset the action term and its history.

        :param env_ids: IDs of the environments to reset.
        :type env_ids: list[int] | None
        """
        super().reset()

        self.reset_history(env_ids=env_ids)

    
    def record_actions_to_file(self, throttle_tensor, steering_tensor):
        """
        Save the throttle and steering commands to a CSV file for analysis.
        """
        

        log_dir = "action_logs"
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "throttle_steering_commands.csv")

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for throttle, steering in zip(throttle_tensor.cpu().numpy(), steering_tensor.cpu().numpy()):
                writer.writerow([throttle, steering])

        self.logger.info(f"Recorded actions to {file_path}")
