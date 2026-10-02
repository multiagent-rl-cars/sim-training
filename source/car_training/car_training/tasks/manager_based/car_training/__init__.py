# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Basic-Car-Training-v0",
    entry_point="car_training.tasks.manager_based.car_training.car_training_env:CarTrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_training_env_cfg:CarTrainingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Reset-Agent",
    entry_point="car_training.tasks.manager_based.car_training.reset_training_env:CarResetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_training_env_cfg:CarTrainingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_reset_agent_cfg.yaml",
    },
)

gym.register(
    id="Multiagent-Car-Training-v0",
    entry_point="car_training.tasks.manager_based.car_training.multiagent_training_env:MultiAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multiagent_training_env_cfg:MultiAgentEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)