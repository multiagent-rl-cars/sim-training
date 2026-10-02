"""The inherited residual image/velocity encoder, independent of Isaac imports.

The extraction deliberately retains layer names, BatchNorm, initialization order,
the dummy forward pass, and constructor prints. In particular, constructing the
encoder updates BatchNorm statistics once, as the historical implementation did.
Changing any of these belongs in a separately evaluated model revision.
"""

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """Two convolutions with BatchNorm and an identity skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class CustomMultiInputExtractor(BaseFeaturesExtractor):
    """Encode channel-first track/obstacle images and planar velocity.

    Input keys are ``track_image`` (B, C_track, H, W), ``obstacle_image``
    (B, C_obstacle, H, W), and ``robot_vel`` (B, V). The inherited observation
    producers supply float32 images in [0, 1] and scaled, unclipped velocity.
    The common configuration is one track channel, four obstacle-history
    channels, 48 x 48 pixels, and two velocity components.

    This encoder does not normalize, clip, or validate inputs. SB3 preprocessing
    depends on observation-space metadata; preserve that metadata when loading a
    checkpoint. Call ``validate_feature_extractor_inputs`` explicitly in preflight
    or analysis to check the normalized-input contract without adding synchronization
    overhead or changing the inherited training forward pass.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        track_image_space = cast(spaces.Box, observation_space.spaces["track_image"])
        obstacle_image_space = cast(spaces.Box, observation_space.spaces["obstacle_image"])
        vel_space = cast(spaces.Box, observation_space.spaces["robot_vel"])
        channels = track_image_space.shape[0] + obstacle_image_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Flatten(),
        )

        # Preserve the historical train-mode dummy pass, including its BN updates.
        with torch.no_grad():
            sample_input = torch.zeros(
                1, channels, track_image_space.shape[1], track_image_space.shape[2]
            )
            n_flatten = self.cnn(sample_input).shape[1]
            print(f"CNN output size: {n_flatten}")
            print(f"image shape: {track_image_space.shape}")

        self.image_fc = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self.vel_fc = nn.Sequential(nn.Linear(vel_space.shape[0], 64), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        img = observations["track_image"]
        obstacle_image = observations["obstacle_image"]
        img = torch.cat([img, obstacle_image], dim=1)
        vel = observations["robot_vel"]
        img_features = self.image_fc(self.cnn(img))
        vel_features = self.vel_fc(vel)
        fused = torch.cat([img_features, vel_features], dim=1)
        return self.fc(fused)
