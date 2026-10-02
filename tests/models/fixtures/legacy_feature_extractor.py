"""Frozen pre-extraction encoder for numerical compatibility checks.

Copied from car_training/tasks/manager_based/car_training/agents/feature_extractor.py
on 2026-09-25. Load directly by file path; do not import the Isaac package.
This intentionally retains the inherited implementation, including formatting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity   # skip connection
        return F.relu(out)

class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        track_image_space = observation_space.spaces["track_image"]
        obstacle_image_space = observation_space.spaces["obstacle_image"]
        vel_space = observation_space.spaces["robot_vel"]

        # --- CNN with residual blocks ---
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=track_image_space.shape[0] + obstacle_image_space.shape[0], out_channels=32, kernel_size=5, stride=2, padding=2),
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

        # Compute CNN output size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, track_image_space.shape[0] + obstacle_image_space.shape[0], track_image_space.shape[1], track_image_space.shape[2])  # (1, C, H, W)
            n_flatten = self.cnn(sample_input).shape[1]
            print(f"CNN output size: {n_flatten}")
            print(f"image shape: {track_image_space.shape}")

        self.image_fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )

        # --- Velocity branch ---
        self.vel_fc = nn.Sequential(
            nn.Linear(vel_space.shape[0], 64),
            nn.ReLU(),
        )

        # --- Fusion ---
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),

        )

    def forward(self, observations):
        img = observations["track_image"]
        obstacle_image = observations["obstacle_image"]

        img = torch.cat([img, obstacle_image], dim=1)  # concatenate along channel dimension
        vel = observations["robot_vel"]

        img_features = self.image_fc(self.cnn(img)) 
        vel_features = self.vel_fc(vel)

        fused = torch.cat([img_features, vel_features], dim=1)
        return self.fc(fused)
