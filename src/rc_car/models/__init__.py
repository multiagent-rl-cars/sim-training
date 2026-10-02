"""Portable PyTorch models; importing this module does not start Isaac Sim.

The default CPU dependencies include these models. The legacy ``car_training``
import path still initializes the Isaac extension and is not a portable entry point.
"""

from rc_car.models.feature_extractor import CustomMultiInputExtractor, ResidualBlock
from rc_car.models.inputs import validate_feature_extractor_inputs

__all__ = [
    "CustomMultiInputExtractor",
    "ResidualBlock",
    "validate_feature_extractor_inputs",
]
