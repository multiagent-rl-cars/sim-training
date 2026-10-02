"""Checkpoint-compatible aliases for the portable model implementation.

The enclosing ``car_training`` package still imports Isaac on initialization.
For simulator-independent model analysis, import ``rc_car.models`` instead.
Historical pickle/SB3 references to the two classes remain resolvable here when
an Isaac runtime and its ``rc-car-research`` dependency are installed.
"""

from rc_car.models.feature_extractor import CustomMultiInputExtractor, ResidualBlock

__all__ = ["CustomMultiInputExtractor", "ResidualBlock"]
