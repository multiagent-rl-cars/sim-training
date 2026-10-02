"""Compatibility imports for historical standalone experiment scripts.

New code should import ``rc_car.experiments``. This source-path fallback also
supports the old Isaac runtime before its packaging configuration is migrated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rc_car.experiments import (  # noqa: E402
    log_video,
    start_tracking,
    validate_config,
)

__all__ = ["log_video", "start_tracking", "validate_config"]
