from isaaclab.utils import configclass


@configclass
class GroundPlaceCfg:
    size: tuple[float, float] = (1000.0, 1000.0)
    visible : bool = False
    each_track_size: tuple[float, float] = (10, 10) # Note that this is set inside the USD file directly, but we need to use it in our code. If we change the USD file such that the tracks have different size, we need to change this number. I was not able to cleanly read this size from usd file driectly. (sasan, Dec 2025)
    multi_agent_track_size: tuple[float, float] = (40, 40) # this is the size of the track for multi-agent training. It is larger than the single agent track to allow for more room for the cars to interact. This is not set in the USD file and is only used in our code, so it can be changed without changing the USD file. (sasan, Dec 2025)

@configclass
class TrackRigidPropsCfg:
    rigid_body_enabled: bool = True
    max_linear_velocity: float = 0.0
    max_angular_velocity: float = 0.0
    max_depenetration_velocity: float = 0.0
    enable_gyroscopic_forces: bool = False

@configclass
class TrackParamsCfg:
    """All tunable numbers for the ground plane in one place."""
    rigid_props: TrackRigidPropsCfg = TrackRigidPropsCfg()