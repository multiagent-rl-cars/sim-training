from isaaclab.utils import configclass



## NOTE: The settings for the opics is based on the 6mm lens purchased from AMAZON for the FLIR camera.
@configclass
class OverheadCameraOpticsParamsCfg:
    focal_length: float = 0.4 # (in cm) 6mm lens
    focus_distance: float = 40 # (in m) Focus at ground level where the car is. This would not matter if fstop is 0 (no depth of field)
    horizontal_aperture: float = 0.72 # (in cm) This is based on sensor size 1/1.8", which corresponds to horizontal size of 7.2mm and vertical size of 5.4mm
    vertical_aperture: float = 0.54 # (in cm) This is based on sensor size 1/1.8", which corresponds to horizontal size of 7.2mm and vertical size of 5.4mmgoo
    clipping_range: tuple[float, float] = (0.01, 1e6) # objects closer than 0.01m or farther than 1e6m are not rendered
    fstop: float = 0 # disable depth of field, if enabled set to 2.0 and set realistic focal distance




@configclass
class OverheadCameraPoseParamsCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 27.9) # If the size of the track changes, we need to change the height of the camera too. 
    rot: tuple[float, float, float, float] = (1.57, 0.0, 0.0, -0.7071)
    convention: str = "opengl"


@configclass
class OverheadCameraImageParamsCfg:
    width: int = 2048
    height: int = 2048
    data_types: list[str] = ["rgb"]


@configclass
class OverheadCameraParamsCfg:
    """All tunable parameters for the overhead camera."""
    prim_path: str = "/World/OverheadCamera"

    optics: OverheadCameraOpticsParamsCfg = OverheadCameraOpticsParamsCfg()
    pose: OverheadCameraPoseParamsCfg = OverheadCameraPoseParamsCfg()
    image: OverheadCameraImageParamsCfg = OverheadCameraImageParamsCfg()