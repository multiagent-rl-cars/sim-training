"""Conservative generated-asset overrides shared by GPU checks and experiments.

Import only after AppLauncher has started Isaac Sim.
"""


def configure_fixture(cfg, task):
    if task == "Multiagent-Car-Training-v0":
        from car_training.tasks.manager_based.car_training.multiagent_training_env_cfg import ASSET_ROOT, OverheadCameraCfg
        cfg.scene.track.spawn.usd_path = str(ASSET_ROOT / "scene_assets/tracks_thickness_random/track_0.usd")
        cfg.scene.ground_plane.ground_plane_params.multi_agent_track_size = (10, 10)
        cfg.scene.env_spacing = 0.0
        cfg.scene.filter_collisions = False
        cfg.events.reset_robot_position_random_marl.params.update(z=0.11, min_spawn_distance=1.5)
    else:
        from car_training.tasks.manager_based.car_training.car_training_env_cfg import ASSET_ROOT, OverheadCameraCfg
    if not (ASSET_ROOT / "fixture.json").is_file():
        raise RuntimeError("--fixture requires CAR_TRAINING_ASSET_ROOT pointing at generated test assets")
    # These settings belong only to the fixture, not the inherited real-car model.
    cfg.scene.track.spawn.rigid_props = None
    cfg.sim.dt = 1 / 120
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.scene.robot.init_state.pos = (0.0, -3.3, 0.11)
    cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.scene.robot.actuators["throttle"].effort_limit = 2.0
    cfg.scene.robot.actuators["throttle"].damping = 0.1
    cfg.scene.robot.actuators["steering"].effort_limit = 2.0
    cfg.scene.robot.actuators["steering"].stiffness = 10.0
    cfg.scene.robot.actuators["steering"].damping = 1.0
    camera = OverheadCameraCfg()
    camera.offset.pos = (0.0, 0.0, 9.4)
    camera.width = 512
    camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
    # Match the hard-coded 10x10m image-to-world mapping.
    camera.height = camera.width
    camera.spawn.focal_length = 24.0
    camera.spawn.horizontal_aperture = 10 * camera.spawn.focal_length / camera.offset.pos[2]
    cfg.scene.overhead_camera = camera
