"""Generate self-contained USD fixtures, NOT calibrated Nano-TTR assets.

Run with `uv run --group assets python scripts/generate_test_assets.py`.
No Isaac runtime or external meshes/textures are required.
"""
import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def new_stage(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("comment", "Procedural test fixture; not a calibrated physical-car model")
    return stage


def cube(stage, path, position, size, color, mass=None, collision=True):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3d(*size))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    if mass is not None:
        UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
        UsdPhysics.MassAPI.Apply(shape.GetPrim()).CreateMassAttr(mass)
    return shape


def body(stage, path, position, mass):
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(mass)
    return prim


def joint(stage, name, parent, child, anchor_parent, axis):
    j = UsdPhysics.RevoluteJoint.Define(stage, f"/Car/Joints/{name}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*anchor_parent))
    j.CreateLocalPos1Attr(Gf.Vec3f(0))
    j.CreateAxisAttr(axis)
    j.CreateCollisionEnabledAttr(False)
    if axis == "Z":
        j.CreateLowerLimitAttr(-40.0)
        j.CreateUpperLimitAttr(40.0)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)


def car(path):
    stage = new_stage(path, "/Car")
    chassis = "/Car/Rigid_Bodies/Chassis"
    body(stage, chassis, (0, 0, 0.10), 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(chassis))
    cube(stage, chassis + "/Visual", (0, 0, 0), (0.32, 0.16, 0.07), (0.05, 0.8, 0.05))
    for end, x in [("Front", 0.13), ("Rear", -0.13)]:
        for side, y in [("Left", 0.115), ("Right", -0.115)]:
            wheel_path = f"/Car/Rigid_Bodies/Wheel_{end}_{side}"
            wheel_pos = (x, y, 0.065)
            body(stage, wheel_path, wheel_pos, 0.08)
            shape = UsdGeom.Cylinder.Define(stage, wheel_path + "/Tire")
            shape.CreateAxisAttr("Y")
            shape.CreateRadiusAttr(0.065)
            shape.CreateHeightAttr(0.035)
            shape.CreateDisplayColorAttr([Gf.Vec3f(0.12)])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            parent = chassis
            anchor = (x, y, -0.035)
            if end == "Front":
                parent = f"/Car/Rigid_Bodies/Knuckle_{side}"
                body(stage, parent, wheel_pos, 0.02)
                # Small collision-free geometry supplies a finite inertia tensor.
                cube(stage, parent + "/Visual", (0, 0, 0), (0.02, 0.02, 0.02), (0.4, 0.4, 0.4), collision=False)
                joint(stage, f"Knuckle__Upright__Front_{side}", chassis, parent, anchor, "Z")
                anchor = (0, 0, 0)
            middle = "Knuckle" if end == "Front" else "Upright"
            joint(stage, f"Wheel__{middle}__{end}_{side}", parent, wheel_path, anchor, "Y")
    stage.GetRootLayer().Save()


def track(path):
    stage = new_stage(path, "/Track")
    cube(stage, "/Track/Base", (0, 0, -0.015), (10, 10, 0.03), (0.95, 0.95, 0.95))
    # Thin black circular line matches the inherited contour-based observations.
    vertices = []
    faces = []
    segments = 256
    for i in range(segments):
        a = 2 * math.pi * i / segments
        for radius in (3.25, 3.35):
            vertices.append((radius * math.cos(a), radius * math.sin(a), 0.002))
        j = (i + 1) % segments
        faces.extend([2*i, 2*i+1, 2*j+1, 2*j])
    mesh = UsdGeom.Mesh.Define(stage, "/Track/Line")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([4] * segments)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0)])
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated_assets"))
    args = parser.parse_args()
    root = args.output.resolve()
    # Refuse to overwrite assets, including originals placed at a custom path.
    if root.exists() and any(root.iterdir()):
        parser.error(f"Output is not empty: {root}. Choose a new output directory.")
    car(root / "robot_assets/leatherback_simple_better.usd")
    for i in range(12):
        track(root / f"scene_assets/tracks_thickness_random/track_{i}.usd")
    (root / "fixture.json").write_text(json.dumps({
        "kind": "procedural-smoke-test-only", "version": 1,
        "track": "12 identical circular tracks, radius 3.3m, 10x10m floor",
        "car": "six joints, wheel radius 0.065m, wheelbase 0.26m, track width 0.23m",
        "limitations": "Uncalibrated geometry, mass, friction and actuation. Not valid sim-to-real training data."
    }, indent=2) + "\n")
    print(f"Generated test fixtures at {root}")
    print(f"export CAR_TRAINING_ASSET_ROOT='{root}'")


if __name__ == "__main__":
    main()
