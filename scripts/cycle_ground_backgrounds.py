"""Cycle the PNG backgrounds over the render mesh of ``/World/GroundPlane``.

Run this file with Isaac Sim 6.0.0's Python launcher, for example::

    /home/rog/Downloads/isaacsim/python.sh scripts/cycle_ground_backgrounds.py

The default paths are resolved relative to this file, so the command can be
run from any working directory.  The USD stage is changed in memory only; the
source ``assets/world.usd`` is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_WORLD = PROJECT_DIR / "assets" / "world.usd"
DEFAULT_BACKGROUNDS_DIR = PROJECT_DIR / "assets" / "backgrounds"
DEFAULT_CAMERA_PATHS = [
    "/World/Camera_01",
    "/World/Camera_02",
    "/World/Camera_03",
    "/World/Camera_04",
]
DEFAULT_FUSION_OUTPUT = PROJECT_DIR / "outputs" / "target_fusion_ground_truth.jsonl"
GROUND_PLANE_PATH = "/World/GroundPlane"
GROUND_MESH_PATH = "/World/GroundPlane/CollisionMesh"
MANNEQUIN_PATH = "/World/Mannequin"
MATERIAL_PATH = "/World/Looks/BackgroundSwapMaterial"
MANNEQUIN_Z_OFFSET = 0.5
SETTLE_SPEED_THRESHOLD = 0.02
SETTLE_STABLE_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply every PNG in a directory to /World/GroundPlane in sequence."
    )
    parser.add_argument(
        "--world",
        type=Path,
        default=DEFAULT_WORLD,
        help=f"USD stage to open (default: {DEFAULT_WORLD})",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=Path,
        default=DEFAULT_BACKGROUNDS_DIR,
        help=f"Directory containing PNG backgrounds (default: {DEFAULT_BACKGROUNDS_DIR})",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=10.0,
        help="Maximum simulated seconds to wait for the mannequin to settle (default: 10.0)",
    )
    parser.add_argument(
        "--camera-prims",
        nargs="+",
        default=DEFAULT_CAMERA_PATHS,
        metavar="CAMERA_PRIM",
        help=(
            "Camera prim paths to validate (default: "
            + " ".join(DEFAULT_CAMERA_PATHS)
            + ")"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible mannequin placement (default: 0)",
    )
    parser.add_argument(
        "--scene-hold-seconds",
        type=float,
        default=2.0,
        help="Seconds to hold each GUI scene after visualization is added (default: 2.0)",
    )
    parser.add_argument(
        "--fusion-output",
        type=Path,
        default=DEFAULT_FUSION_OUTPUT,
        help=f"Ground-truth fusion output path (default: {DEFAULT_FUSION_OUTPUT})",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the Isaac Sim window open after all backgrounds finish",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a GUI; useful for validation or later capture workflows",
    )
    return parser.parse_args()


def find_backgrounds(backgrounds_dir: Path) -> list[Path]:
    """Return PNG files in stable filename order."""
    backgrounds_dir = backgrounds_dir.expanduser().resolve()
    if not backgrounds_dir.is_dir():
        raise FileNotFoundError(f"Background directory does not exist: {backgrounds_dir}")

    backgrounds = sorted(
        (path for path in backgrounds_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name.lower(),
    )
    if not backgrounds:
        raise FileNotFoundError(f"No PNG backgrounds found in: {backgrounds_dir}")
    return backgrounds


def validate_camera_prims(stage, camera_paths: list[str], camera_type) -> None:
    """Fail clearly unless every configured path is a USD camera prim."""
    if not camera_paths:
        raise ValueError("--camera-prims requires at least one camera prim path")

    duplicate_paths = sorted({path for path in camera_paths if camera_paths.count(path) > 1})
    missing_paths = [path for path in camera_paths if not stage.GetPrimAtPath(path).IsValid()]
    non_camera_paths = [
        path
        for path in camera_paths
        if stage.GetPrimAtPath(path).IsValid() and not stage.GetPrimAtPath(path).IsA(camera_type)
    ]

    errors = []
    if duplicate_paths:
        errors.append("duplicate paths: " + ", ".join(duplicate_paths))
    if missing_paths:
        errors.append("missing prims: " + ", ".join(missing_paths))
    if non_camera_paths:
        errors.append("non-camera prims: " + ", ".join(non_camera_paths))
    if errors:
        raise RuntimeError("Camera validation failed; " + "; ".join(errors))


def wait_for_stage(simulation_app, stage_utils) -> None:
    """Wait until the USD stage has finished loading."""
    simulation_app.update()
    simulation_app.update()
    while stage_utils.is_stage_loading() and simulation_app.is_running():
        simulation_app.update()


def settle_mannequin(simulation_app, mannequin, simulation_manager, timeout: float) -> bool:
    """Advance physics until the mannequin is stable or the timeout expires."""
    physics_dt = simulation_manager.get_physics_dt()
    max_steps = max(1, math.ceil(timeout / physics_dt))
    stable_steps_required = max(1, math.ceil(SETTLE_STABLE_SECONDS / physics_dt))
    stable_steps = 0

    for _ in range(max_steps):
        if not simulation_app.is_running():
            return False
        simulation_app.update()
        linear_velocity, angular_velocity = mannequin.get_velocities()
        linear_speed = float(np.linalg.norm(linear_velocity.numpy()[0]))
        angular_speed = float(np.linalg.norm(angular_velocity.numpy()[0]))
        if max(linear_speed, angular_speed) <= SETTLE_SPEED_THRESHOLD:
            stable_steps += 1
            if stable_steps >= stable_steps_required:
                return True
        else:
            stable_steps = 0
    return False


def hold_scene(simulation_app, duration_seconds: float) -> None:
    """Keep the paused GUI scene visible for a wall-clock duration."""
    if duration_seconds <= 0.0:
        return
    deadline = time.monotonic() + duration_seconds
    while simulation_app.is_running() and time.monotonic() < deadline:
        simulation_app.update()


def main() -> None:
    args = parse_args()
    if args.settle_timeout <= 0:
        raise ValueError("--settle-timeout must be greater than zero")
    if args.scene_hold_seconds < 0:
        raise ValueError("--scene-hold-seconds must be nonnegative")

    world_path = args.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"USD stage does not exist: {world_path}")
    backgrounds = find_backgrounds(args.backgrounds_dir)

    # Isaac Sim extension imports must happen after SimulationApp starts.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    fusion_output = None
    try:
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.usd
        from isaacsim.core.experimental.materials import OmniPbrMaterial
        from isaacsim.core.experimental.objects import GroundPlane
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics
        from target_fusion import (
            aim_cameras_at_target,
            build_ground_truth_ray,
            build_ground_truth_record,
            compute_world_target_center,
            draw_ground_truth_rays,
            fuse_rays,
        )

        stage_utils.open_stage(str(world_path))
        wait_for_stage(simulation_app, stage_utils)

        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(GROUND_PLANE_PATH).IsValid():
            raise RuntimeError(
                f"The loaded stage does not contain the required prim: {GROUND_PLANE_PATH}"
            )
        if not stage.GetPrimAtPath(GROUND_MESH_PATH).IsValid():
            raise RuntimeError(f"The loaded stage does not contain the required prim: {GROUND_MESH_PATH}")

        mannequin_prim = stage.GetPrimAtPath(MANNEQUIN_PATH)
        if not mannequin_prim.IsValid():
            raise RuntimeError(f"The loaded stage does not contain the required prim: {MANNEQUIN_PATH}")
        if not mannequin_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Required rigid-body API is missing from {MANNEQUIN_PATH}")
        validate_camera_prims(stage, args.camera_prims, UsdGeom.Camera)

        # Capture the authored pose before physics is initialized. The pose is
        # retained for later randomization passes so only translation changes.
        mannequin = RigidPrim(MANNEQUIN_PATH, reset_xform_op_properties=False)
        authored_positions, authored_orientations = mannequin.get_world_poses()

        # Initialize PhysX once while the timeline remains paused. Playing the
        # timeline later will then advance the already-loaded physics scene.
        if SimulationManager.get_physics_simulation_view() is None:
            SimulationManager.initialize_physics()
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()

        authored_positions = authored_positions.numpy()[0].copy()
        authored_orientations = authored_orientations.numpy()[0].copy()
        ground_range = UsdGeom.Imageable(stage.GetPrimAtPath(GROUND_MESH_PATH)).ComputeWorldBound(
            Usd.TimeCode.Default(), "default"
        ).ComputeAlignedRange()
        ground_min = np.asarray(ground_range.GetMin(), dtype=float)
        ground_max = np.asarray(ground_range.GetMax(), dtype=float)
        ground_center = (ground_min + ground_max) * 0.5
        ground_half_extent = (ground_max - ground_min) * 0.25
        randomizer = random.Random(args.seed)

        # templates=None preserves any existing template while wrapping the
        # supplied composite ground plane. Bind to its render mesh only.
        ground_plane = GroundPlane(GROUND_PLANE_PATH, templates=None)
        material = OmniPbrMaterial(MATERIAL_PATH)
        material.set_input_values("diffuse_color_constant", [1.0, 1.0, 1.0])

        # CollisionMesh already has normalized authored UVs:
        # (0, 0), (1, 0), (1, 1), (0, 1). Keep those UVs so one image spans
        # the complete square plane. Projected UVW mapping uses the plane's
        # world/object dimensions instead, which makes the texture tile.
        material.set_input_values("project_uvw", False)
        material.set_input_values("texture_scale", [1.0, 1.0])
        material.set_input_values("texture_translate", [0.0, 0.0])
        ground_plane.meshes.apply_visual_materials(material)

        print(f"Loaded stage: {world_path}")
        print(f"Found {len(backgrounds)} PNG background(s) in {args.backgrounds_dir.resolve()}")
        print(f"Validated camera prim(s): {', '.join(args.camera_prims)}")
        fusion_output_path = args.fusion_output.expanduser().resolve()
        if fusion_output_path == world_path or fusion_output_path in backgrounds:
            raise ValueError("--fusion-output must not overwrite the USD stage or a background image")
        fusion_output_path.parent.mkdir(parents=True, exist_ok=True)
        fusion_output = fusion_output_path.open("w", encoding="utf-8")
        print(f"Writing fusion ground truth: {fusion_output_path}")

        for scene_index, background_path in enumerate(backgrounds):
            material.set_input_values("diffuse_texture", str(background_path))
            random_position = authored_positions.copy()
            random_position[0] = randomizer.uniform(
                ground_center[0] - ground_half_extent[0], ground_center[0] + ground_half_extent[0]
            )
            random_position[1] = randomizer.uniform(
                ground_center[1] - ground_half_extent[1], ground_center[1] + ground_half_extent[1]
            )
            random_position[2] = authored_positions[2] + MANNEQUIN_Z_OFFSET
            mannequin.set_world_poses(
                positions=[random_position],
                orientations=[authored_orientations],
            )
            mannequin.set_velocities(
                linear_velocities=np.zeros((1, 3), dtype=np.float32),
                angular_velocities=np.zeros((1, 3), dtype=np.float32),
            )
            print(f"[background] {background_path.name} at ({random_position[0]:.3f}, "
                  f"{random_position[1]:.3f}, {random_position[2]:.3f})")
            simulation_app.update()  # Flush material and pose edits before playing.
            timeline.play()
            settled = settle_mannequin(simulation_app, mannequin, SimulationManager, args.settle_timeout)
            timeline.pause()
            if not settled:
                print(f"[warning] Mannequin did not settle before timeout for {background_path.name}")
            if not simulation_app.is_running():
                return

            target_world = compute_world_target_center(stage, MANNEQUIN_PATH)
            camera_aims = aim_cameras_at_target(
                stage,
                args.camera_prims,
                target_world,
            )
            rays = [
                build_ground_truth_ray(
                    camera_aim["camera_path"],
                    camera_aim["position_world"],
                    target_world,
                )
                for camera_aim in camera_aims
            ]
            fusion_result = fuse_rays(rays, target_world=target_world)
            simulation_app.update()  # Flush camera orientation edits before drawing.
            if not args.headless:
                draw_ground_truth_rays(rays, target_world)

            fusion_error = (
                "n/a" if fusion_result.error_m is None else f"{fusion_result.error_m:.6g}m"
            )
            residual = (
                "n/a"
                if fusion_result.rms_residual_m is None
                else f"{fusion_result.rms_residual_m:.6g}m"
            )
            print(
                f"[fusion] scene={scene_index} background={background_path.name} "
                f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f}) "
                f"valid={fusion_result.valid} error={fusion_error} residual={residual}"
            )
            record = build_ground_truth_record(
                scene_index=scene_index,
                background_path=str(background_path),
                target_prim_path=MANNEQUIN_PATH,
                target_world=target_world,
                mannequin_position_world=random_position,
                camera_aims=camera_aims,
                rays=rays,
                fusion_result=fusion_result,
                settled=settled,
            )
            fusion_output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            fusion_output.flush()
            if not args.headless:
                hold_scene(simulation_app, args.scene_hold_seconds)
            if not simulation_app.is_running():
                return

        if args.keep_open and not args.headless:
            print("Finished cycling; keeping the Isaac Sim window open.")
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        if fusion_output is not None:
            fusion_output.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
