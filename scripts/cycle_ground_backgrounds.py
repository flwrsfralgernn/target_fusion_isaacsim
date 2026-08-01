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
DEFAULT_SCHEMA_V2_OUTPUT = PROJECT_DIR / "outputs" / "target_fusion_bbox_v2.jsonl"
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_DIR / "outputs" / "target_fusion_bbox_v2_images"
DEFAULT_RAW_OUTPUT_DIR = PROJECT_DIR / "outputs" / "sdg_raw"
RAW_FRAME_PADDING = 6
GROUND_PLANE_PATH = "/World/GroundPlane"
GROUND_MESH_PATH = "/World/GroundPlane/CollisionMesh"
MANNEQUIN_PATH = "/World/Mannequin"
MATERIAL_PATH = "/World/Looks/BackgroundSwapMaterial"
MANNEQUIN_Z_OFFSET = 0.5
SETTLE_SPEED_THRESHOLD = 0.02
SETTLE_STABLE_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synchronized multi-camera mannequin captures over PNG backgrounds."
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
        "--frames",
        type=int,
        default=None,
        help=(
            "Number of captures to generate; backgrounds repeat in stable order when this exceeds "
            "the number of PNGs (default: one capture per background)"
        ),
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
        "--resolution",
        type=int,
        nargs=2,
        default=(640, 480),
        metavar=("WIDTH", "HEIGHT"),
        help="Render-product resolution as width height (default: 640 480)",
    )
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=1,
        help="Replicator render subframes per synchronized capture (default: 1)",
    )
    parser.add_argument(
        "--target-label",
        default="mannequin",
        help="Semantic class label selected from bbox annotations (default: mannequin)",
    )
    parser.add_argument(
        "--max-occlusion-ratio",
        type=float,
        default=None,
        help="Reject target boxes above this occlusion ratio (default: no ratio filter)",
    )
    parser.add_argument(
        "--bbox-border-tolerance-px",
        type=float,
        default=0.0,
        help="Treat boxes within this many pixels of an image edge as clipped (default: 0)",
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
        default=None,
        help=(
            "Optional compatibility schema-v1 output path; disabled by default so the "
            f"existing baseline at {DEFAULT_FUSION_OUTPUT} is preserved"
        ),
    )
    parser.add_argument(
        "--schema-v2-output",
        type=Path,
        default=DEFAULT_SCHEMA_V2_OUTPUT,
        help=f"Schema-v2 bbox fusion output path (default: {DEFAULT_SCHEMA_V2_OUTPUT})",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT_DIR,
        help=f"Directory for annotated camera images (default: {DEFAULT_IMAGE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_DIR,
        help=(
            "Directory for Isaac BasicWriter RGB/bbox/camera outputs (default: "
            f"{DEFAULT_RAW_OUTPUT_DIR})"
        ),
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


def label_mannequin_for_bbox(stage, mannequin_prim, imageable_type, *, target_label: str) -> None:
    """Clear stage semantics and label every imageable mannequin prim."""
    from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
    from pxr import Usd

    for prim in stage.Traverse():
        if prim.IsA(imageable_type):
            remove_all_labels(prim)
    labeled_count = 0
    for prim in Usd.PrimRange(mannequin_prim):
        if prim.IsA(imageable_type):
            add_labels(prim, labels=[target_label], taxonomy="class")
            labeled_count += 1
    if labeled_count == 0:
        raise RuntimeError(f"No imageable mannequin prims found under {mannequin_prim.GetPath()}")


def read_bbox_annotator(annotator):
    """Normalize Replicator bbox annotator payloads to data rows and metadata."""
    payload = annotator.get_data()
    if payload is None:
        return None, {}
    if isinstance(payload, dict):
        return payload.get("data"), payload.get("info", payload)
    return payload, {}


def read_rgb_annotator(annotator):
    """Normalize an RGB annotator payload to its image array."""
    payload = annotator.get_data()
    if isinstance(payload, dict):
        return payload.get("data")
    return payload


def set_render_product_updates(render_products, enabled: bool) -> None:
    """Enable or disable Hydra updates for every attached render product."""
    for render_product in render_products:
        hydra_texture = getattr(render_product, "hydra_texture", None)
        if hydra_texture is None:
            raise RuntimeError("Render product has no hydra_texture interface")
        hydra_texture.set_updates_enabled(bool(enabled))


def basic_writer_capture_paths(
    raw_output_dir: Path,
    render_product_name: str,
    frame_index: int,
    *,
    frame_padding: int = RAW_FRAME_PADDING,
) -> dict[str, Path]:
    """Return the files BasicWriter creates for one multi-render-product frame.

    With ``use_common_output_dir=True``, BasicWriter prefixes each annotator
    file with the render-product name.  Keeping this naming contract in one
    helper lets the schema and the later YOLO exporter refer to clean RGB and
    Isaac-native bbox files without scanning ambiguous directories.
    """
    raw_output_dir = Path(raw_output_dir).expanduser().resolve()
    render_product_name = str(render_product_name)
    if not render_product_name or Path(render_product_name).name != render_product_name:
        raise ValueError("render_product_name must be a non-empty filename-safe name")
    if int(frame_index) < 0:
        raise ValueError("frame_index must be nonnegative")
    if int(frame_padding) <= 0:
        raise ValueError("frame_padding must be positive")

    frame = f"{int(frame_index):0{int(frame_padding)}d}"
    prefix = f"{render_product_name}_"
    return {
        "rgb": raw_output_dir / "rgb" / f"{prefix}rgb_{frame}.png",
        "bbox": (
            raw_output_dir
            / "bounding_box_2d_tight"
            / f"{prefix}bounding_box_2d_tight_{frame}.npy"
        ),
        "bbox_labels": (
            raw_output_dir
            / "bounding_box_2d_tight"
            / f"{prefix}bounding_box_2d_tight_labels_{frame}.json"
        ),
        "bbox_prim_paths": (
            raw_output_dir
            / "bounding_box_2d_tight"
            / f"{prefix}bounding_box_2d_tight_prim_paths_{frame}.json"
        ),
        "camera_params": (
            raw_output_dir
            / "camera_params"
            / f"{prefix}camera_params_{frame}.json"
        ),
    }


def initialize_basic_writer(rep, render_products, raw_output_dir: Path):
    """Attach one Isaac BasicWriter to every render product."""
    if not render_products:
        raise RuntimeError("BasicWriter requires at least one render product")
    raw_output_dir = Path(raw_output_dir).expanduser().resolve()
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(raw_output_dir))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        bounding_box_2d_tight=True,
        camera_params=True,
        frame_padding=RAW_FRAME_PADDING,
        use_common_output_dir=True,
    )
    writer.attach(render_products)
    return writer


def capture_synchronized_camera_views(
    orchestrator,
    bbox_annotators,
    rgb_annotators,
    *,
    render_products=None,
    rt_subframes: int,
    render_resolution,
    target_label: str,
    max_occlusion_ratio,
    border_tolerance_px: float,
):
    """Capture and read every configured camera at one Replicator step.

    Render products are attached before this function is called and remain
    enabled for the lifetime of the writer. Replicator captures all attached
    products during one orchestrator step, so the loops below only consume the
    synchronized buffers and never trigger a camera-specific capture.
    """
    try:
        from target_fusion import extract_target_bbox
    except ModuleNotFoundError:
        # The capture script is normally launched directly by Isaac Sim, while
        # the pure-Python tests import it as ``scripts.cycle_ground_backgrounds``.
        from scripts.target_fusion import extract_target_bbox

    if len(bbox_annotators) != len(rgb_annotators):
        raise RuntimeError(
            "RGB and bbox annotator counts must match before synchronized capture"
        )
    if not bbox_annotators:
        raise RuntimeError("Synchronized capture requires at least one camera")

    render_products = [] if render_products is None else list(render_products)
    if render_products and len(render_products) != len(bbox_annotators):
        raise RuntimeError(
            "Render-product and annotator counts must match before synchronized capture"
        )

    # Keep Hydra updates enabled between steps.  BasicWriter consumes the
    # render-product data asynchronously after the step returns; disabling a
    # product here can leave the writer with only the first frame even though
    # the in-memory annotators appear to have captured successfully.
    orchestrator.step(
        delta_time=0.0,
        rt_subframes=rt_subframes,
        pause_timeline=True,
        wait_for_render=True,
    )

    target_bboxes = []
    for annotator in bbox_annotators:
        annotation_data, annotation_info = read_bbox_annotator(annotator)
        target_bboxes.append(
            extract_target_bbox(
                annotation_data,
                annotation_info,
                resolution=render_resolution,
                target_label=target_label,
                max_occlusion_ratio=max_occlusion_ratio,
                border_tolerance_px=border_tolerance_px,
            )
        )
    rgb_frames = [read_rgb_annotator(annotator) for annotator in rgb_annotators]
    return target_bboxes, rgb_frames


def save_annotated_capture_image(
    rgb_data,
    bbox,
    output_path: Path,
    *,
    target_label: str,
) -> None:
    """Save one RGB capture with its selected bbox and validity annotation."""
    from PIL import Image, ImageDraw

    if rgb_data is None:
        raise RuntimeError(f"RGB annotator returned no image for {output_path.name}")
    image_array = np.asarray(rgb_data)
    if image_array.ndim != 3 or image_array.shape[2] not in (3, 4):
        raise RuntimeError(
            f"RGB annotator returned unsupported shape {image_array.shape} for {output_path.name}"
        )
    if not np.all(np.isfinite(image_array)):
        raise RuntimeError(f"RGB annotator returned nonfinite pixels for {output_path.name}")
    if np.issubdtype(image_array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image_array)) <= 1.0 else 1.0
        image_array = np.clip(image_array * scale, 0.0, 255.0).astype(np.uint8)
    elif image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    if bbox.valid:
        bounds = [
            int(round(bbox.x_min)),
            int(round(bbox.y_min)),
            int(round(bbox.x_max)),
            int(round(bbox.y_max)),
        ]
        draw.rectangle(bounds, outline=(40, 255, 40), width=3)
        label = f"{target_label} ({bbox.center_uv[0]:.1f}, {bbox.center_uv[1]:.1f})"
        draw.text((max(0, bounds[0]), max(0, bounds[1] - 16)), label, fill=(40, 255, 40))
    else:
        label = f"{target_label}: INVALID - {bbox.reason or 'unknown reason'}"
        draw.text((8, 8), label, fill=(255, 60, 60))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def build_camera_calibration(stage, camera_path: str, resolution):
    """Snapshot USD pinhole calibration and camera-to-world rotation."""
    from pxr import Gf, Usd, UsdGeom
    from target_fusion import CameraCalibration

    prim = stage.GetPrimAtPath(camera_path)
    camera = UsdGeom.Camera(prim)
    focal_length = camera.GetFocalLengthAttr().Get()
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    vertical_aperture = camera.GetVerticalApertureAttr().Get()
    projection = camera.GetProjectionAttr().Get()
    if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
        raise RuntimeError(f"Camera {camera_path} is missing pinhole calibration attributes")

    horizontal_offset = camera.GetHorizontalApertureOffsetAttr().Get()
    vertical_offset = camera.GetVerticalApertureOffsetAttr().Get()
    world_matrix = Gf.Matrix4d(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    rotation = np.column_stack(
        [
            np.asarray(list(world_matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))), dtype=np.float64),
            np.asarray(list(world_matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))), dtype=np.float64),
            np.asarray(list(world_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))), dtype=np.float64),
        ]
    )
    rotation = rotation / np.linalg.norm(rotation, axis=0, keepdims=True)
    translation = np.asarray(list(world_matrix.ExtractTranslation()), dtype=np.float64)
    return CameraCalibration(
        camera_path=camera_path,
        resolution=tuple(resolution),
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        vertical_aperture=float(vertical_aperture),
        horizontal_aperture_offset=0.0 if horizontal_offset is None else float(horizontal_offset),
        vertical_aperture_offset=0.0 if vertical_offset is None else float(vertical_offset),
        projection=str(projection),
        origin_world=translation,
        rotation_world_from_camera=rotation,
    )


def main() -> None:
    args = parse_args()
    if args.settle_timeout <= 0:
        raise ValueError("--settle-timeout must be greater than zero")
    if args.scene_hold_seconds < 0:
        raise ValueError("--scene-hold-seconds must be nonnegative")
    if args.frames is not None and args.frames <= 0:
        raise ValueError("--frames must be greater than zero")
    if len(args.camera_prims) != 4:
        raise ValueError("--camera-prims must contain exactly four cameras")
    if len(args.resolution) != 2 or any(value <= 0 for value in args.resolution):
        raise ValueError("--resolution must contain two positive dimensions")
    if args.rt_subframes <= 0:
        raise ValueError("--rt-subframes must be greater than zero")
    if args.max_occlusion_ratio is not None and not 0.0 <= args.max_occlusion_ratio <= 1.0:
        raise ValueError("--max-occlusion-ratio must be between 0 and 1")
    if args.bbox_border_tolerance_px < 0.0:
        raise ValueError("--bbox-border-tolerance-px must be nonnegative")

    world_path = args.world.expanduser().resolve()
    if not world_path.is_file():
        raise FileNotFoundError(f"USD stage does not exist: {world_path}")
    backgrounds = find_backgrounds(args.backgrounds_dir)

    # Isaac Sim extension imports must happen after SimulationApp starts.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    schema_v2_output = None
    legacy_output = None
    raw_manifest_output = None
    writer = None
    render_products = []
    bbox_annotators = []
    rgb_annotators = []
    render_product_names = []
    try:
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.experimental.materials import OmniPbrMaterial
        from isaacsim.core.experimental.objects import GroundPlane
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics
        from target_fusion import (
            CameraObservation,
            FusionResult,
            aim_cameras_at_target,
            build_rays_from_available_observations,
            build_ground_truth_record,
            build_schema_v2_record,
            clear_debug_draw,
            compute_world_target_center,
            DEFAULT_GROUND_TRUTH_RAY_COLOR,
            draw_fused_rays,
            evaluate_fusion,
            fuse_rays,
            get_camera_world_pose,
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
        label_mannequin_for_bbox(
            stage,
            mannequin_prim,
            UsdGeom.Imageable,
            target_label=args.target_label,
        )

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

        # Establish fixed camera poses once. The target height comes from the
        # authored mannequin bounds plus the constant scene translation; the
        # per-scene mannequin position is never used to re-aim cameras.
        authored_target_center = compute_world_target_center(stage, MANNEQUIN_PATH)
        fixed_camera_target = np.array(
            [ground_center[0], ground_center[1], authored_target_center[2] + MANNEQUIN_Z_OFFSET],
            dtype=np.float64,
        )
        fixed_camera_aims = aim_cameras_at_target(
            stage,
            args.camera_prims,
            fixed_camera_target,
        )
        simulation_app.update()
        camera_calibrations = [
            build_camera_calibration(stage, camera_path, tuple(args.resolution))
            for camera_path in args.camera_prims
        ]

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

        rep.orchestrator.set_capture_on_play(False)
        render_resolution = tuple(args.resolution)
        for camera_index, camera_path in enumerate(args.camera_prims):
            render_product_name = f"TargetFusion_Camera_{camera_index + 1:02d}"
            render_product = rep.create.render_product(
                camera_path,
                render_resolution,
                name=render_product_name,
            )
            annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            annotator.attach(render_product)
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach(render_product)
            render_products.append(render_product)
            bbox_annotators.append(annotator)
            rgb_annotators.append(rgb_annotator)
            render_product_names.append(render_product_name)
        simulation_app.update()

        print(f"Loaded stage: {world_path}")
        print(f"Found {len(backgrounds)} PNG background(s) in {args.backgrounds_dir.resolve()}")
        print(f"Validated camera prim(s): {', '.join(args.camera_prims)}")
        schema_v2_output_path = args.schema_v2_output.expanduser().resolve()
        output_paths = [schema_v2_output_path]
        if args.fusion_output is not None:
            output_paths.append(args.fusion_output.expanduser().resolve())
        if any(path == world_path or path in backgrounds for path in output_paths):
            raise ValueError("fusion output paths must not overwrite the USD stage or a background image")
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("schema-v2 and compatibility output paths must be different")
        image_output_dir = args.image_output_dir.expanduser().resolve()
        if image_output_dir == world_path or image_output_dir in backgrounds:
            raise ValueError("--image-output-dir must not overwrite the USD stage or a background image")
        raw_output_dir = args.raw_output_dir.expanduser().resolve()
        if raw_output_dir == world_path or raw_output_dir in backgrounds:
            raise ValueError("--raw-output-dir must not overwrite the USD stage or a background image")
        if raw_output_dir == image_output_dir:
            raise ValueError("--raw-output-dir and --image-output-dir must be different directories")
        image_output_dir.mkdir(parents=True, exist_ok=True)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        schema_v2_output_path.parent.mkdir(parents=True, exist_ok=True)
        schema_v2_output = schema_v2_output_path.open("w", encoding="utf-8")
        raw_manifest_path = raw_output_dir / "manifest.jsonl"
        raw_manifest_output = raw_manifest_path.open("w", encoding="utf-8")
        writer = initialize_basic_writer(rep, render_products, raw_output_dir)
        print(f"Writing schema-v2 fusion output: {schema_v2_output_path}")
        print(f"Writing Isaac BasicWriter raw outputs: {raw_output_dir}")
        print(f"Writing raw capture manifest: {raw_manifest_path}")
        if args.fusion_output is not None:
            legacy_output_path = output_paths[1]
            legacy_output_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_output = legacy_output_path.open("w", encoding="utf-8")
            print(f"Writing optional schema-v1 compatibility output: {legacy_output_path}")
        print(f"Writing annotated camera images: {image_output_dir}")

        capture_count = len(backgrounds) if args.frames is None else args.frames
        for scene_index in range(capture_count):
            background_path = backgrounds[scene_index % len(backgrounds)]
            # Cycle boundary: remove visualization from the previous capture
            # before changing the scene for the next one.
            if not args.headless:
                clear_debug_draw()

            # 1-2. Randomize the background and mannequin pose while paused.
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
            for fixed_camera_aim in fixed_camera_aims:
                actual_position, actual_forward = get_camera_world_pose(
                    stage,
                    fixed_camera_aim["camera_path"],
                )
                if np.linalg.norm(actual_position - fixed_camera_aim["position_world"]) > 1e-6:
                    raise RuntimeError(
                        f"Fixed camera {fixed_camera_aim['camera_path']} moved during the scene"
                    )
                if float(np.dot(actual_forward, fixed_camera_aim["forward_world"])) < 0.999999:
                    raise RuntimeError(
                        f"Fixed camera {fixed_camera_aim['camera_path']} changed orientation during the scene"
                    )

            # 3-4. Fire one synchronized capture for every attached camera and
            # leave the timeline paused while the resulting rays are inspected.
            target_bboxes, rgb_frames = capture_synchronized_camera_views(
                rep.orchestrator,
                bbox_annotators,
                rgb_annotators,
                render_products=render_products,
                rt_subframes=args.rt_subframes,
                render_resolution=render_resolution,
                target_label=args.target_label,
                max_occlusion_ratio=args.max_occlusion_ratio,
                border_tolerance_px=args.bbox_border_tolerance_px,
            )
            if len(rgb_frames) != len(target_bboxes):
                raise RuntimeError(
                    "RGB annotator count does not match bbox annotator count for synchronized capture"
                )
            image_paths = []
            for camera_index, (rgb_frame, bbox) in enumerate(zip(rgb_frames, target_bboxes)):
                image_path = image_output_dir / (
                    f"scene_{scene_index:04d}_camera_{camera_index + 1:02d}.png"
                )
                save_annotated_capture_image(
                    rgb_frame,
                    bbox,
                    image_path,
                    target_label=args.target_label,
                )
                image_paths.append(str(image_path))
            raw_capture_paths = [
                basic_writer_capture_paths(raw_output_dir, render_product_name, scene_index)
                for render_product_name in render_product_names
            ]
            raw_image_paths = [str(paths["rgb"]) for paths in raw_capture_paths]
            raw_bbox_paths = [str(paths["bbox"]) for paths in raw_capture_paths]
            raw_camera_params_paths = [
                str(paths["camera_params"]) for paths in raw_capture_paths
            ]
            observations = [
                CameraObservation(
                    camera_path=camera_path,
                    calibration=calibration,
                    bbox=bbox,
                    capture_id=scene_index,
                )
                for camera_path, calibration, bbox in zip(
                    args.camera_prims,
                    camera_calibrations,
                    target_bboxes,
                )
            ]
            # A camera may miss the mannequin while the other cameras see it.
            # Keep that observation invalid, but continue constructing rays
            # from every available camera in this synchronized capture.
            rays = build_rays_from_available_observations(observations)

            partial_fusion = fuse_rays(rays)
            valid_observation_count = sum(observation.valid for observation in observations)
            if valid_observation_count != len(args.camera_prims):
                fusion_result = FusionResult(
                    fused_position_world=partial_fusion.fused_position_world,
                    rms_residual_m=partial_fusion.rms_residual_m,
                    rank=partial_fusion.rank,
                    condition_number=partial_fusion.condition_number,
                    valid=False,
                    reason=(
                        "required four valid camera observations; "
                        f"got {valid_observation_count}"
                    ),
                    ray_diagnostics=partial_fusion.ray_diagnostics,
                    pairwise_angles_deg=partial_fusion.pairwise_angles_deg,
                )
            else:
                fusion_result = partial_fusion
            camera_aims = [
                fixed_camera_aim
                for fixed_camera_aim, observation in zip(fixed_camera_aims, observations)
                if observation.valid
            ]
            fusion_evaluation = evaluate_fusion(fusion_result, target_world)
            simulation_app.update()  # Flush camera orientation edits before drawing.
            if not args.headless and rays and fusion_result.valid:
                draw_fused_rays(
                    rays,
                    fusion_result.fused_position_world,
                    camera_colors=(DEFAULT_GROUND_TRUTH_RAY_COLOR,),
                    truth_world=target_world,
                )

            fusion_error = (
                "n/a"
                if fusion_evaluation.error_m is None
                else f"{fusion_evaluation.error_m:.6g}m"
            )
            residual = (
                "n/a"
                if fusion_result.rms_residual_m is None
                else f"{fusion_result.rms_residual_m:.6g}m"
            )
            print(
                f"[fusion] scene={scene_index} background={background_path.name} "
                f"target=({target_world[0]:.3f}, {target_world[1]:.3f}, {target_world[2]:.3f}) "
                f"valid={fusion_result.valid} error={fusion_error} residual={residual} "
                f"bbox_valid={valid_observation_count}/{len(observations)}"
            )
            schema_v2_record = build_schema_v2_record(
                scene_index=scene_index,
                capture_id=scene_index,
                background_path=str(background_path),
                target_prim_path=MANNEQUIN_PATH,
                resolution=render_resolution,
                target_label=args.target_label,
                rt_subframes=args.rt_subframes,
                observations=observations,
                rays=rays,
                fusion_result=fusion_result,
                fusion_evaluation=fusion_evaluation,
                settled=settled,
                image_paths=image_paths,
                raw_image_paths=raw_image_paths,
                raw_bbox_paths=raw_bbox_paths,
                raw_camera_params_paths=raw_camera_params_paths,
            )
            schema_v2_output.write(
                json.dumps(schema_v2_record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            schema_v2_output.flush()
            raw_manifest_record = {
                "manifest_version": 1,
                "capture_id": scene_index,
                "scene_index": scene_index,
                "background_path": str(background_path),
                "target_prim_path": MANNEQUIN_PATH,
                "target_label": args.target_label,
                "resolution": list(render_resolution),
                "rt_subframes": args.rt_subframes,
                "settled": bool(settled),
                "cameras": [
                    {
                        "camera_path": observation.camera_path,
                        "render_product_name": render_product_name,
                        "rgb_path": str(raw_paths["rgb"]),
                        "bbox_path": str(raw_paths["bbox"]),
                        "bbox_labels_path": str(raw_paths["bbox_labels"]),
                        "bbox_prim_paths_path": str(raw_paths["bbox_prim_paths"]),
                        "camera_params_path": str(raw_paths["camera_params"]),
                        "annotated_image_path": image_paths[camera_index],
                        "bbox": observation.bbox.as_dict()
                        if observation.bbox is not None
                        else None,
                    }
                    for camera_index, (observation, render_product_name, raw_paths) in enumerate(
                        zip(observations, render_product_names, raw_capture_paths)
                    )
                ],
            }
            raw_manifest_output.write(
                json.dumps(raw_manifest_record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            raw_manifest_output.flush()
            if legacy_output is not None:
                legacy_record = build_ground_truth_record(
                    scene_index=scene_index,
                    background_path=str(background_path),
                    target_prim_path=MANNEQUIN_PATH,
                    target_world=target_world,
                    mannequin_position_world=random_position,
                    camera_aims=camera_aims,
                    rays=rays,
                    fusion_result=fusion_result,
                    fusion_evaluation=fusion_evaluation,
                    settled=settled,
                )
                legacy_record["bbox_observations"] = [
                    observation.as_dict() for observation in observations
                ]
                legacy_record["fusion_source"] = "bounding_box_centers"
                legacy_record["image_paths"] = image_paths
                legacy_output.write(
                    json.dumps(legacy_record, separators=(",", ":"), allow_nan=False) + "\n"
                )
                legacy_output.flush()
            if not args.headless:
                hold_scene(simulation_app, args.scene_hold_seconds)
                # 5. End the cycle with a clean viewport for the next scene.
                clear_debug_draw()
            if not simulation_app.is_running():
                return

        if args.keep_open and not args.headless:
            print("Finished cycling; keeping the Isaac Sim window open.")
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        if not args.headless:
            try:
                clear_debug_draw()
            except Exception:
                pass
        if schema_v2_output is not None:
            schema_v2_output.close()
        if legacy_output is not None:
            legacy_output.close()
        if raw_manifest_output is not None:
            raw_manifest_output.close()
        if writer is not None:
            try:
                rep.orchestrator.wait_until_complete()
            except Exception:
                pass
            try:
                writer.detach()
            except Exception:
                pass
        try:
            set_render_product_updates(render_products, False)
        except Exception:
            pass
        for annotator in bbox_annotators:
            try:
                annotator.detach()
            except Exception:
                pass
        for annotator in rgb_annotators:
            try:
                annotator.detach()
            except Exception:
                pass
        for render_product in render_products:
            try:
                render_product.destroy()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
