"""Run a five-trial, prediction-first interactive YOLO demonstration.

Launch with Isaac Sim's Python environment::

    /home/rog/Downloads/isaacsim/python.sh scripts/demo_yolo_prediction_first.py

Each trial shows YOLO rays first, waits for Enter in the launching terminal,
then reveals Isaac ground truth and waits again before advancing.
"""

from __future__ import annotations

import argparse
import select
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

import numpy as np
from PIL import Image, ImageDraw

try:
    from cycle_ground_backgrounds import (
        DEFAULT_BACKGROUNDS_DIR,
        DEFAULT_CAMERA_PATHS,
        DEFAULT_WORLD,
        GROUND_MESH_PATH,
        GROUND_PLANE_PATH,
        MANNEQUIN_PATH,
        MANNEQUIN_Z_OFFSET,
        MATERIAL_PATH,
        _color_temperature_rgb_gains,
        build_camera_calibration,
        capture_synchronized_camera_views,
        label_mannequin_for_bbox,
        set_render_product_updates,
        settle_mannequin,
        validate_camera_prims,
        validate_yolo_runtime,
        wait_for_stage,
    )
    from target_fusion import (
        CameraObservation,
        aim_cameras_at_target,
        clear_debug_draw,
        compute_display_ray_length_to_ground,
        compute_world_target_center,
        draw_comparison_rays,
    )
    from yolo_inference import (
        FusionComparison,
        YoloInferenceResult,
        build_yolo_observations,
        compare_observation_fusions,
        fuse_observations,
        infer_yolo_frames,
        load_yolo_model,
        normalize_rgb_frame,
    )
except ModuleNotFoundError:
    from scripts.cycle_ground_backgrounds import (
        DEFAULT_BACKGROUNDS_DIR,
        DEFAULT_CAMERA_PATHS,
        DEFAULT_WORLD,
        GROUND_MESH_PATH,
        GROUND_PLANE_PATH,
        MANNEQUIN_PATH,
        MANNEQUIN_Z_OFFSET,
        MATERIAL_PATH,
        _color_temperature_rgb_gains,
        build_camera_calibration,
        capture_synchronized_camera_views,
        label_mannequin_for_bbox,
        set_render_product_updates,
        settle_mannequin,
        validate_camera_prims,
        validate_yolo_runtime,
        wait_for_stage,
    )
    from scripts.target_fusion import (
        CameraObservation,
        aim_cameras_at_target,
        clear_debug_draw,
        compute_display_ray_length_to_ground,
        compute_world_target_center,
        draw_comparison_rays,
    )
    from scripts.yolo_inference import (
        FusionComparison,
        YoloInferenceResult,
        build_yolo_observations,
        compare_observation_fusions,
        fuse_observations,
        infer_yolo_frames,
        load_yolo_model,
        normalize_rgb_frame,
    )


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = (
    PROJECT_DIR
    / "outputs"
    / "yolo_training_runs"
    / "mannequin_yolo11n_bbox"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs" / "yolo_prediction_demo"
DEMO_RESOLUTION = (640, 480)
DEMO_TARGET_LABEL = "mannequin"
DEMO_RT_SUBFRAMES = 1


@dataclass(frozen=True)
class DemoTrial:
    """One deterministic presentation condition."""

    name: str
    background_filename: str
    position_fraction_xy: tuple[float, float] = (0.0, 0.0)
    resolution_scale: float = 1.0
    brightness_offset: float = 0.0
    exposure_stops: float = 0.0
    color_temperature_k: float = 6500.0
    rgb_noise_std: float = 0.0
    seed: int = 0

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")


DEMO_TRIALS = (
    DemoTrial("Clean baseline", "01_aerial_grass_rock.png", seed=101),
    DemoTrial(
        "Positional challenge",
        "07_road_001.png",
        position_fraction_xy=(0.35, -0.20),
        seed=202,
    ),
    DemoTrial(
        "Reduced camera resolution",
        "03_gravel_ground_01.png",
        resolution_scale=0.75,
        seed=303,
    ),
    DemoTrial(
        "Difficult illumination",
        "04_forest_ground_01.png",
        brightness_offset=-0.06,
        exposure_stops=-0.40,
        color_temperature_k=4800.0,
        seed=404,
    ),
    DemoTrial(
        "Combined stress",
        "06_asphalt_023s.png",
        position_fraction_xy=(-0.40, 0.30),
        resolution_scale=0.75,
        brightness_offset=0.04,
        exposure_stops=0.30,
        color_temperature_k=8000.0,
        rgb_noise_std=8.0,
        seed=505,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show five interactive YOLO predictions before revealing Isaac ground truth."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="Ultralytics device, such as 0 or cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def validate_demo_preflight(model_path: Path, *, stream: TextIO | None = None) -> Path:
    """Fail before Isaac startup when the checkpoint or terminal is unavailable."""
    stream = sys.stdin if stream is None else stream
    if not callable(getattr(stream, "isatty", None)) or not stream.isatty():
        raise RuntimeError("interactive demo must be launched from a terminal")
    resolved_model = Path(model_path).expanduser()
    if not resolved_model.is_absolute():
        resolved_model = PROJECT_DIR / resolved_model
    resolved_model = resolved_model.resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint does not exist yet: {resolved_model}. "
            "Wait for training to produce best.pt before launching the GUI demo."
        )
    return resolved_model


def resolve_trial_position(
    trial: DemoTrial,
    *,
    ground_center,
    ground_half_extent,
    z_position: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one relative XY challenge to a world position and metric offset."""
    center = np.asarray(ground_center, dtype=np.float64)
    half_extent = np.asarray(ground_half_extent, dtype=np.float64)
    if center.shape != (3,) or half_extent.shape != (3,):
        raise ValueError("ground center and half extent must contain three values")
    fraction = np.asarray(trial.position_fraction_xy, dtype=np.float64)
    offset = np.array(
        [fraction[0] * half_extent[0], fraction[1] * half_extent[1], 0.0],
        dtype=np.float64,
    )
    position = center + offset
    position[2] = float(z_position)
    return position, offset


def apply_trial_camera_condition(
    rgb_frame,
    trial: DemoTrial,
    *,
    camera_index: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply one trial's exact camera degradation to a normalized RGB frame."""
    frame = normalize_rgb_frame(rgb_frame)
    height, width = frame.shape[:2]
    intermediate = (
        max(1, int(round(width * trial.resolution_scale))),
        max(1, int(round(height * trial.resolution_scale))),
    )
    if intermediate != (width, height):
        image = Image.fromarray(frame, mode="RGB")
        image = image.resize(intermediate, resample=Image.Resampling.LANCZOS)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        frame = np.asarray(image).copy()

    gains = _color_temperature_rgb_gains(trial.color_temperature_k)
    adjusted = frame.astype(np.float64)
    adjusted *= (2.0**trial.exposure_stops) * gains.reshape((1, 1, 3))
    adjusted += trial.brightness_offset * 255.0
    if trial.rgb_noise_std > 0.0:
        rng = np.random.default_rng(trial.seed + int(camera_index))
        adjusted += rng.normal(0.0, trial.rgb_noise_std, size=adjusted.shape)
    return np.clip(np.rint(adjusted), 0.0, 255.0).astype(np.uint8), intermediate


def save_demo_camera_images(
    frame,
    inference: YoloInferenceResult,
    *,
    trial_directory: Path,
    camera_index: int,
) -> tuple[Path, Path]:
    """Save the exact model input and a separate prediction overlay."""
    rgb = normalize_rgb_frame(frame)
    trial_directory.mkdir(parents=True, exist_ok=True)
    stem = f"camera_{camera_index + 1:02d}"
    input_path = trial_directory / f"{stem}_input.png"
    prediction_path = trial_directory / f"{stem}_prediction.png"
    Image.fromarray(rgb, mode="RGB").save(input_path, format="PNG")

    prediction_image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(prediction_image)
    if inference.detection is None:
        draw.rectangle((5, 5, 245, 29), fill=(0, 0, 0))
        draw.text((9, 9), "YOLO: NO DETECTION", fill=(255, 70, 70))
    else:
        bbox = inference.detection.bbox
        bounds = tuple(
            int(round(value))
            for value in (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        )
        draw.rectangle(bounds, outline=(40, 140, 255), width=3)
        label = f"YOLO {inference.detection.confidence:.3f}"
        text_x = max(0, bounds[0])
        text_y = max(0, bounds[1] - 18)
        draw.rectangle((text_x, text_y, text_x + 112, text_y + 17), fill=(0, 0, 0))
        draw.text((text_x + 3, text_y + 2), label, fill=(40, 140, 255))
    prediction_image.save(prediction_path, format="PNG")
    return input_path, prediction_path


def run_trial_inference(
    loaded_model,
    frames,
    *,
    confidence: float,
    iou: float,
    device=None,
    inference_runner: Callable | None = None,
) -> list[YoloInferenceResult]:
    """Run one trial through an injectable inference runner for checkpoint-free tests."""
    runner = infer_yolo_frames if inference_runner is None else inference_runner
    results = runner(
        loaded_model,
        frames,
        confidence_threshold=confidence,
        iou_threshold=iou,
        image_size=640,
        device=device,
    )
    results = list(results)
    if len(results) != len(frames):
        raise RuntimeError("YOLO inference must return one result per camera frame")
    return results


def format_trial_changes(
    trial: DemoTrial,
    *,
    position_offset,
    intermediate_resolution: tuple[int, int],
    final_resolution: tuple[int, int] = DEMO_RESOLUTION,
) -> list[str]:
    """Return precise, presenter-friendly descriptions of applied changes."""
    offset = np.asarray(position_offset, dtype=np.float64)
    lines = [f"Background: {trial.background_filename}"]
    if np.linalg.norm(offset) > 0.0:
        lines.append(
            "Positional offset: "
            f"X={offset[0]:+.3f} m, Y={offset[1]:+.3f} m, Z={offset[2]:+.3f} m"
        )
    else:
        lines.append("Positional offset: none (ground center)")

    camera_changes = []
    if trial.resolution_scale != 1.0:
        camera_changes.append(
            f"resolution {trial.resolution_scale:.2f}x "
            f"({intermediate_resolution[0]}x{intermediate_resolution[1]} -> "
            f"{final_resolution[0]}x{final_resolution[1]})"
        )
    if trial.brightness_offset != 0.0:
        camera_changes.append(f"brightness {trial.brightness_offset:+.3f}")
    if trial.exposure_stops != 0.0:
        camera_changes.append(f"exposure {trial.exposure_stops:+.2f} stops")
    if trial.color_temperature_k != 6500.0:
        camera_changes.append(f"color temperature {trial.color_temperature_k:.0f} K")
    if trial.rgb_noise_std != 0.0:
        camera_changes.append(f"RGB Gaussian pixel noise sigma={trial.rgb_noise_std:.1f}")
    lines.append(
        "Camera condition: clean" if not camera_changes else "Camera condition: " + ", ".join(camera_changes)
    )
    return lines


def wait_for_terminal(
    simulation_app,
    prompt: str,
    *,
    stream: TextIO | None = None,
    select_fn: Callable = select.select,
) -> bool:
    """Wait for Enter or q while continuing to service the Isaac application."""
    stream = sys.stdin if stream is None else stream
    if not callable(getattr(stream, "isatty", None)) or not stream.isatty():
        raise RuntimeError("interactive demo requires a terminal-connected stdin")
    print(prompt, end="", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
        readable, _, _ = select_fn([stream], [], [], 0.0)
        if not readable:
            continue
        response = stream.readline()
        if response == "":
            raise EOFError("terminal input closed during the interactive demo")
        return response.strip().lower() != "q"
    return False


def run_display_sequence(
    *,
    show_prediction: Callable[[], None],
    show_ground_truth: Callable[[], None],
    clear_display: Callable[[], None],
    wait: Callable[[str], bool],
    report_ground_truth: Callable[[], None],
) -> bool:
    """Run the fixed prediction/prompt/ground-truth/prompt presentation order."""
    show_prediction()
    if not wait("Press Enter to reveal ground truth, or q to quit: "):
        clear_display()
        return False
    clear_display()
    show_ground_truth()
    report_ground_truth()
    if not wait("Press Enter for the next trial, or q to quit: "):
        clear_display()
        return False
    clear_display()
    return True


def prediction_view_kwargs(yolo_fusion, *, ray_length: float) -> dict:
    """Build prediction-only draw arguments without leaking truth geometry."""
    return {
        "ground_truth_rays": [],
        "yolo_rays": yolo_fusion.rays,
        "yolo_fused_position_world": (
            yolo_fusion.fusion.fused_position_world
            if yolo_fusion.fusion.valid
            else None
        ),
        "truth_world": None,
        "ray_length": ray_length,
    }


def ground_truth_view_kwargs(ground_truth_fusion, *, target_world, ray_length: float) -> dict:
    """Build the ground-truth-only draw arguments used after the reveal."""
    return {
        "ground_truth_rays": ground_truth_fusion.rays,
        "yolo_rays": [],
        "ground_truth_fused_position_world": (
            ground_truth_fusion.fusion.fused_position_world
            if ground_truth_fusion.fusion.valid
            else None
        ),
        "truth_world": target_world,
        "ray_length": ray_length,
    }


def print_detection_summary(
    inference_results: list[YoloInferenceResult],
    *,
    camera_paths: list[str],
) -> None:
    for camera_path, result in zip(camera_paths, inference_results):
        camera_name = Path(camera_path).name
        if result.detection is None:
            print(f"  {camera_name}: MISS - {result.reason or 'no target detection'}")
        else:
            print(
                f"  {camera_name}: confidence={result.detection.confidence:.3f}, "
                f"inference={result.inference_time_ms:.1f} ms"
            )


def print_comparison_summary(comparison: FusionComparison) -> None:
    print("Ground-truth comparison:")
    for camera in comparison.cameras:
        print(
            f"  {Path(camera.camera_path).name}: "
            f"IoU={camera.bbox_iou if camera.bbox_iou is not None else 'n/a'}, "
            f"center_error_px={camera.center_error_px if camera.center_error_px is not None else 'n/a'}, "
            f"ray_angle_deg={camera.ray_angle_deg if camera.ray_angle_deg is not None else 'n/a'}"
        )
    print(
        "  fused_position_difference_m="
        f"{comparison.fused_position_delta_m if comparison.fused_position_delta_m is not None else 'n/a'}"
    )


def _timestamped_output_directory(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root.expanduser().resolve() / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root.expanduser().resolve() / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> None:
    args = parse_args()
    model_path = validate_demo_preflight(args.model)
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--confidence and --iou must be in [0, 1]")
    validate_yolo_runtime("after-ground-truth")

    world_path = DEFAULT_WORLD.resolve()
    backgrounds_dir = DEFAULT_BACKGROUNDS_DIR.resolve()
    background_paths = {
        path.name: path.resolve() for path in backgrounds_dir.glob("*.png") if path.is_file()
    }
    missing_backgrounds = [
        trial.background_filename
        for trial in DEMO_TRIALS
        if trial.background_filename not in background_paths
    ]
    if missing_backgrounds:
        raise FileNotFoundError("missing demo backgrounds: " + ", ".join(missing_backgrounds))
    if not world_path.is_file():
        raise FileNotFoundError(f"USD stage does not exist: {world_path}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    render_products = []
    bbox_annotators = []
    rgb_annotators = []
    try:
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.experimental.materials import OmniPbrMaterial
        from isaacsim.core.experimental.objects import GroundPlane
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics

        loaded_model = load_yolo_model(
            model_path,
            target_label=DEMO_TARGET_LABEL,
            project_dir=PROJECT_DIR,
            relative_to=PROJECT_DIR,
        )
        print(f"Loaded YOLO model: {loaded_model.info.path}")

        stage_utils.open_stage(str(world_path))
        wait_for_stage(simulation_app, stage_utils)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage did not open")
        for required_path in (GROUND_PLANE_PATH, GROUND_MESH_PATH, MANNEQUIN_PATH):
            if not stage.GetPrimAtPath(required_path).IsValid():
                raise RuntimeError(f"loaded stage is missing required prim: {required_path}")
        mannequin_prim = stage.GetPrimAtPath(MANNEQUIN_PATH)
        if not mannequin_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"required rigid-body API is missing from {MANNEQUIN_PATH}")
        validate_camera_prims(stage, DEFAULT_CAMERA_PATHS, UsdGeom.Camera)
        label_mannequin_for_bbox(
            stage,
            mannequin_prim,
            UsdGeom.Imageable,
            target_label=DEMO_TARGET_LABEL,
        )

        mannequin = RigidPrim(MANNEQUIN_PATH, reset_xform_op_properties=False)
        authored_positions, authored_orientations = mannequin.get_world_poses()
        if SimulationManager.get_physics_simulation_view() is None:
            SimulationManager.initialize_physics()
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        authored_position = authored_positions.numpy()[0].copy()
        authored_orientation = authored_orientations.numpy()[0].copy()

        ground_range = UsdGeom.Imageable(stage.GetPrimAtPath(GROUND_MESH_PATH)).ComputeWorldBound(
            Usd.TimeCode.Default(), "default"
        ).ComputeAlignedRange()
        ground_min = np.asarray(ground_range.GetMin(), dtype=np.float64)
        ground_max = np.asarray(ground_range.GetMax(), dtype=np.float64)
        ground_center = (ground_min + ground_max) * 0.5
        ground_half_extent = (ground_max - ground_min) * 0.25

        authored_target = compute_world_target_center(stage, MANNEQUIN_PATH)
        fixed_camera_target = np.array(
            [ground_center[0], ground_center[1], authored_target[2] + MANNEQUIN_Z_OFFSET],
            dtype=np.float64,
        )
        aim_cameras_at_target(stage, DEFAULT_CAMERA_PATHS, fixed_camera_target)
        simulation_app.update()
        calibrations = [
            build_camera_calibration(stage, path, DEMO_RESOLUTION)
            for path in DEFAULT_CAMERA_PATHS
        ]

        ground_plane = GroundPlane(GROUND_PLANE_PATH, templates=None)
        material = OmniPbrMaterial(MATERIAL_PATH)
        material.set_input_values("diffuse_color_constant", [1.0, 1.0, 1.0])
        material.set_input_values("project_uvw", False)
        material.set_input_values("texture_scale", [1.0, 1.0])
        material.set_input_values("texture_translate", [0.0, 0.0])
        ground_plane.meshes.apply_visual_materials(material)

        rep.orchestrator.set_capture_on_play(False)
        for camera_index, camera_path in enumerate(DEFAULT_CAMERA_PATHS):
            render_product = rep.create.render_product(
                camera_path,
                DEMO_RESOLUTION,
                name=f"PredictionDemo_Camera_{camera_index + 1:02d}",
            )
            bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            bbox_annotator.attach(render_product)
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach(render_product)
            render_products.append(render_product)
            bbox_annotators.append(bbox_annotator)
            rgb_annotators.append(rgb_annotator)
        simulation_app.update()

        run_directory = _timestamped_output_directory(args.output_root)
        print(f"Demo output: {run_directory}")
        for trial_index, trial in enumerate(DEMO_TRIALS, start=1):
            if not simulation_app.is_running():
                break
            clear_debug_draw()
            material.set_input_values(
                "diffuse_texture", str(background_paths[trial.background_filename])
            )
            requested_position, position_offset = resolve_trial_position(
                trial,
                ground_center=ground_center,
                ground_half_extent=ground_half_extent,
                z_position=authored_position[2] + MANNEQUIN_Z_OFFSET,
            )
            mannequin.set_world_poses(
                positions=[requested_position],
                orientations=[authored_orientation],
            )
            mannequin.set_velocities(
                linear_velocities=np.zeros((1, 3), dtype=np.float32),
                angular_velocities=np.zeros((1, 3), dtype=np.float32),
            )
            simulation_app.update()
            timeline.play()
            settled = settle_mannequin(
                simulation_app,
                mannequin,
                SimulationManager,
                10.0,
            )
            timeline.pause()
            if not settled:
                print(f"[warning] mannequin did not settle in trial {trial_index}")

            target_world = compute_world_target_center(stage, MANNEQUIN_PATH)
            target_bboxes, captured_frames = capture_synchronized_camera_views(
                rep.orchestrator,
                bbox_annotators,
                rgb_annotators,
                render_products=render_products,
                rt_subframes=DEMO_RT_SUBFRAMES,
                render_resolution=DEMO_RESOLUTION,
                target_label=DEMO_TARGET_LABEL,
                max_occlusion_ratio=0.5,
                border_tolerance_px=0.0,
            )
            conditioned_frames = []
            intermediate_resolutions = []
            for camera_index, frame in enumerate(captured_frames):
                conditioned, intermediate = apply_trial_camera_condition(
                    frame,
                    trial,
                    camera_index=camera_index,
                )
                conditioned_frames.append(conditioned)
                intermediate_resolutions.append(intermediate)

            inference_results = run_trial_inference(
                loaded_model,
                conditioned_frames,
                confidence=args.confidence,
                iou=args.iou,
                device=args.device,
            )
            trial_directory = run_directory / f"trial_{trial_index:02d}_{trial.slug}"
            for camera_index, (frame, inference) in enumerate(
                zip(conditioned_frames, inference_results)
            ):
                save_demo_camera_images(
                    frame,
                    inference,
                    trial_directory=trial_directory,
                    camera_index=camera_index,
                )

            ground_truth_observations = [
                CameraObservation(
                    camera_path=camera_path,
                    calibration=calibration,
                    bbox=bbox,
                    capture_id=trial_index,
                )
                for camera_path, calibration, bbox in zip(
                    DEFAULT_CAMERA_PATHS,
                    calibrations,
                    target_bboxes,
                )
            ]
            ground_truth_fusion = fuse_observations(
                ground_truth_observations,
                target_world=target_world,
            )
            yolo_observations = build_yolo_observations(
                calibrations,
                inference_results,
                capture_id=trial_index,
            )
            yolo_fusion = fuse_observations(yolo_observations, target_world=target_world)
            comparison = compare_observation_fusions(
                ground_truth_fusion,
                yolo_fusion,
                inference_results=inference_results,
            )
            display_length = compute_display_ray_length_to_ground(
                [*ground_truth_fusion.rays, *yolo_fusion.rays],
                ground_plane_z=0.0,
                extra_length_m=1.0,
            )

            print(f"\n=== Trial {trial_index}/5: {trial.name} ===")
            print(f"Saved images: {trial_directory}")
            for line in format_trial_changes(
                trial,
                position_offset=position_offset,
                intermediate_resolution=intermediate_resolutions[0],
            ):
                print(line)
            print("YOLO prediction:")
            print_detection_summary(inference_results, camera_paths=DEFAULT_CAMERA_PATHS)

            def show_prediction() -> None:
                clear_debug_draw()
                if yolo_fusion.rays or yolo_fusion.fusion.fused_position_world is not None:
                    draw_comparison_rays(
                        **prediction_view_kwargs(
                            yolo_fusion,
                            ray_length=display_length,
                        )
                    )

            def show_ground_truth() -> None:
                draw_comparison_rays(
                    **ground_truth_view_kwargs(
                        ground_truth_fusion,
                        target_world=target_world,
                        ray_length=display_length,
                    )
                )

            should_continue = run_display_sequence(
                show_prediction=show_prediction,
                show_ground_truth=show_ground_truth,
                clear_display=clear_debug_draw,
                wait=lambda prompt: wait_for_terminal(simulation_app, prompt),
                report_ground_truth=lambda: print_comparison_summary(comparison),
            )
            if not should_continue:
                print("Demo ended by user.")
                break
        else:
            print(f"\nDemo complete. Saved 40 PNGs under {run_directory}")
    finally:
        try:
            clear_debug_draw()
        except Exception:
            pass
        for annotator in [*bbox_annotators, *rgb_annotators]:
            try:
                annotator.detach()
            except Exception:
                pass
        try:
            set_render_product_updates(render_products, False)
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
