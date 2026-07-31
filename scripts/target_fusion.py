"""Geometry and Isaac Sim helpers for multi-camera target fusion.

The numerical functions remain importable without Isaac Sim.  USD-dependent
helpers keep their Isaac imports inside the functions so the same module can
be exercised by regular Python unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np


_VECTOR_EPSILON = 1e-12
_DEFAULT_RANK_TOLERANCE = 1e-10
_DEFAULT_CONDITION_LIMIT = 1e12
DEFAULT_CAMERA_COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 0.8, 1.0, 1.0),
    (1.0, 0.8, 0.1, 1.0),
    (0.7, 0.3, 1.0, 1.0),
)
DEFAULT_TARGET_COLOR = (0.1, 1.0, 0.2, 1.0)


def _coerce_vector3(value, *, name: str) -> np.ndarray:
    """Convert *value* to a finite 3-vector."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def normalize(vector, *, name: str = "vector") -> np.ndarray:
    """Return a finite unit 3-vector or raise a descriptive ``ValueError``."""
    result = _coerce_vector3(vector, name=name)
    magnitude = float(np.linalg.norm(result))
    if not isfinite(magnitude) or magnitude <= _VECTOR_EPSILON:
        raise ValueError(f"{name} must have nonzero length")
    return result / magnitude


@dataclass
class CameraRay:
    """A world-space ray emitted from one camera."""

    camera_path: str
    origin_world: np.ndarray
    direction_world: np.ndarray
    target_distance_m: float

    def __post_init__(self) -> None:
        self.origin_world = _coerce_vector3(self.origin_world, name="origin_world")
        self.direction_world = normalize(self.direction_world, name="direction_world")
        self.target_distance_m = float(self.target_distance_m)
        if not isfinite(self.target_distance_m) or self.target_distance_m <= _VECTOR_EPSILON:
            raise ValueError("target_distance_m must be finite and positive")

    def point_at(self, distance_m: float) -> np.ndarray:
        """Return a point on the ray at the requested nonnegative distance."""
        distance = float(distance_m)
        if not isfinite(distance) or distance < 0.0:
            raise ValueError("distance_m must be finite and nonnegative")
        return self.origin_world + distance * self.direction_world

    def as_dict(self) -> dict:
        """Return JSON-compatible ground-truth fields for this ray."""
        return {
            "camera_path": self.camera_path,
            "origin_world": self.origin_world.tolist(),
            "direction_world": self.direction_world.tolist(),
            "target_distance_m": self.target_distance_m,
        }


@dataclass
class FusionResult:
    """Result and diagnostics from least-squares ray intersection."""

    fused_position_world: np.ndarray | None
    error_m: float | None
    rms_residual_m: float | None
    rank: int
    condition_number: float
    valid: bool
    reason: str | None = None

    def as_dict(self) -> dict:
        """Return JSON-compatible fusion fields."""
        condition_number = (
            float(self.condition_number) if isfinite(self.condition_number) else None
        )
        return {
            "fused_position_world": (
                None if self.fused_position_world is None else self.fused_position_world.tolist()
            ),
            "error_m": None if self.error_m is None else float(self.error_m),
            "rms_residual_m": (
                None if self.rms_residual_m is None else float(self.rms_residual_m)
            ),
            "rank": int(self.rank),
            "condition_number": condition_number,
            "valid": bool(self.valid),
            "reason": self.reason,
        }


def build_ground_truth_ray(
    camera_path: str,
    camera_origin_world,
    target_world,
) -> CameraRay:
    """Construct the exact bearing ray from a camera origin to a target."""
    origin = _coerce_vector3(camera_origin_world, name="camera_origin_world")
    target = _coerce_vector3(target_world, name="target_world")
    offset = target - origin
    distance = float(np.linalg.norm(offset))
    if not isfinite(distance) or distance <= _VECTOR_EPSILON:
        raise ValueError(f"camera {camera_path!r} cannot share the target position")
    return CameraRay(
        camera_path=str(camera_path),
        origin_world=origin,
        direction_world=offset / distance,
        target_distance_m=distance,
    )


def compute_world_target_center(
    stage,
    prim_path: str,
    *,
    time_code=None,
) -> np.ndarray:
    """Return the world-space aligned-bounds center of a USD prim."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Target prim does not exist: {prim_path}")
    if not prim.IsA(UsdGeom.Imageable):
        raise RuntimeError(f"Target prim is not imageable: {prim_path}")

    if time_code is None:
        time_code = Usd.TimeCode.Default()
    world_range = UsdGeom.Imageable(prim).ComputeWorldBound(time_code, "default").ComputeAlignedRange()
    minimum = _coerce_vector3(list(world_range.GetMin()), name=f"{prim_path} world bound minimum")
    maximum = _coerce_vector3(list(world_range.GetMax()), name=f"{prim_path} world bound maximum")
    if np.any(maximum < minimum):
        raise RuntimeError(f"Target prim has an invalid world bound: {prim_path}")
    return (minimum + maximum) * 0.5


def _get_camera_prim(stage, camera_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        raise RuntimeError(f"Camera prim does not exist: {camera_path}")
    if not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera path is not a UsdGeom.Camera: {camera_path}")
    return prim


def get_camera_world_pose(
    stage,
    camera_path: str,
    *,
    time_code=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a camera's world position and negative-Z forward direction."""
    from pxr import Gf, Usd, UsdGeom

    prim = _get_camera_prim(stage, camera_path)
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    world_matrix = Gf.Matrix4d(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
    )
    translation = world_matrix.ExtractTranslation()
    forward = world_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    position_world = _coerce_vector3(list(translation), name=f"{camera_path} world position")
    forward_world = normalize(list(forward), name=f"{camera_path} forward direction")
    return position_world, forward_world


def aim_camera_at_target(
    stage,
    camera_path: str,
    target_world,
    *,
    up_world=(0.0, 0.0, 1.0),
    alignment_tolerance: float = 0.999999,
) -> dict:
    """Aim one camera at a world target while preserving its world position."""
    from isaacsim.core.experimental.prims import XformPrim
    from pxr import Gf

    if not isfinite(alignment_tolerance) or not 0.0 < alignment_tolerance <= 1.0:
        raise ValueError("alignment_tolerance must be in the interval (0, 1]")

    position_world, _ = get_camera_world_pose(stage, camera_path)
    target = _coerce_vector3(target_world, name="target_world")
    desired_forward = normalize(target - position_world, name=f"{camera_path} target bearing")
    up = normalize(up_world, name="up_world")

    eye = Gf.Vec3d(*position_world.tolist())
    target_point = Gf.Vec3d(*target.tolist())
    up_point = Gf.Vec3d(*up.tolist())
    world_matrix = Gf.Matrix4d().SetLookAt(eye, target_point, up_point).GetInverse()
    quaternion = world_matrix.ExtractRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    orientation_world = np.array(
        [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float64,
    )

    # The wrapper converts the requested world pose into the camera's local
    # parent frame. Resetting xform ops preserves the current world pose and
    # leaves camera attributes such as focal length and clipping unchanged.
    camera = XformPrim(camera_path, reset_xform_op_properties=True)
    camera.set_world_poses(
        positions=[position_world.tolist()],
        orientations=[orientation_world.tolist()],
    )

    actual_position, actual_forward = get_camera_world_pose(stage, camera_path)
    alignment = float(np.dot(actual_forward, desired_forward))
    position_error = float(np.linalg.norm(actual_position - position_world))
    if alignment < alignment_tolerance:
        raise RuntimeError(
            f"Camera {camera_path} failed target alignment: dot={alignment:.9f}, "
            f"required>={alignment_tolerance:.9f}"
        )

    return {
        "camera_path": camera_path,
        "position_world": actual_position,
        "forward_world": actual_forward,
        "desired_forward_world": desired_forward,
        "alignment": alignment,
        "position_error_m": position_error,
    }


def aim_cameras_at_target(
    stage,
    camera_paths: Sequence[str],
    target_world,
    *,
    up_world=(0.0, 0.0, 1.0),
    alignment_tolerance: float = 0.999999,
) -> list[dict]:
    """Aim every configured camera at one target and return pose diagnostics."""
    if not camera_paths:
        raise ValueError("camera_paths must contain at least one camera path")
    return [
        aim_camera_at_target(
            stage,
            camera_path,
            target_world,
            up_world=up_world,
            alignment_tolerance=alignment_tolerance,
        )
        for camera_path in camera_paths
    ]


def draw_ground_truth_rays(
    rays: Sequence[CameraRay],
    target_world,
    *,
    clear_existing: bool = True,
    line_thickness: float = 3.0,
    target_size: float = 12.0,
    camera_colors: Sequence[tuple[float, float, float, float]] = DEFAULT_CAMERA_COLORS,
    target_color: tuple[float, float, float, float] = DEFAULT_TARGET_COLOR,
) -> dict:
    """Draw exact camera-to-target rays and a target marker in the viewport.

    The debug-draw primitives are transient and are not authored into the USD
    stage.  Callers running headless can skip this function entirely.
    """
    if not rays:
        raise ValueError("rays must contain at least one CameraRay")
    if not isfinite(line_thickness) or line_thickness <= 0.0:
        raise ValueError("line_thickness must be finite and positive")
    if not isfinite(target_size) or target_size <= 0.0:
        raise ValueError("target_size must be finite and positive")
    if not camera_colors:
        raise ValueError("camera_colors must contain at least one RGBA color")

    target = _coerce_vector3(target_world, name="target_world")
    from isaacsim.util.debug_draw import _debug_draw

    draw_interface = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw_interface.clear_lines()
        draw_interface.clear_points()

    start_points = [ray.origin_world.tolist() for ray in rays]
    end_points = [target.tolist() for _ in rays]
    colors = [list(camera_colors[index % len(camera_colors)]) for index in range(len(rays))]
    thicknesses = [line_thickness] * len(rays)
    draw_interface.draw_lines(start_points, end_points, colors, thicknesses)
    draw_interface.draw_points([target.tolist()], [list(target_color)], [target_size])
    return {
        "ray_count": len(rays),
        "target_world": target,
        "camera_colors": colors,
    }


def _invalid_result(reason: str, *, rank: int = 0, condition_number: float = float("inf")) -> FusionResult:
    return FusionResult(
        fused_position_world=None,
        error_m=None,
        rms_residual_m=None,
        rank=rank,
        condition_number=condition_number,
        valid=False,
        reason=reason,
    )


def fuse_rays(
    rays: Sequence[CameraRay],
    *,
    target_world=None,
    rank_tolerance: float = _DEFAULT_RANK_TOLERANCE,
    condition_limit: float = _DEFAULT_CONDITION_LIMIT,
) -> FusionResult:
    """Estimate the common point of camera rays by least-squares intersection.

    For each ray ``(o_i, d_i)``, the perpendicular projection matrix is
    ``P_i = I - d_i d_i^T``.  The returned estimate solves
    ``sum(P_i) x = sum(P_i o_i)``.  ``target_world`` is optional and is used
    only to report the ground-truth position error.
    """
    if len(rays) < 2:
        return _invalid_result("at least two rays are required")
    if not isfinite(rank_tolerance) or rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be finite and positive")
    if not isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive")

    system_matrix = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for ray in rays:
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
        direction = normalize(ray.direction_world, name=f"{ray.camera_path} direction_world")
        origin = _coerce_vector3(ray.origin_world, name=f"{ray.camera_path} origin_world")
        projection = identity - np.outer(direction, direction)
        system_matrix += projection
        right_hand_side += projection @ origin

    if not np.all(np.isfinite(system_matrix)) or not np.all(np.isfinite(right_hand_side)):
        return _invalid_result("ray system contains nonfinite values")

    singular_values = np.linalg.svd(system_matrix, compute_uv=False)
    largest_singular_value = float(singular_values[0])
    rank_cutoff = rank_tolerance * max(largest_singular_value, _VECTOR_EPSILON)
    rank = int(np.count_nonzero(singular_values > rank_cutoff))
    smallest_singular_value = float(singular_values[-1])
    condition_number = (
        largest_singular_value / smallest_singular_value
        if smallest_singular_value > _VECTOR_EPSILON
        else float("inf")
    )

    if rank < 3:
        return _invalid_result(
            "ray geometry is rank deficient",
            rank=rank,
            condition_number=condition_number,
        )
    if condition_number > condition_limit:
        return _invalid_result(
            "ray geometry is ill-conditioned",
            rank=rank,
            condition_number=condition_number,
        )

    fused_position, _, _, _ = np.linalg.lstsq(
        system_matrix,
        right_hand_side,
        rcond=rank_tolerance,
    )
    if not np.all(np.isfinite(fused_position)):
        return _invalid_result(
            "ray solution contains nonfinite values",
            rank=rank,
            condition_number=condition_number,
        )

    squared_distances = []
    for ray in rays:
        offset = fused_position - ray.origin_world
        perpendicular_offset = offset - np.dot(offset, ray.direction_world) * ray.direction_world
        squared_distances.append(float(np.dot(perpendicular_offset, perpendicular_offset)))
    rms_residual = float(np.sqrt(np.mean(squared_distances)))

    error_m = None
    if target_world is not None:
        target = _coerce_vector3(target_world, name="target_world")
        error_m = float(np.linalg.norm(fused_position - target))

    return FusionResult(
        fused_position_world=fused_position,
        error_m=error_m,
        rms_residual_m=rms_residual,
        rank=rank,
        condition_number=condition_number,
        valid=True,
    )


def build_ground_truth_record(
    *,
    scene_index: int,
    background_path: str,
    target_prim_path: str,
    target_world,
    camera_aims: Sequence[dict],
    rays: Sequence[CameraRay],
    fusion_result: FusionResult,
    mannequin_position_world=None,
    settled: bool,
) -> dict:
    """Build one JSON-compatible ground-truth record for a generated scene."""
    if len(camera_aims) != len(rays):
        raise ValueError("camera_aims and rays must contain the same number of entries")
    if not rays:
        raise ValueError("at least one camera ray is required")
    if not isinstance(fusion_result, FusionResult):
        raise TypeError("fusion_result must be a FusionResult")

    target = _coerce_vector3(target_world, name="target_world")
    mannequin_position = None
    if mannequin_position_world is not None:
        mannequin_position = _coerce_vector3(
            mannequin_position_world,
            name="mannequin_position_world",
        ).tolist()

    cameras = []
    for camera_aim, ray in zip(camera_aims, rays):
        if not isinstance(camera_aim, dict):
            raise TypeError("camera_aims must contain dictionaries")
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
        camera_path = str(camera_aim.get("camera_path", ""))
        if not camera_path:
            raise ValueError("each camera aim must contain a camera_path")
        if camera_path != ray.camera_path:
            raise ValueError(
                f"camera aim path {camera_path!r} does not match ray path {ray.camera_path!r}"
            )

        cameras.append(
            {
                "camera_path": camera_path,
                "position_world": _coerce_vector3(
                    camera_aim["position_world"],
                    name=f"{camera_path} position_world",
                ).tolist(),
                "forward_world": _coerce_vector3(
                    camera_aim["forward_world"],
                    name=f"{camera_path} forward_world",
                ).tolist(),
                "desired_forward_world": _coerce_vector3(
                    camera_aim["desired_forward_world"],
                    name=f"{camera_path} desired_forward_world",
                ).tolist(),
                "alignment": float(camera_aim["alignment"]),
                "position_error_m": float(camera_aim["position_error_m"]),
                "ray": ray.as_dict(),
            }
        )

    return {
        "schema_version": 1,
        "scene_index": int(scene_index),
        "background_path": str(background_path),
        "target_prim_path": str(target_prim_path),
        "mannequin_position_world": mannequin_position,
        "target_center_world": target.tolist(),
        "settled": bool(settled),
        "camera_count": len(cameras),
        "cameras": cameras,
        "fusion": fusion_result.as_dict(),
    }
