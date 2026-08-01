"""Geometry and Isaac Sim helpers for multi-camera target fusion.

The numerical functions remain importable without Isaac Sim.  USD-dependent
helpers keep their Isaac imports inside the functions so the same module can
be exercised by regular Python unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
DEFAULT_GROUND_TRUTH_RAY_COLOR = (0.0, 1.0, 0.0, 1.0)
DEFAULT_YOLO_RAY_COLOR = (0.1, 0.4, 1.0, 1.0)
DEFAULT_GROUND_TRUTH_FUSED_COLOR = (0.0, 0.8, 0.0, 1.0)
DEFAULT_YOLO_FUSED_COLOR = (0.0, 0.2, 1.0, 1.0)
DEFAULT_TARGET_COLOR = (0.1, 1.0, 0.2, 1.0)
DEFAULT_TRUTH_EVALUATION_COLOR = (1.0, 0.1, 0.1, 1.0)


def _coerce_vector3(value, *, name: str) -> np.ndarray:
    """Convert *value* to a finite 3-vector."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _coerce_vector2(value, *, name: str) -> np.ndarray:
    """Convert *value* to a finite 2-vector."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError(f"{name} must be a 2-vector, got shape {vector.shape}")
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


def _coerce_matrix3(value, *, name: str) -> np.ndarray:
    """Convert *value* to a finite 3x3 matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _coerce_resolution(value, *, name: str = "resolution") -> tuple[int, int]:
    """Convert a ``(width, height)`` resolution to validated integers."""
    values = np.asarray(value)
    if values.shape != (2,):
        raise ValueError(f"{name} must contain width and height")
    try:
        width, height = (int(values[0]), int(values[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integer dimensions") from exc
    if width <= 0 or height <= 0 or width != values[0] or height != values[1]:
        raise ValueError(f"{name} dimensions must be positive integers")
    return width, height


@dataclass
class CameraCalibration:
    """Pinhole calibration and world pose for one rendered camera.

    The resolution follows the Replicator render-product convention of
    ``(width, height)``.  USD camera aperture and focal-length values are kept
    in their authored units; only their ratios are used to calculate pixels.
    ``rotation_world_from_camera`` maps USD camera-local vectors to world
    vectors, where the optical axis is local ``-Z``.
    """

    camera_path: str
    resolution: tuple[int, int]
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    horizontal_aperture_offset: float = 0.0
    vertical_aperture_offset: float = 0.0
    projection: str = "perspective"
    origin_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation_world_from_camera: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.camera_path = str(self.camera_path)
        if not self.camera_path:
            raise ValueError("camera_path must not be empty")
        self.resolution = _coerce_resolution(self.resolution)
        for attribute_name in (
            "focal_length",
            "horizontal_aperture",
            "vertical_aperture",
            "horizontal_aperture_offset",
            "vertical_aperture_offset",
        ):
            value = float(getattr(self, attribute_name))
            if not isfinite(value):
                raise ValueError(f"{attribute_name} must be finite")
            setattr(self, attribute_name, value)
        if self.focal_length <= 0.0:
            raise ValueError("focal_length must be positive")
        if self.horizontal_aperture <= 0.0 or self.vertical_aperture <= 0.0:
            raise ValueError("camera apertures must be positive")
        self.projection = str(self.projection).lower()
        if self.projection != "perspective":
            raise ValueError(f"unsupported camera projection: {self.projection!r}")
        self.origin_world = _coerce_vector3(self.origin_world, name="origin_world")
        self.rotation_world_from_camera = _coerce_matrix3(
            self.rotation_world_from_camera,
            name="rotation_world_from_camera",
        )

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Return the pixel-space pinhole intrinsic matrix ``K``."""
        width, height = self.resolution
        fx = width * self.focal_length / self.horizontal_aperture
        fy = height * self.focal_length / self.vertical_aperture
        cx = width * (0.5 + self.horizontal_aperture_offset / self.horizontal_aperture)
        cy = height * (0.5 + self.vertical_aperture_offset / self.vertical_aperture)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def forward_world(self) -> np.ndarray:
        """Return the normalized world-space direction of local ``-Z``."""
        return normalize(
            self.rotation_world_from_camera @ np.array([0.0, 0.0, -1.0]),
            name="camera forward direction",
        )

    def as_dict(self) -> dict:
        """Return JSON-compatible calibration fields."""
        return {
            "camera_path": self.camera_path,
            "resolution": list(self.resolution),
            "focal_length": self.focal_length,
            "horizontal_aperture": self.horizontal_aperture,
            "vertical_aperture": self.vertical_aperture,
            "horizontal_aperture_offset": self.horizontal_aperture_offset,
            "vertical_aperture_offset": self.vertical_aperture_offset,
            "projection": self.projection,
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "origin_world": self.origin_world.tolist(),
            "rotation_world_from_camera": self.rotation_world_from_camera.tolist(),
        }


@dataclass
class BoundingBox2D:
    """One target 2D bounding-box observation in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    resolution: tuple[int, int]
    semantic_id: int | None = None
    semantic_label: str | None = None
    occlusion_ratio: float | None = None
    clipped: bool = False
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        self.resolution = _coerce_resolution(self.resolution)
        coordinates = np.asarray(
            [self.x_min, self.y_min, self.x_max, self.y_max],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("bounding-box coordinates must be finite")
        self.x_min, self.y_min, self.x_max, self.y_max = coordinates.tolist()
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("bounding-box maximums must not be smaller than minimums")
        if self.semantic_id is not None:
            self.semantic_id = int(self.semantic_id)
        if self.semantic_label is not None:
            self.semantic_label = str(self.semantic_label)
        if self.occlusion_ratio is not None:
            self.occlusion_ratio = float(self.occlusion_ratio)
            if not isfinite(self.occlusion_ratio) or not 0.0 <= self.occlusion_ratio <= 1.0:
                raise ValueError("occlusion_ratio must be between 0 and 1")
        self.clipped = bool(self.clipped)
        self.valid = bool(self.valid)
        if self.reason is not None:
            self.reason = str(self.reason)

    @property
    def center_uv(self) -> np.ndarray:
        """Return the floating-point center ``[u, v]`` of the box."""
        return np.array(
            [(self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5],
            dtype=np.float64,
        )

    @property
    def area(self) -> float:
        """Return the pixel area of the box."""
        return float((self.x_max - self.x_min) * (self.y_max - self.y_min))

    def as_dict(self) -> dict:
        """Return JSON-compatible bbox fields."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "center_uv": self.center_uv.tolist(),
            "resolution": list(self.resolution),
            "semantic_id": self.semantic_id,
            "semantic_label": self.semantic_label,
            "occlusion_ratio": self.occlusion_ratio,
            "clipped": self.clipped,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass
class CameraObservation:
    """Camera calibration paired with a target bbox from one capture."""

    camera_path: str
    calibration: CameraCalibration
    bbox: BoundingBox2D | None
    capture_id: str | int
    valid: bool | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        self.camera_path = str(self.camera_path)
        if not self.camera_path:
            raise ValueError("camera_path must not be empty")
        if not isinstance(self.calibration, CameraCalibration):
            raise TypeError("calibration must be a CameraCalibration")
        if self.calibration.camera_path != self.camera_path:
            raise ValueError("camera_path must match calibration.camera_path")
        if self.bbox is not None and not isinstance(self.bbox, BoundingBox2D):
            raise TypeError("bbox must be a BoundingBox2D or None")
        if self.valid is None:
            self.valid = self.bbox is not None and self.bbox.valid
        else:
            self.valid = bool(self.valid)
        if self.reason is not None:
            self.reason = str(self.reason)

    @property
    def center_uv(self) -> np.ndarray | None:
        """Return the observed bbox center, or ``None`` for a missing bbox."""
        return None if self.bbox is None else self.bbox.center_uv

    def as_dict(self) -> dict:
        """Return JSON-compatible observation fields."""
        return {
            "camera_path": self.camera_path,
            "capture_id": self.capture_id,
            "calibration": self.calibration.as_dict(),
            "bbox": None if self.bbox is None else self.bbox.as_dict(),
            "valid": self.valid,
            "reason": self.reason,
        }


def _row_value(row, name: str):
    """Read a named field from a structured row or mapping."""
    if isinstance(row, dict):
        if name not in row:
            raise KeyError(name)
        return row[name]
    try:
        return row[name]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise KeyError(name) from exc


def _semantic_label_matches(label_value, target_label: str) -> bool:
    """Match a Replicator semantic-label value against a target class."""
    if isinstance(label_value, dict):
        if "class" in label_value:
            return _semantic_label_matches(label_value["class"], target_label)
        return any(_semantic_label_matches(value, target_label) for value in label_value.values())
    if isinstance(label_value, (list, tuple, set)):
        return any(_semantic_label_matches(value, target_label) for value in label_value)
    return str(label_value) == target_label


def _semantic_id_matches(semantic_id, id_to_labels, target_label: str) -> bool:
    """Resolve an annotator semantic ID through either integer or string keys."""
    if not isinstance(id_to_labels, dict):
        return False
    candidates = [semantic_id, str(semantic_id)]
    try:
        candidates.append(int(semantic_id))
    except (TypeError, ValueError, OverflowError):
        pass
    for candidate in candidates:
        if candidate in id_to_labels and _semantic_label_matches(id_to_labels[candidate], target_label):
            return True
    return False


def _invalid_bbox(
    resolution: tuple[int, int],
    *,
    target_label: str,
    reason: str,
    semantic_id: int | None = None,
    occlusion_ratio: float | None = None,
    clipped: bool = False,
) -> BoundingBox2D:
    """Construct a serializable invalid bbox while preserving the reason."""
    return BoundingBox2D(
        x_min=0.0,
        y_min=0.0,
        x_max=0.0,
        y_max=0.0,
        resolution=resolution,
        semantic_id=semantic_id,
        semantic_label=target_label,
        occlusion_ratio=occlusion_ratio,
        clipped=clipped,
        valid=False,
        reason=reason,
    )


def extract_target_bbox(
    annotation_data,
    annotation_info: dict | None,
    *,
    resolution,
    target_label: str = "mannequin",
    max_occlusion_ratio: float | None = None,
    border_tolerance_px: float = 0.0,
) -> BoundingBox2D:
    """Select and validate one target bbox from Replicator annotator data.

    ``bounding_box_2d_tight`` may return multiple rows when semantic labels
    are inherited by several imageable prims. All rows matching
    ``target_label`` are unioned into one silhouette box. The returned object
    is always serializable; invalid captures carry a precise ``reason``.
    """
    validated_resolution = _coerce_resolution(resolution)
    target_label = str(target_label)
    if not target_label:
        raise ValueError("target_label must not be empty")
    if max_occlusion_ratio is not None:
        max_occlusion_ratio = float(max_occlusion_ratio)
        if not isfinite(max_occlusion_ratio) or not 0.0 <= max_occlusion_ratio <= 1.0:
            raise ValueError("max_occlusion_ratio must be between 0 and 1")
    border_tolerance_px = float(border_tolerance_px)
    if not isfinite(border_tolerance_px) or border_tolerance_px < 0.0:
        raise ValueError("border_tolerance_px must be finite and nonnegative")

    if annotation_data is None:
        return _invalid_bbox(
            validated_resolution,
            target_label=target_label,
            reason="bbox annotator returned no data",
        )

    try:
        rows = list(annotation_data)
    except TypeError as exc:
        raise ValueError("annotation_data must be an iterable of bbox rows") from exc
    if not rows:
        return _invalid_bbox(
            validated_resolution,
            target_label=target_label,
            reason=f"target bbox not found for semantic label {target_label!r}",
        )

    info = annotation_info or {}
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    target_rows = []
    for row in rows:
        try:
            semantic_id = _row_value(row, "semanticId")
        except KeyError:
            continue
        if _semantic_id_matches(semantic_id, id_to_labels, target_label):
            target_rows.append(row)

    if not target_rows:
        return _invalid_bbox(
            validated_resolution,
            target_label=target_label,
            reason=f"target bbox not found for semantic label {target_label!r}",
        )

    parsed_rows = []
    for row in target_rows:
        try:
            semantic_id = int(_row_value(row, "semanticId"))
            coordinates = np.asarray(
                [
                    _row_value(row, "x_min"),
                    _row_value(row, "y_min"),
                    _row_value(row, "x_max"),
                    _row_value(row, "y_max"),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return _invalid_bbox(
                validated_resolution,
                target_label=target_label,
                reason=f"target bbox row is malformed: {exc}",
            )
        if not np.all(np.isfinite(coordinates)):
            return _invalid_bbox(
                validated_resolution,
                target_label=target_label,
                semantic_id=semantic_id,
                reason="target bbox contains nonfinite coordinates",
            )
        x_min, y_min, x_max, y_max = coordinates.tolist()
        if x_max <= x_min or y_max <= y_min:
            return _invalid_bbox(
                validated_resolution,
                target_label=target_label,
                semantic_id=semantic_id,
                reason="target bbox has zero or negative area",
            )
        try:
            occlusion = _row_value(row, "occlusionRatio")
        except KeyError:
            occlusion = None
        if occlusion is not None:
            try:
                occlusion = float(occlusion)
            except (TypeError, ValueError, OverflowError) as exc:
                return _invalid_bbox(
                    validated_resolution,
                    target_label=target_label,
                    semantic_id=semantic_id,
                    reason=f"target bbox occlusion ratio is malformed: {exc}",
                )
            if not isfinite(occlusion) or not 0.0 <= occlusion <= 1.0:
                return _invalid_bbox(
                    validated_resolution,
                    target_label=target_label,
                    semantic_id=semantic_id,
                    reason="target bbox occlusion ratio is outside [0, 1]",
                )
        parsed_rows.append((semantic_id, coordinates, occlusion))

    union_minimum = np.min(np.stack([row[1][:2] for row in parsed_rows]), axis=0)
    union_maximum = np.max(np.stack([row[1][2:] for row in parsed_rows]), axis=0)
    x_min, y_min = union_minimum.tolist()
    x_max, y_max = union_maximum.tolist()
    width, height = validated_resolution
    clipped = (
        x_min <= border_tolerance_px
        or y_min <= border_tolerance_px
        or x_max >= width - border_tolerance_px
        or y_max >= height - border_tolerance_px
    )
    if clipped:
        reason = "target bbox is clipped by the image boundary"
    else:
        reason = None

    occlusions = [row[2] for row in parsed_rows if row[2] is not None]
    occlusion_ratio = max(occlusions) if occlusions else None
    if reason is None and max_occlusion_ratio is not None and occlusion_ratio is not None:
        if occlusion_ratio > max_occlusion_ratio:
            reason = (
                f"target bbox occlusion ratio {occlusion_ratio:.6g} exceeds "
                f"limit {max_occlusion_ratio:.6g}"
            )

    semantic_ids = {row[0] for row in parsed_rows}
    return BoundingBox2D(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        resolution=validated_resolution,
        semantic_id=next(iter(semantic_ids)) if len(semantic_ids) == 1 else None,
        semantic_label=target_label,
        occlusion_ratio=occlusion_ratio,
        clipped=clipped,
        valid=reason is None,
        reason=reason,
    )


@dataclass
class CameraRay:
    """A world-space ray emitted from one camera."""

    camera_path: str
    origin_world: np.ndarray
    direction_world: np.ndarray
    target_distance_m: float | None = None

    def __post_init__(self) -> None:
        self.origin_world = _coerce_vector3(self.origin_world, name="origin_world")
        self.direction_world = normalize(self.direction_world, name="direction_world")
        if self.target_distance_m is not None:
            self.target_distance_m = float(self.target_distance_m)
            if not isfinite(self.target_distance_m) or self.target_distance_m <= _VECTOR_EPSILON:
                raise ValueError("target_distance_m must be finite and positive when provided")

    def point_at(self, distance_m: float) -> np.ndarray:
        """Return a point on the ray at the requested nonnegative distance."""
        distance = float(distance_m)
        if not isfinite(distance) or distance < 0.0:
            raise ValueError("distance_m must be finite and nonnegative")
        return self.origin_world + distance * self.direction_world

    def as_dict(self) -> dict:
        """Return JSON-compatible fields for this bearing ray."""
        return {
            "camera_path": self.camera_path,
            "origin_world": self.origin_world.tolist(),
            "direction_world": self.direction_world.tolist(),
            "target_distance_m": self.target_distance_m,
        }


def compute_display_ray_length_to_ground(
    rays: Sequence[CameraRay],
    *,
    ground_plane_z: float = 0.0,
    extra_length_m: float = 1.0,
    fallback_length_m: float = 10.0,
) -> float:
    """Return one shared display length that carries downward rays into ground.

    The visualizer uses one common length for ground-truth and YOLO rays so
    their directions remain comparable.  The common length is based on the
    farthest positive intersection with the horizontal ground plane, plus a
    small margin below the plane.  Rays that do not point toward the plane
    are ignored; ``fallback_length_m`` keeps the helper useful for isolated
    visualization tests and upward-pointing rays.
    """
    if not isfinite(float(ground_plane_z)):
        raise ValueError("ground_plane_z must be finite")
    extra_length = float(extra_length_m)
    fallback_length = float(fallback_length_m)
    if not isfinite(extra_length) or extra_length < 0.0:
        raise ValueError("extra_length_m must be finite and nonnegative")
    if not isfinite(fallback_length) or fallback_length <= 0.0:
        raise ValueError("fallback_length_m must be finite and positive")

    ground_intersections = []
    for ray in rays:
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
        direction_z = float(ray.direction_world[2])
        if direction_z >= -_VECTOR_EPSILON:
            continue
        distance = (float(ground_plane_z) - float(ray.origin_world[2])) / direction_z
        if isfinite(distance) and distance > _VECTOR_EPSILON:
            ground_intersections.append(distance)

    if not ground_intersections:
        return fallback_length
    return max(fallback_length, max(ground_intersections) + extra_length)


def build_camera_ray_from_pixel(
    calibration: CameraCalibration,
    pixel_uv,
) -> CameraRay:
    """Back-project one image pixel into a world-space bearing ray.

    Replicator pixels use ``(u, v)`` with ``u`` increasing rightward and
    ``v`` increasing downward.  USD camera coordinates use ``+X`` right,
    ``+Y`` up, and look along local ``-Z``; the sign conversion is therefore
    applied before the calibrated camera-to-world rotation.
    """
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("calibration must be a CameraCalibration")
    pixel = _coerce_vector2(pixel_uv, name="pixel_uv")
    homogeneous_pixel = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    image_coordinates = np.linalg.solve(calibration.intrinsic_matrix, homogeneous_pixel)
    direction_camera = np.array(
        [image_coordinates[0], -image_coordinates[1], -image_coordinates[2]],
        dtype=np.float64,
    )
    direction_world = calibration.rotation_world_from_camera @ direction_camera
    return CameraRay(
        camera_path=calibration.camera_path,
        origin_world=calibration.origin_world,
        direction_world=direction_world,
    )


def build_camera_ray_from_observation(observation: CameraObservation) -> CameraRay:
    """Build one bearing ray from a valid target bbox observation."""
    if not isinstance(observation, CameraObservation):
        raise TypeError("observation must be a CameraObservation")
    if observation.bbox is None:
        raise ValueError(f"camera {observation.camera_path!r} has no target bbox")
    if not observation.valid or not observation.bbox.valid:
        reason = observation.reason or observation.bbox.reason or "invalid target bbox"
        raise ValueError(f"camera {observation.camera_path!r} observation is invalid: {reason}")
    return build_camera_ray_from_pixel(observation.calibration, observation.bbox.center_uv)


def build_rays_from_available_observations(
    observations: Sequence[CameraObservation],
) -> list[CameraRay]:
    """Build rays for every visible camera and retain failures per observation.

    A target can be outside one camera's view while remaining visible in the
    others. Invalid observations are skipped, not fatal to the remaining ray
    construction. The caller can then decide whether the available geometry
    is sufficient for fusion.
    """
    rays = []
    for observation in observations:
        if not observation.valid:
            continue
        try:
            rays.append(build_camera_ray_from_observation(observation))
        except (TypeError, ValueError) as exc:
            observation.valid = False
            observation.reason = f"failed to construct bearing ray: {exc}"
    return rays


@dataclass
class FusionResult:
    """Result and diagnostics from least-squares ray intersection."""

    fused_position_world: np.ndarray | None
    rms_residual_m: float | None
    rank: int
    condition_number: float
    valid: bool
    reason: str | None = None
    ray_diagnostics: list[dict] = field(default_factory=list)
    pairwise_angles_deg: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return JSON-compatible fusion fields and geometry diagnostics."""
        condition_number = (
            float(self.condition_number) if isfinite(self.condition_number) else None
        )
        pairwise_angles = [float(angle) for angle in self.pairwise_angles_deg]
        return {
            "fused_position_world": (
                None if self.fused_position_world is None else self.fused_position_world.tolist()
            ),
            "rms_residual_m": (
                None if self.rms_residual_m is None else float(self.rms_residual_m)
            ),
            "rank": int(self.rank),
            "condition_number": condition_number,
            "valid": bool(self.valid),
            "reason": self.reason,
            "ray_diagnostics": self.ray_diagnostics,
            "pairwise_angles_deg": pairwise_angles,
            "min_pairwise_angle_deg": min(pairwise_angles) if pairwise_angles else None,
            "max_pairwise_angle_deg": max(pairwise_angles) if pairwise_angles else None,
        }


@dataclass
class FusionEvaluation:
    """Ground-truth comparison kept separate from ray estimation."""

    target_world: np.ndarray
    error_m: float | None
    valid: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        self.target_world = _coerce_vector3(self.target_world, name="target_world")
        if self.error_m is not None:
            self.error_m = float(self.error_m)
            if not isfinite(self.error_m):
                raise ValueError("error_m must be finite when provided")
        self.valid = bool(self.valid)
        if self.reason is not None:
            self.reason = str(self.reason)

    def as_dict(self) -> dict:
        """Return JSON-compatible evaluation fields."""
        return {
            "target_world": self.target_world.tolist(),
            "error_m": None if self.error_m is None else float(self.error_m),
            "valid": self.valid,
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


def draw_fused_rays(
    rays: Sequence[CameraRay],
    fused_position_world,
    *,
    truth_world=None,
    clear_existing: bool = True,
    line_thickness: float = 3.0,
    target_size: float = 12.0,
    camera_colors: Sequence[tuple[float, float, float, float]] = DEFAULT_CAMERA_COLORS,
    target_color: tuple[float, float, float, float] = DEFAULT_TARGET_COLOR,
    truth_color: tuple[float, float, float, float] = DEFAULT_TRUTH_EVALUATION_COLOR,
) -> dict:
    """Draw inferred rays to their closest points on the fused estimate.

    The fused marker is the primary estimate. ``truth_world`` is optional and
    is rendered only as a separate evaluation marker; it never determines ray
    endpoints or the fused estimate.
    """
    if not rays:
        raise ValueError("rays must contain at least one CameraRay")
    if not isfinite(line_thickness) or line_thickness <= 0.0:
        raise ValueError("line_thickness must be finite and positive")
    if not isfinite(target_size) or target_size <= 0.0:
        raise ValueError("target_size must be finite and positive")
    if not camera_colors:
        raise ValueError("camera_colors must contain at least one RGBA color")

    fused = _coerce_vector3(fused_position_world, name="fused_position_world")
    closest_points = []
    for ray in rays:
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
        forward_distance = float(np.dot(fused - ray.origin_world, ray.direction_world))
        closest_points.append(ray.point_at(max(0.0, forward_distance)).tolist())

    truth = None if truth_world is None else _coerce_vector3(truth_world, name="truth_world")
    from isaacsim.util.debug_draw import _debug_draw

    draw_interface = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw_interface.clear_lines()
        draw_interface.clear_points()

    start_points = [ray.origin_world.tolist() for ray in rays]
    colors = [list(camera_colors[index % len(camera_colors)]) for index in range(len(rays))]
    draw_interface.draw_lines(start_points, closest_points, colors, [line_thickness] * len(rays))
    point_positions = [fused.tolist()]
    point_colors = [list(target_color)]
    point_sizes = [target_size]
    if truth is not None:
        point_positions.append(truth.tolist())
        point_colors.append(list(truth_color))
        point_sizes.append(target_size)
    draw_interface.draw_points(point_positions, point_colors, point_sizes)
    return {
        "ray_count": len(rays),
        "fused_position_world": fused,
        "closest_points_world": np.asarray(closest_points, dtype=np.float64),
        "truth_world": truth,
        "camera_colors": colors,
    }


def draw_comparison_rays(
    ground_truth_rays: Sequence[CameraRay] | None = None,
    yolo_rays: Sequence[CameraRay] | None = None,
    *,
    ground_truth_fused_position_world=None,
    yolo_fused_position_world=None,
    truth_world=None,
    clear_existing: bool = True,
    ray_length: float = 10.0,
    line_thickness: float = 3.0,
    point_size: float = 12.0,
    ground_truth_ray_color: tuple[float, float, float, float] = DEFAULT_GROUND_TRUTH_RAY_COLOR,
    yolo_ray_color: tuple[float, float, float, float] = DEFAULT_YOLO_RAY_COLOR,
    ground_truth_fused_color: tuple[float, float, float, float] = DEFAULT_GROUND_TRUTH_FUSED_COLOR,
    yolo_fused_color: tuple[float, float, float, float] = DEFAULT_YOLO_FUSED_COLOR,
    truth_color: tuple[float, float, float, float] = DEFAULT_TRUTH_EVALUATION_COLOR,
) -> dict:
    """Draw ground-truth and YOLO rays with a shared display length.

    The two ray sets are emitted from their camera origins to the same fixed
    ``ray_length`` so their directions can be compared directly. Fused
    positions and the exact world target are rendered as separate markers;
    neither marker determines a ray endpoint.
    """
    ground_truth = [] if ground_truth_rays is None else list(ground_truth_rays)
    yolo = [] if yolo_rays is None else list(yolo_rays)
    for ray in [*ground_truth, *yolo]:
        if not isinstance(ray, CameraRay):
            raise TypeError("ground_truth_rays and yolo_rays must contain only CameraRay instances")
    if not ground_truth and not yolo and all(
        position is None
        for position in (
            ground_truth_fused_position_world,
            yolo_fused_position_world,
            truth_world,
        )
    ):
        raise ValueError("at least one ray or marker position is required")
    if not isfinite(ray_length) or ray_length <= 0.0:
        raise ValueError("ray_length must be finite and positive")
    if not isfinite(line_thickness) or line_thickness <= 0.0:
        raise ValueError("line_thickness must be finite and positive")
    if not isfinite(point_size) or point_size <= 0.0:
        raise ValueError("point_size must be finite and positive")

    ground_truth_endpoints = [ray.point_at(ray_length).tolist() for ray in ground_truth]
    yolo_endpoints = [ray.point_at(ray_length).tolist() for ray in yolo]
    ground_truth_fused = (
        None
        if ground_truth_fused_position_world is None
        else _coerce_vector3(
            ground_truth_fused_position_world,
            name="ground_truth_fused_position_world",
        )
    )
    yolo_fused = (
        None
        if yolo_fused_position_world is None
        else _coerce_vector3(yolo_fused_position_world, name="yolo_fused_position_world")
    )
    truth = None if truth_world is None else _coerce_vector3(truth_world, name="truth_world")

    from isaacsim.util.debug_draw import _debug_draw

    draw_interface = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw_interface.clear_lines()
        draw_interface.clear_points()

    if ground_truth:
        draw_interface.draw_lines(
            [ray.origin_world.tolist() for ray in ground_truth],
            ground_truth_endpoints,
            [list(ground_truth_ray_color)] * len(ground_truth),
            [line_thickness] * len(ground_truth),
        )
    if yolo:
        draw_interface.draw_lines(
            [ray.origin_world.tolist() for ray in yolo],
            yolo_endpoints,
            [list(yolo_ray_color)] * len(yolo),
            [line_thickness] * len(yolo),
        )

    point_positions = []
    point_colors = []
    point_sizes = []
    if ground_truth_fused is not None:
        point_positions.append(ground_truth_fused.tolist())
        point_colors.append(list(ground_truth_fused_color))
        point_sizes.append(point_size)
    if yolo_fused is not None:
        point_positions.append(yolo_fused.tolist())
        point_colors.append(list(yolo_fused_color))
        point_sizes.append(point_size)
    if truth is not None:
        point_positions.append(truth.tolist())
        point_colors.append(list(truth_color))
        point_sizes.append(point_size)
    if point_positions:
        draw_interface.draw_points(point_positions, point_colors, point_sizes)

    def endpoint_array(points: list[list[float]]) -> np.ndarray:
        return (
            np.empty((0, 3), dtype=np.float64)
            if not points
            else np.asarray(points, dtype=np.float64)
        )

    return {
        "ground_truth_ray_count": len(ground_truth),
        "yolo_ray_count": len(yolo),
        "ray_length": float(ray_length),
        "ground_truth_endpoints_world": endpoint_array(ground_truth_endpoints),
        "yolo_endpoints_world": endpoint_array(yolo_endpoints),
        "ground_truth_fused_position_world": ground_truth_fused,
        "yolo_fused_position_world": yolo_fused,
        "truth_world": truth,
    }


def clear_debug_draw() -> None:
    """Clear transient ray and marker geometry from the Isaac viewport."""
    from isaacsim.util.debug_draw import _debug_draw

    draw_interface = _debug_draw.acquire_debug_draw_interface()
    draw_interface.clear_lines()
    draw_interface.clear_points()


def _invalid_result(
    reason: str,
    *,
    rank: int = 0,
    condition_number: float = float("inf"),
    fused_position_world: np.ndarray | None = None,
    rms_residual_m: float | None = None,
    ray_diagnostics: list[dict] | None = None,
    pairwise_angles_deg: list[float] | None = None,
) -> FusionResult:
    return FusionResult(
        fused_position_world=fused_position_world,
        rms_residual_m=rms_residual_m,
        rank=rank,
        condition_number=condition_number,
        valid=False,
        reason=reason,
        ray_diagnostics=[] if ray_diagnostics is None else ray_diagnostics,
        pairwise_angles_deg=[] if pairwise_angles_deg is None else pairwise_angles_deg,
    )


def _pairwise_ray_angles_deg(rays: Sequence[CameraRay]) -> list[float]:
    """Return acute pairwise ray angles, where zero means parallel geometry."""
    angles = []
    for first_index in range(len(rays)):
        first_direction = normalize(
            rays[first_index].direction_world,
            name=f"{rays[first_index].camera_path} direction_world",
        )
        for second_index in range(first_index + 1, len(rays)):
            second_direction = normalize(
                rays[second_index].direction_world,
                name=f"{rays[second_index].camera_path} direction_world",
            )
            cosine = float(np.clip(abs(np.dot(first_direction, second_direction)), 0.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
    return angles


def fuse_rays(
    rays: Sequence[CameraRay],
    *,
    rank_tolerance: float = _DEFAULT_RANK_TOLERANCE,
    condition_limit: float = _DEFAULT_CONDITION_LIMIT,
) -> FusionResult:
    """Estimate the common point of camera rays by least-squares intersection.

    For each ray ``(o_i, d_i)``, the perpendicular projection matrix is
    ``P_i = I - d_i d_i^T``.  The returned estimate solves
    ``sum(P_i) x = sum(P_i o_i)``.  This estimator consumes only camera rays;
    compare against world truth separately with ``evaluate_fusion``.
    """
    if len(rays) < 2:
        return _invalid_result("at least two rays are required")
    if not isfinite(rank_tolerance) or rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be finite and positive")
    if not isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive")

    for ray in rays:
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
    pairwise_angles_deg = _pairwise_ray_angles_deg(rays)

    system_matrix = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for ray in rays:
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
            pairwise_angles_deg=pairwise_angles_deg,
        )
    if condition_number > condition_limit:
        return _invalid_result(
            "ray geometry is ill-conditioned",
            rank=rank,
            condition_number=condition_number,
            pairwise_angles_deg=pairwise_angles_deg,
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
            pairwise_angles_deg=pairwise_angles_deg,
        )

    ray_diagnostics = []
    squared_distances = []
    for ray in rays:
        offset = fused_position - ray.origin_world
        forward_distance = float(np.dot(offset, ray.direction_world))
        perpendicular_offset = offset - np.dot(offset, ray.direction_world) * ray.direction_world
        residual_m = float(np.linalg.norm(perpendicular_offset))
        squared_distances.append(residual_m * residual_m)
        ray_diagnostics.append(
            {
                "camera_path": ray.camera_path,
                "forward_distance_m": forward_distance,
                "perpendicular_residual_m": residual_m,
            }
        )
    rms_residual = float(np.sqrt(np.mean(squared_distances)))

    if any(
        not isfinite(diagnostic["forward_distance_m"])
        or diagnostic["forward_distance_m"] <= _VECTOR_EPSILON
        for diagnostic in ray_diagnostics
    ):
        return _invalid_result(
            "ray solution lies behind one or more cameras",
            rank=rank,
            condition_number=condition_number,
            fused_position_world=fused_position,
            rms_residual_m=rms_residual,
            ray_diagnostics=ray_diagnostics,
            pairwise_angles_deg=pairwise_angles_deg,
        )

    return FusionResult(
        fused_position_world=fused_position,
        rms_residual_m=rms_residual,
        rank=rank,
        condition_number=condition_number,
        valid=True,
        ray_diagnostics=ray_diagnostics,
        pairwise_angles_deg=pairwise_angles_deg,
    )


def evaluate_fusion(result: FusionResult, target_world) -> FusionEvaluation:
    """Compare a completed fusion result with truth without changing it."""
    if not isinstance(result, FusionResult):
        raise TypeError("result must be a FusionResult")
    target = _coerce_vector3(target_world, name="target_world")
    if not result.valid or result.fused_position_world is None:
        return FusionEvaluation(
            target_world=target,
            error_m=None,
            valid=False,
            reason=result.reason or "fusion result is invalid",
        )
    error_m = float(np.linalg.norm(result.fused_position_world - target))
    return FusionEvaluation(target_world=target, error_m=error_m, valid=True)


def build_schema_v2_record(
    *,
    scene_index: int,
    capture_id: str | int,
    background_path: str,
    target_prim_path: str,
    resolution,
    target_label: str,
    rt_subframes: int,
    observations: Sequence[CameraObservation],
    rays: Sequence[CameraRay],
    fusion_result: FusionResult,
    fusion_evaluation: FusionEvaluation | None,
    settled: bool,
    image_paths: Sequence[str] | None = None,
    training_image_paths: Sequence[str] | None = None,
    raw_image_paths: Sequence[str] | None = None,
    raw_bbox_paths: Sequence[str] | None = None,
    raw_camera_params_paths: Sequence[str] | None = None,
) -> dict:
    """Build schema-v2 output with observations and inferred geometry separate."""
    validated_resolution = _coerce_resolution(resolution)
    if len(observations) != 4:
        raise ValueError("schema-v2 records require exactly four camera observations")
    path_fields = {
        "image_paths": image_paths,
        "training_image_paths": training_image_paths,
        "raw_image_paths": raw_image_paths,
        "raw_bbox_paths": raw_bbox_paths,
        "raw_camera_params_paths": raw_camera_params_paths,
    }
    for field_name, paths in path_fields.items():
        if paths is not None and len(paths) != len(observations):
            raise ValueError(f"{field_name} must contain one path per camera observation")
    if not isinstance(fusion_result, FusionResult):
        raise TypeError("fusion_result must be a FusionResult")
    if fusion_evaluation is not None and not isinstance(fusion_evaluation, FusionEvaluation):
        raise TypeError("fusion_evaluation must be a FusionEvaluation or None")
    if int(rt_subframes) <= 0:
        raise ValueError("rt_subframes must be positive")

    ray_by_camera = {}
    for ray in rays:
        if not isinstance(ray, CameraRay):
            raise TypeError("rays must contain only CameraRay instances")
        if ray.camera_path in ray_by_camera:
            raise ValueError(f"duplicate inferred ray for camera {ray.camera_path!r}")
        ray_by_camera[ray.camera_path] = ray

    observation_paths = set()
    inferred_rays = []
    observation_dicts = []
    for observation_index, observation in enumerate(observations):
        if not isinstance(observation, CameraObservation):
            raise TypeError("observations must contain only CameraObservation instances")
        if observation.camera_path in observation_paths:
            raise ValueError(f"duplicate observation for camera {observation.camera_path!r}")
        observation_paths.add(observation.camera_path)
        observation_dict = observation.as_dict()
        if image_paths is not None:
            observation_dict["image_path"] = str(image_paths[observation_index])
        if training_image_paths is not None:
            observation_dict["training_image_path"] = str(
                training_image_paths[observation_index]
            )
        if raw_image_paths is not None:
            observation_dict["raw_image_path"] = str(raw_image_paths[observation_index])
        if raw_bbox_paths is not None:
            observation_dict["raw_bbox_path"] = str(raw_bbox_paths[observation_index])
        if raw_camera_params_paths is not None:
            observation_dict["raw_camera_params_path"] = str(
                raw_camera_params_paths[observation_index]
            )
        observation_dicts.append(observation_dict)
        ray = ray_by_camera.pop(observation.camera_path, None)
        reason = observation.reason
        if not observation.valid:
            reason = reason or (None if observation.bbox is None else observation.bbox.reason)
            reason = reason or "camera observation is invalid"
        elif ray is None:
            reason = reason or "valid camera observation did not produce a ray"
        inferred_rays.append(
            {
                "camera_path": observation.camera_path,
                "capture_id": observation.capture_id,
                "valid": bool(observation.valid and ray is not None),
                "ray": None if ray is None else ray.as_dict(),
                "reason": reason,
            }
        )
    if ray_by_camera:
        unexpected = ", ".join(sorted(ray_by_camera))
        raise ValueError(f"inferred rays have no matching observations: {unexpected}")

    return {
        "schema_version": 2,
        "capture": {
            "scene_index": int(scene_index),
            "capture_id": capture_id,
            "background_path": str(background_path),
            "target_prim_path": str(target_prim_path),
            "resolution": list(validated_resolution),
            "target_label": str(target_label),
            "rt_subframes": int(rt_subframes),
            "settled": bool(settled),
            "camera_count": len(observation_dicts),
            "valid_camera_count": sum(item["valid"] for item in inferred_rays),
            "fusion_source": "bounding_box_centers",
        },
        "camera_observations": observation_dicts,
        "inferred_rays": inferred_rays,
        "fusion": fusion_result.as_dict(),
        "ground_truth_evaluation": (
            None if fusion_evaluation is None else fusion_evaluation.as_dict()
        ),
    }
