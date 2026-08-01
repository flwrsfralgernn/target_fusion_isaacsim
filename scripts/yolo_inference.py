"""Model loading and target-class validation for live YOLO inference.

This module deliberately has no Isaac Sim imports.  The capture pipeline can
resolve and validate a local checkpoint independently, while the Ultralytics
dependency is imported only when :func:`load_yolo_model` is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

try:
    from target_fusion import (
        BoundingBox2D,
        CameraCalibration,
        CameraObservation,
        CameraRay,
        FusionEvaluation,
        FusionResult,
        build_rays_from_available_observations,
        evaluate_fusion,
        fuse_rays,
        normalize,
    )
except ModuleNotFoundError:
    from scripts.target_fusion import (
        BoundingBox2D,
        CameraCalibration,
        CameraObservation,
        CameraRay,
        FusionEvaluation,
        FusionResult,
        build_rays_from_available_observations,
        evaluate_fusion,
        fuse_rays,
        normalize,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SUPPORTED_MODEL_ALIASES = ("yolo11n.pt", "yolo26n.pt")


@dataclass(frozen=True)
class YoloModelInfo:
    """Validated metadata needed by the later capture/inference pipeline."""

    reference: str
    path: Path
    task: str
    class_names: tuple[str, ...]
    target_label: str
    target_class_id: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible model metadata."""
        return {
            "reference": self.reference,
            "path": str(self.path),
            "task": self.task,
            "class_names": list(self.class_names),
            "target_label": self.target_label,
            "target_class_id": self.target_class_id,
        }


@dataclass(frozen=True)
class LoadedYoloModel:
    """A loaded Ultralytics model paired with validated metadata."""

    model: Any
    info: YoloModelInfo


@dataclass(frozen=True)
class YoloDetection:
    """One selected target detection in pixel coordinates."""

    bbox: BoundingBox2D
    confidence: float
    class_id: int
    class_name: str
    raw_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BoundingBox2D):
            raise TypeError("bbox must be a BoundingBox2D")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        class_id = int(self.class_id)
        if class_id < 0:
            raise ValueError("class_id must be nonnegative")
        class_name = str(self.class_name)
        if not class_name:
            raise ValueError("class_name must not be empty")
        coordinates = tuple(float(value) for value in self.raw_xyxy)
        if len(coordinates) != 4 or not all(isfinite(value) for value in coordinates):
            raise ValueError("raw_xyxy must contain four finite coordinates")
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            raise ValueError("raw_xyxy must have positive area")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "raw_xyxy", coordinates)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible detection fields."""
        return {
            "bbox": self.bbox.as_dict(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "raw_xyxy": list(self.raw_xyxy),
        }


@dataclass(frozen=True)
class YoloInferenceResult:
    """One-frame target-detection result, including miss diagnostics."""

    detection: YoloDetection | None
    frame_resolution: tuple[int, int]
    inference_time_ms: float
    total_detection_count: int
    target_candidate_count: int
    qualified_candidate_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        width, height = (int(value) for value in self.frame_resolution)
        if width <= 0 or height <= 0:
            raise ValueError("frame_resolution must contain positive dimensions")
        inference_time_ms = float(self.inference_time_ms)
        if not isfinite(inference_time_ms) or inference_time_ms < 0.0:
            raise ValueError("inference_time_ms must be finite and nonnegative")
        object.__setattr__(self, "frame_resolution", (width, height))
        object.__setattr__(self, "inference_time_ms", inference_time_ms)
        for field_name in (
            "total_detection_count",
            "target_candidate_count",
            "qualified_candidate_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(self, field_name, value)
        if self.target_candidate_count > self.total_detection_count:
            raise ValueError("target_candidate_count cannot exceed total_detection_count")
        if self.qualified_candidate_count > self.target_candidate_count:
            raise ValueError("qualified_candidate_count cannot exceed target_candidate_count")
        if self.reason is not None:
            object.__setattr__(self, "reason", str(self.reason))

    @property
    def valid(self) -> bool:
        """Whether a target detection was selected."""
        return self.detection is not None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible inference fields."""
        return {
            "valid": self.valid,
            "detection": None if self.detection is None else self.detection.as_dict(),
            "frame_resolution": list(self.frame_resolution),
            "inference_time_ms": self.inference_time_ms,
            "total_detection_count": self.total_detection_count,
            "target_candidate_count": self.target_candidate_count,
            "qualified_candidate_count": self.qualified_candidate_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObservationFusion:
    """Rays, fusion, and optional world-target evaluation for one source."""

    observations: tuple[CameraObservation, ...]
    rays: tuple[CameraRay, ...]
    fusion: FusionResult
    evaluation: FusionEvaluation | None
    valid_camera_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible source-fusion fields."""
        return {
            "observations": [observation.as_dict() for observation in self.observations],
            "rays": [ray.as_dict() for ray in self.rays],
            "valid_camera_count": self.valid_camera_count,
            "fusion": self.fusion.as_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.as_dict(),
        }


@dataclass(frozen=True)
class CameraComparison:
    """Per-camera bbox and ray differences between ground truth and YOLO."""

    camera_path: str
    ground_truth_valid: bool
    yolo_valid: bool
    ground_truth_reason: str | None
    yolo_reason: str | None
    yolo_confidence: float | None
    bbox_iou: float | None
    center_error_px: float | None
    center_error_normalized: float | None
    ray_angle_deg: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible camera comparison fields."""
        return {
            "camera_path": self.camera_path,
            "ground_truth_valid": self.ground_truth_valid,
            "yolo_valid": self.yolo_valid,
            "ground_truth_reason": self.ground_truth_reason,
            "yolo_reason": self.yolo_reason,
            "yolo_confidence": self.yolo_confidence,
            "bbox_iou": self.bbox_iou,
            "center_error_px": self.center_error_px,
            "center_error_normalized": self.center_error_normalized,
            "ray_angle_deg": self.ray_angle_deg,
        }


@dataclass(frozen=True)
class FusionComparison:
    """Complete ground-truth-vs-YOLO comparison for one capture."""

    ground_truth: ObservationFusion
    yolo: ObservationFusion
    cameras: tuple[CameraComparison, ...]
    fused_position_delta_m: float | None

    def metrics(self) -> dict[str, Any]:
        """Return compact aggregate comparison metrics."""
        ious = [camera.bbox_iou for camera in self.cameras if camera.bbox_iou is not None]
        center_errors = [
            camera.center_error_px
            for camera in self.cameras
            if camera.center_error_px is not None
        ]
        normalized_center_errors = [
            camera.center_error_normalized
            for camera in self.cameras
            if camera.center_error_normalized is not None
        ]
        ray_angles = [
            camera.ray_angle_deg
            for camera in self.cameras
            if camera.ray_angle_deg is not None
        ]

        def mean_or_none(values: list[float | None]) -> float | None:
            return None if not values else float(np.mean(values))

        def max_or_none(values: list[float | None]) -> float | None:
            return None if not values else float(max(values))

        return {
            "camera_count": len(self.cameras),
            "ground_truth_valid_camera_count": self.ground_truth.valid_camera_count,
            "yolo_valid_camera_count": self.yolo.valid_camera_count,
            "bbox_comparison_count": len(ious),
            "mean_bbox_iou": mean_or_none(ious),
            "mean_center_error_px": mean_or_none(center_errors),
            "max_center_error_px": max_or_none(center_errors),
            "mean_center_error_normalized": mean_or_none(normalized_center_errors),
            "mean_ray_angle_deg": mean_or_none(ray_angles),
            "max_ray_angle_deg": max_or_none(ray_angles),
            "ray_comparison_count": len(ray_angles),
            "fused_position_delta_m": self.fused_position_delta_m,
            "ground_truth_fusion_valid": self.ground_truth.fusion.valid,
            "yolo_fusion_valid": self.yolo.fusion.valid,
            "ground_truth_rms_residual_m": self.ground_truth.fusion.rms_residual_m,
            "yolo_rms_residual_m": self.yolo.fusion.rms_residual_m,
            "ground_truth_target_error_m": (
                None
                if self.ground_truth.evaluation is None
                else self.ground_truth.evaluation.error_m
            ),
            "yolo_target_error_m": (
                None if self.yolo.evaluation is None else self.yolo.evaluation.error_m
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible comparison fields."""
        return {
            "ground_truth": self.ground_truth.as_dict(),
            "yolo": self.yolo.as_dict(),
            "cameras": [camera.as_dict() for camera in self.cameras],
            "metrics": self.metrics(),
        }


def model_alias_paths(*, project_dir: Path = PROJECT_DIR) -> dict[str, Path]:
    """Return the supported model aliases and their repository-local paths."""
    root = Path(project_dir).expanduser().resolve()
    return {alias: root / alias for alias in SUPPORTED_MODEL_ALIASES}


def resolve_model_path(
    model_reference: str | Path,
    *,
    project_dir: Path = PROJECT_DIR,
    relative_to: Path | None = None,
) -> Path:
    """Resolve a supported alias or explicit local ``.pt`` checkpoint path.

    The bare aliases ``yolo11n.pt`` and ``yolo26n.pt`` always resolve relative
    to the repository root.  Other relative paths resolve relative to the
    caller's current directory unless ``relative_to`` is supplied.
    """
    reference = str(model_reference).strip()
    if not reference:
        raise ValueError("model_reference must not be empty")

    aliases = model_alias_paths(project_dir=project_dir)
    if reference in aliases:
        path = aliases[reference]
    else:
        path = Path(reference).expanduser()
        if not path.is_absolute():
            base_dir = Path.cwd() if relative_to is None else Path(relative_to).expanduser()
            path = base_dir / path

    path = path.resolve()
    if path.suffix.lower() != ".pt":
        raise ValueError(f"YOLO model must be a local .pt checkpoint: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model checkpoint does not exist: {path}")
    return path


def _normalize_class_names(names: Any) -> tuple[str, ...]:
    """Normalize Ultralytics list/dict class metadata into class-id order."""
    if isinstance(names, Mapping):
        indexed_names = {}
        for key, value in names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"model class id is not an integer: {key!r}") from exc
            if class_id < 0:
                raise ValueError(f"model class id must be nonnegative: {class_id}")
            indexed_names[class_id] = str(value)
        if not indexed_names:
            raise ValueError("model does not define any classes")
        expected_ids = set(range(max(indexed_names) + 1))
        if set(indexed_names) != expected_ids:
            raise ValueError("model class ids must be contiguous starting at zero")
        return tuple(indexed_names[index] for index in range(len(indexed_names)))

    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        normalized = tuple(str(value) for value in names)
        if not normalized:
            raise ValueError("model does not define any classes")
        return normalized

    raise ValueError("model class metadata must be a sequence or mapping")


def _build_model_info(
    model: Any,
    *,
    reference: str,
    path: Path,
    target_label: str,
) -> YoloModelInfo:
    """Validate model task/classes and build serializable metadata."""
    target_label = str(target_label).strip()
    if not target_label:
        raise ValueError("target_label must not be empty")

    task = str(getattr(model, "task", "") or "").strip().lower()
    if task != "detect":
        raise ValueError(
            f"YOLO model {path} is not a detection model; reported task is {task!r}"
        )

    class_names = _normalize_class_names(getattr(model, "names", None))
    matching_ids = [class_id for class_id, name in enumerate(class_names) if name == target_label]
    if not matching_ids:
        available = ", ".join(repr(name) for name in class_names)
        raise ValueError(
            f"YOLO model {path} does not contain target class {target_label!r}; "
            f"available classes: {available}"
        )

    return YoloModelInfo(
        reference=reference,
        path=path,
        task=task,
        class_names=class_names,
        target_label=target_label,
        target_class_id=matching_ids[0],
    )


def load_yolo_model(
    model_reference: str | Path,
    *,
    target_label: str = "mannequin",
    project_dir: Path = PROJECT_DIR,
    relative_to: Path | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> LoadedYoloModel:
    """Resolve, load, and validate one local Ultralytics detection model.

    ``model_factory`` is injectable for unit tests and must behave like
    ``ultralytics.YOLO``.  Production callers should leave it unset.
    """
    path = resolve_model_path(
        model_reference,
        project_dir=project_dir,
        relative_to=relative_to,
    )
    reference = str(model_reference)
    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YOLO inference requires the ultralytics package; "
                "install it in the Python environment used for capture"
            ) from exc
        model_factory = YOLO

    model = model_factory(str(path))
    info = _build_model_info(
        model,
        reference=reference,
        path=path,
        target_label=target_label,
    )
    return LoadedYoloModel(model=model, info=info)


def normalize_rgb_frame(rgb_frame: Any) -> np.ndarray:
    """Normalize one Isaac RGB/RGBA frame to contiguous uint8 RGB."""
    array = np.asarray(rgb_frame)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(
            f"RGB frame must have shape (height, width, 3/4), got {array.shape}"
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("RGB frame must have positive width and height")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"RGB frame must contain numeric pixels, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("RGB frame must contain only finite pixels")

    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.max(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_yolo_bgr(rgb_frame: np.ndarray) -> np.ndarray:
    """Convert normalized RGB pixels to the BGR ndarray Ultralytics expects."""
    return np.ascontiguousarray(rgb_frame[..., ::-1])


def _to_numpy(values: Any, *, name: str) -> np.ndarray:
    """Convert NumPy/Torch-like model output to a CPU NumPy array."""
    converted = values
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        return np.asarray(converted)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"YOLO result field {name!r} is not numeric") from exc


def _extract_prediction_boxes(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned ``xyxy``, confidence, and class-id arrays."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    xyxy = _to_numpy(getattr(boxes, "xyxy", None), name="xyxy")
    confidence = _to_numpy(getattr(boxes, "conf", None), name="conf")
    class_ids = _to_numpy(getattr(boxes, "cls", None), name="cls")
    if xyxy.size == 0:
        xyxy = np.empty((0, 4), dtype=np.float64)
    elif xyxy.ndim == 1 and xyxy.size == 4:
        xyxy = xyxy.reshape(1, 4)
    elif xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise RuntimeError(f"YOLO result xyxy field has unsupported shape {xyxy.shape}")

    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.float64).reshape(-1)
    if len(xyxy) != len(confidence) or len(xyxy) != len(class_ids):
        raise RuntimeError(
            "YOLO result fields xyxy, conf, and cls contain different numbers of boxes"
        )
    return xyxy.astype(np.float64), confidence, class_ids


def _first_prediction(predictions: Any) -> Any | None:
    """Return the first result from Ultralytics' result list."""
    if predictions is None:
        return None
    if isinstance(predictions, (list, tuple)):
        return None if not predictions else predictions[0]
    return predictions


def _validate_inference_options(
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
) -> tuple[float, float, int]:
    """Validate and normalize the public inference options."""
    confidence = float(confidence_threshold)
    iou = float(iou_threshold)
    size = int(image_size)
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    if not isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    if size <= 0:
        raise ValueError("image_size must be positive")
    return confidence, iou, size


def infer_yolo_frame(
    loaded_model: LoadedYoloModel,
    rgb_frame: Any,
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.70,
    image_size: int = 640,
    device: str | int | None = None,
) -> YoloInferenceResult:
    """Run target detection on one RGB/RGBA frame.

    The selected detection is the highest-confidence box for the validated
    target class that passes ``confidence_threshold``.  Coordinates are kept
    in pixels, clamped to the frame, and marked as clipped when clamping was
    required.  A missing target is returned as a structured invalid result.
    """
    if not isinstance(loaded_model, LoadedYoloModel):
        raise TypeError("loaded_model must be a LoadedYoloModel")
    confidence, iou, size = _validate_inference_options(
        confidence_threshold,
        iou_threshold,
        image_size,
    )
    model_predict = getattr(loaded_model.model, "predict", None)
    if not callable(model_predict):
        raise TypeError("loaded YOLO model does not provide a callable predict method")

    started = perf_counter()
    normalized_rgb = normalize_rgb_frame(rgb_frame)
    height, width = normalized_rgb.shape[:2]
    model_frame = _to_yolo_bgr(normalized_rgb)
    predictions = model_predict(
        source=model_frame,
        conf=confidence,
        iou=iou,
        imgsz=size,
        device=device,
        classes=[loaded_model.info.target_class_id],
        verbose=False,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0

    result = _first_prediction(predictions)
    if result is None:
        return YoloInferenceResult(
            detection=None,
            frame_resolution=(width, height),
            inference_time_ms=elapsed_ms,
            total_detection_count=0,
            target_candidate_count=0,
            qualified_candidate_count=0,
            reason="YOLO returned no prediction result",
        )

    xyxy, confidences, class_ids = _extract_prediction_boxes(result)
    total_count = len(xyxy)
    target_candidates = []
    qualified_candidates = []
    for index, (coordinates, box_confidence, box_class_id) in enumerate(
        zip(xyxy, confidences, class_ids)
    ):
        if not isfinite(float(box_class_id)) or int(box_class_id) != float(box_class_id):
            continue
        if int(box_class_id) != loaded_model.info.target_class_id:
            continue
        target_candidates.append(index)
        if isfinite(float(box_confidence)) and float(box_confidence) >= confidence:
            qualified_candidates.append(index)

    if not target_candidates:
        reason = f"no detections for target class {loaded_model.info.target_label!r}"
    elif not qualified_candidates:
        reason = f"target detections are below confidence threshold {confidence:.6g}"
    else:
        selected_index = max(
            qualified_candidates,
            key=lambda index: float(confidences[index]),
        )
        raw_coordinates = tuple(float(value) for value in xyxy[selected_index])
        if not all(isfinite(value) for value in raw_coordinates):
            raise RuntimeError("YOLO target detection contains nonfinite coordinates")
        x_min, y_min, x_max, y_max = raw_coordinates
        clamped = (
            max(0.0, min(float(width), x_min)),
            max(0.0, min(float(height), y_min)),
            max(0.0, min(float(width), x_max)),
            max(0.0, min(float(height), y_max)),
        )
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            reason = "target detection has no visible area after image-boundary clamp"
        else:
            clipped = clamped != raw_coordinates
            bbox = BoundingBox2D(
                x_min=clamped[0],
                y_min=clamped[1],
                x_max=clamped[2],
                y_max=clamped[3],
                resolution=(width, height),
                semantic_id=loaded_model.info.target_class_id,
                semantic_label=loaded_model.info.target_label,
                clipped=clipped,
                valid=True,
                reason=("detection bbox was clamped to the image boundary" if clipped else None),
            )
            detection = YoloDetection(
                bbox=bbox,
                confidence=float(confidences[selected_index]),
                class_id=loaded_model.info.target_class_id,
                class_name=loaded_model.info.target_label,
                raw_xyxy=raw_coordinates,
            )
            return YoloInferenceResult(
                detection=detection,
                frame_resolution=(width, height),
                inference_time_ms=elapsed_ms,
                total_detection_count=total_count,
                target_candidate_count=len(target_candidates),
                qualified_candidate_count=len(qualified_candidates),
            )

    return YoloInferenceResult(
        detection=None,
        frame_resolution=(width, height),
        inference_time_ms=elapsed_ms,
        total_detection_count=total_count,
        target_candidate_count=len(target_candidates),
        qualified_candidate_count=len(qualified_candidates),
        reason=reason,
    )


def infer_yolo_frames(
    loaded_model: LoadedYoloModel,
    rgb_frames: Sequence[Any],
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.70,
    image_size: int = 640,
    device: str | int | None = None,
) -> list[YoloInferenceResult]:
    """Run the same validated target detector over ordered camera frames."""
    if not isinstance(loaded_model, LoadedYoloModel):
        raise TypeError("loaded_model must be a LoadedYoloModel")
    return [
        infer_yolo_frame(
            loaded_model,
            rgb_frame,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            device=device,
        )
        for rgb_frame in rgb_frames
    ]


def build_yolo_observations(
    calibrations: Sequence[CameraCalibration],
    inference_results: Sequence[YoloInferenceResult],
    *,
    capture_id: str | int,
) -> list[CameraObservation]:
    """Convert ordered YOLO frame results into camera observations."""
    if len(calibrations) != len(inference_results):
        raise ValueError("calibrations and inference_results must have the same length")
    observations = []
    for calibration, inference in zip(calibrations, inference_results):
        if not isinstance(calibration, CameraCalibration):
            raise TypeError("calibrations must contain only CameraCalibration instances")
        if not isinstance(inference, YoloInferenceResult):
            raise TypeError("inference_results must contain only YoloInferenceResult instances")
        if inference.frame_resolution != calibration.resolution:
            raise ValueError(
                f"YOLO frame resolution {inference.frame_resolution} does not match "
                f"camera {calibration.camera_path} calibration {calibration.resolution}"
            )

        if inference.valid:
            observations.append(
                CameraObservation(
                    camera_path=calibration.camera_path,
                    calibration=calibration,
                    bbox=inference.detection.bbox,
                    capture_id=capture_id,
                    valid=True,
                )
            )
        else:
            observations.append(
                CameraObservation(
                    camera_path=calibration.camera_path,
                    calibration=calibration,
                    bbox=None,
                    capture_id=capture_id,
                    valid=False,
                    reason=inference.reason or "YOLO did not produce a target detection",
                )
            )
    return observations


def fuse_observations(
    observations: Sequence[CameraObservation],
    *,
    target_world: Any | None = None,
) -> ObservationFusion:
    """Build and fuse all available rays from one observation source."""
    if not observations:
        raise ValueError("observations must contain at least one camera observation")
    if any(not isinstance(observation, CameraObservation) for observation in observations):
        raise TypeError("observations must contain only CameraObservation instances")

    mutable_observations = list(observations)
    rays = build_rays_from_available_observations(mutable_observations)
    ray_camera_paths = {ray.camera_path for ray in rays}
    valid_camera_count = sum(
        observation.camera_path in ray_camera_paths for observation in mutable_observations
    )
    fusion = fuse_rays(rays)
    evaluation = None if target_world is None else evaluate_fusion(fusion, target_world)
    return ObservationFusion(
        observations=tuple(mutable_observations),
        rays=tuple(rays),
        fusion=fusion,
        evaluation=evaluation,
        valid_camera_count=valid_camera_count,
    )


def bbox_iou(
    first: BoundingBox2D | None,
    second: BoundingBox2D | None,
) -> float | None:
    """Return pixel-space IoU for two positive-area boxes, if available."""
    if first is None or second is None or first.area <= 0.0 or second.area <= 0.0:
        return None
    intersection_x_min = max(first.x_min, second.x_min)
    intersection_y_min = max(first.y_min, second.y_min)
    intersection_x_max = min(first.x_max, second.x_max)
    intersection_y_max = min(first.y_max, second.y_max)
    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area
    if union_area <= 0.0:
        return None
    return float(intersection_area / union_area)


def _observation_reason(observation: CameraObservation) -> str | None:
    """Return the most specific available observation failure/status reason."""
    if observation.reason is not None:
        return observation.reason
    return None if observation.bbox is None else observation.bbox.reason


def compare_observation_fusions(
    ground_truth: ObservationFusion,
    yolo: ObservationFusion,
    *,
    inference_results: Sequence[YoloInferenceResult] | None = None,
) -> FusionComparison:
    """Compare two synchronized observation sources and their fused geometry."""
    if not isinstance(ground_truth, ObservationFusion):
        raise TypeError("ground_truth must be an ObservationFusion")
    if not isinstance(yolo, ObservationFusion):
        raise TypeError("yolo must be an ObservationFusion")
    if inference_results is not None and len(inference_results) != len(yolo.observations):
        raise ValueError("inference_results must contain one result per YOLO observation")
    if len(ground_truth.observations) != len(yolo.observations):
        raise ValueError("ground_truth and yolo must contain the same camera observations")

    ground_truth_by_camera = {observation.camera_path: observation for observation in ground_truth.observations}
    yolo_by_camera = {observation.camera_path: observation for observation in yolo.observations}
    if set(ground_truth_by_camera) != set(yolo_by_camera):
        raise ValueError("ground_truth and yolo camera paths must match")
    yolo_result_by_camera = {
        observation.camera_path: result
        for observation, result in zip(yolo.observations, inference_results or [])
    }
    ground_truth_rays = {ray.camera_path: ray for ray in ground_truth.rays}
    yolo_rays = {ray.camera_path: ray for ray in yolo.rays}

    cameras = []
    for ground_truth_observation in ground_truth.observations:
        camera_path = ground_truth_observation.camera_path
        yolo_observation = yolo_by_camera[camera_path]
        yolo_result = yolo_result_by_camera.get(camera_path)
        ground_truth_bbox = ground_truth_observation.bbox
        yolo_bbox = yolo_observation.bbox
        bbox_overlap = bbox_iou(ground_truth_bbox, yolo_bbox)
        center_error_px = None
        center_error_normalized = None
        if ground_truth_bbox is not None and yolo_bbox is not None:
            center_error_px = float(np.linalg.norm(ground_truth_bbox.center_uv - yolo_bbox.center_uv))
            width, height = ground_truth_bbox.resolution
            image_diagonal = float(np.hypot(width, height))
            center_error_normalized = (
                None if image_diagonal <= 0.0 else center_error_px / image_diagonal
            )

        ray_angle_deg = None
        if camera_path in ground_truth_rays and camera_path in yolo_rays:
            ground_truth_direction = normalize(
                ground_truth_rays[camera_path].direction_world,
                name=f"{camera_path} ground-truth direction",
            )
            yolo_direction = normalize(
                yolo_rays[camera_path].direction_world,
                name=f"{camera_path} YOLO direction",
            )
            cosine = float(np.clip(np.dot(ground_truth_direction, yolo_direction), -1.0, 1.0))
            ray_angle_deg = float(np.degrees(np.arccos(cosine)))

        cameras.append(
            CameraComparison(
                camera_path=camera_path,
                ground_truth_valid=camera_path in ground_truth_rays,
                yolo_valid=camera_path in yolo_rays,
                ground_truth_reason=_observation_reason(ground_truth_observation),
                yolo_reason=(
                    (None if yolo_result is None or yolo_result.valid else yolo_result.reason)
                    or _observation_reason(yolo_observation)
                ),
                yolo_confidence=(
                    None
                    if yolo_result is None or yolo_result.detection is None
                    else yolo_result.detection.confidence
                ),
                bbox_iou=bbox_overlap,
                center_error_px=center_error_px,
                center_error_normalized=center_error_normalized,
                ray_angle_deg=ray_angle_deg,
            )
        )

    fused_position_delta_m = None
    if (
        ground_truth.fusion.fused_position_world is not None
        and yolo.fusion.fused_position_world is not None
    ):
        fused_position_delta_m = float(
            np.linalg.norm(
                ground_truth.fusion.fused_position_world - yolo.fusion.fused_position_world
            )
        )

    return FusionComparison(
        ground_truth=ground_truth,
        yolo=yolo,
        cameras=tuple(cameras),
        fused_position_delta_m=fused_position_delta_m,
    )
