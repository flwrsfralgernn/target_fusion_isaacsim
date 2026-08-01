"""Export schema-v2 synthetic captures as a YOLO detection dataset.

The capture script writes two kinds of images:

* ``image_path`` points to an annotated diagnostic preview.
* ``training_image_path`` points to the unannotated noise-processed frame.
* ``raw_image_path`` points to the clean Isaac BasicWriter RGB frame.

This exporter prefers the noise-processed training image and falls back to the
clean image for captures made before that field existed. It never reads the
annotated preview or trusts fusion validity as a detection-quality gate. A
camera view with no visible target still gets an image and an empty label file,
which makes it a valid YOLO negative example.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SCHEMA_OUTPUT = PROJECT_DIR / "outputs" / "target_fusion_bbox_v2.jsonl"
DEFAULT_DATASET_OUTPUT = PROJECT_DIR / "outputs" / "yolo_mannequin"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_TARGET_LABEL = "mannequin"
DEFAULT_MIN_BOX_AREA_PX = 1.0
YOLO_DECIMAL_PLACES = 6
YOLO_QUANTIZATION_SCALE = 10**YOLO_DECIMAL_PLACES


@dataclass(frozen=True)
class YoloBox:
    """One normalized YOLO row."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if int(self.class_id) < 0:
            raise ValueError("class_id must be nonnegative")
        values = (self.x_center, self.y_center, self.width, self.height)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("YOLO coordinates must be finite")
        if not all(0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("YOLO coordinates must be in [0, 1]")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("YOLO width and height must be positive")

    def as_line(self) -> str:
        """Return a six-decimal row whose reconstructed box stays in bounds.

        Rounding center and width independently can move a box that touches an
        image edge by a fraction of a micro-unit outside ``[0, 1]``. Quantize
        all values together and constrain the serialized center to the interval
        allowed by the serialized width/height.
        """
        scale = YOLO_QUANTIZATION_SCALE

        def quantize_size(value: float) -> int:
            return max(1, min(scale, int(round(float(value) * scale))))

        def quantize_center(value: float, size_units: int) -> int:
            half_units = (size_units + 1) // 2
            lower = half_units
            upper = scale - half_units
            return max(lower, min(upper, int(round(float(value) * scale))))

        width_units = quantize_size(self.width)
        height_units = quantize_size(self.height)
        x_center_units = quantize_center(self.x_center, width_units)
        y_center_units = quantize_center(self.y_center, height_units)

        def render(units: int) -> str:
            return f"{units / scale:.{YOLO_DECIMAL_PLACES}f}"

        return (
            f"{int(self.class_id)} "
            f"{render(x_center_units)} {render(y_center_units)} "
            f"{render(width_units)} {render(height_units)}"
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "class_id": int(self.class_id),
            "x_center": float(self.x_center),
            "y_center": float(self.y_center),
            "width": float(self.width),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class BBoxExport:
    """Result of converting one schema bbox into a YOLO row or negative."""

    status: str
    source_valid: bool | None
    source_clipped: bool | None
    source_reason: str | None
    original_xyxy_px: tuple[float, float, float, float] | None
    clamped_xyxy_px: tuple[float, float, float, float] | None
    visible_area_px: float
    yolo_box: YoloBox | None
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "source_valid": self.source_valid,
            "source_clipped": self.source_clipped,
            "source_reason": self.source_reason,
            "original_xyxy_px": (
                None
                if self.original_xyxy_px is None
                else list(self.original_xyxy_px)
            ),
            "clamped_xyxy_px": (
                None
                if self.clamped_xyxy_px is None
                else list(self.clamped_xyxy_px)
            ),
            "visible_area_px": float(self.visible_area_px),
            "normalized_xywh": (
                None if self.yolo_box is None else self.yolo_box.as_dict()
            ),
            "yolo_line": None if self.yolo_box is None else self.yolo_box.as_line(),
            "reason": self.reason,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export schema-v2 Isaac captures as a YOLO detection dataset."
    )
    parser.add_argument(
        "--schema-v2-output",
        type=Path,
        default=DEFAULT_SCHEMA_OUTPUT,
        help=f"Schema-v2 JSONL input (default: {DEFAULT_SCHEMA_OUTPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT,
        help=f"YOLO dataset directory (default: {DEFAULT_DATASET_OUTPUT})",
    )
    parser.add_argument(
        "--target-label",
        default=None,
        help=(
            "Target class to export; inferred from the schema when omitted "
            f"(default inference, expected {DEFAULT_TARGET_LABEL!r})"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-prob", type=float, default=0.70)
    parser.add_argument("--val-prob", type=float, default=0.20)
    parser.add_argument("--test-prob", type=float, default=0.10)
    parser.add_argument(
        "--min-box-area-px",
        type=float,
        default=DEFAULT_MIN_BOX_AREA_PX,
        help="Minimum post-clamp bbox area in pixels (default: 1.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new source captures while preserving existing group splits",
    )
    return parser.parse_args()


def normalize_split_probabilities(
    train_prob: float,
    val_prob: float,
    test_prob: float,
) -> dict[str, float]:
    probabilities = {
        "train": float(train_prob),
        "val": float(val_prob),
        "test": float(test_prob),
    }
    if not all(math.isfinite(value) for value in probabilities.values()):
        raise ValueError(f"Split probabilities must be finite: {probabilities}")
    if any(value < 0.0 for value in probabilities.values()):
        raise ValueError(f"Split probabilities must be nonnegative: {probabilities}")
    total = sum(probabilities.values())
    if total <= 0.0:
        raise ValueError("At least one split probability must be greater than zero")
    return {name: value / total for name, value in probabilities.items()}


def choose_split(
    capture_group_id: str,
    *,
    seed: int,
    probabilities: dict[str, float],
) -> str:
    """Choose a deterministic split from the capture group, not the camera view."""
    digest = hashlib.sha256(f"{int(seed)}\0{capture_group_id}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    threshold = 0.0
    for split_name in SPLIT_NAMES:
        threshold += probabilities[split_name]
        if sample < threshold:
            return split_name
    return SPLIT_NAMES[-1]


def _empty_bbox_export(
    *,
    bbox: dict | None,
    reason: str,
    source_valid: bool | None = None,
    source_clipped: bool | None = None,
    source_reason: str | None = None,
    original_xyxy_px: tuple[float, float, float, float] | None = None,
    clamped_xyxy_px: tuple[float, float, float, float] | None = None,
    visible_area_px: float = 0.0,
) -> BBoxExport:
    if bbox is not None:
        source_valid = bbox.get("valid") if source_valid is None else source_valid
        source_clipped = bbox.get("clipped") if source_clipped is None else source_clipped
        source_reason = bbox.get("reason") if source_reason is None else source_reason
    return BBoxExport(
        status="empty",
        source_valid=None if source_valid is None else bool(source_valid),
        source_clipped=None if source_clipped is None else bool(source_clipped),
        source_reason=None if source_reason is None else str(source_reason),
        original_xyxy_px=original_xyxy_px,
        clamped_xyxy_px=clamped_xyxy_px,
        visible_area_px=float(visible_area_px),
        yolo_box=None,
        reason=str(reason),
    )


def convert_bbox_to_yolo(
    bbox: dict | None,
    *,
    image_width: int,
    image_height: int,
    class_id: int = 0,
    min_area_px: float = DEFAULT_MIN_BOX_AREA_PX,
) -> BBoxExport:
    """Clamp one schema bbox and convert it to normalized YOLO coordinates.

    A source bbox can be marked invalid by the fusion pipeline because it is
    clipped or occluded.  If it still has positive visible area after clamp,
    it is retained for YOLO and the source validity/reason is preserved in the
    manifest.  Missing, malformed, zero-area, fully-outside, and too-small
    boxes become empty labels.
    """
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    min_area_px = float(min_area_px)
    if not math.isfinite(min_area_px) or min_area_px < 0.0:
        raise ValueError("min_area_px must be finite and nonnegative")
    if bbox is None:
        return _empty_bbox_export(bbox=None, reason="no target bbox in schema observation")
    if not isinstance(bbox, dict):
        return _empty_bbox_export(
            bbox=None,
            reason="schema target bbox is not an object",
        )

    try:
        original = tuple(
            float(bbox[field]) for field in ("x_min", "y_min", "x_max", "y_max")
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _empty_bbox_export(
            bbox=bbox,
            reason=f"schema target bbox is malformed: {exc}",
        )
    if not all(math.isfinite(value) for value in original):
        return _empty_bbox_export(
            bbox=bbox,
            reason="schema target bbox contains nonfinite coordinates",
            original_xyxy_px=original,
        )
    x_min, y_min, x_max, y_max = original
    if x_max <= x_min or y_max <= y_min:
        return _empty_bbox_export(
            bbox=bbox,
            reason="schema target bbox has zero or negative area",
            original_xyxy_px=original,
        )

    clamped = (
        max(0.0, min(float(image_width), x_min)),
        max(0.0, min(float(image_height), y_min)),
        max(0.0, min(float(image_width), x_max)),
        max(0.0, min(float(image_height), y_max)),
    )
    cx_min, cy_min, cx_max, cy_max = clamped
    visible_width = cx_max - cx_min
    visible_height = cy_max - cy_min
    visible_area = visible_width * visible_height
    source_clipped = bool(bbox.get("clipped", False))
    was_clipped = source_clipped or any(
        abs(original_value - clamped_value) > 0.0
        for original_value, clamped_value in zip(original, clamped)
    )
    if visible_width <= 0.0 or visible_height <= 0.0:
        return _empty_bbox_export(
            bbox=bbox,
            reason="target bbox has no visible area after image-boundary clamp",
            original_xyxy_px=original,
            clamped_xyxy_px=clamped,
            visible_area_px=visible_area,
        )
    if visible_area < min_area_px:
        return _empty_bbox_export(
            bbox=bbox,
            reason=f"target bbox area {visible_area:.6g}px is below minimum {min_area_px:.6g}px",
            original_xyxy_px=original,
            clamped_xyxy_px=clamped,
            visible_area_px=visible_area,
        )

    normalized = YoloBox(
        class_id=int(class_id),
        x_center=max(0.0, min(1.0, ((cx_min + cx_max) * 0.5) / image_width)),
        y_center=max(0.0, min(1.0, ((cy_min + cy_max) * 0.5) / image_height)),
        width=max(0.0, min(1.0, visible_width / image_width)),
        height=max(0.0, min(1.0, visible_height / image_height)),
    )
    return BBoxExport(
        status="clipped" if was_clipped else "visible",
        source_valid=None if bbox.get("valid") is None else bool(bbox.get("valid")),
        source_clipped=None if bbox.get("clipped") is None else bool(bbox.get("clipped")),
        source_reason=None if bbox.get("reason") is None else str(bbox.get("reason")),
        original_xyxy_px=original,
        clamped_xyxy_px=clamped,
        visible_area_px=visible_area,
        yolo_box=normalized,
    )


def _resolve_source_path(value, source_jsonl: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    relative_to_source = (source_jsonl.parent / path).resolve()
    if relative_to_source.exists():
        return relative_to_source
    return (Path.cwd() / path).resolve()


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("YOLO export requires Pillow to inspect RGB image dimensions") from exc
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"could not read RGB image {image_path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"RGB image has invalid dimensions {width}x{height}: {image_path}")
    return int(width), int(height)


def _load_records(schema_path: Path) -> list[dict]:
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema-v2 JSONL does not exist: {schema_path}")
    records = []
    with schema_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on schema line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"schema line {line_number} is not a JSON object")
            observations = record.get("camera_observations")
            if not isinstance(observations, list):
                raise ValueError(
                    f"schema line {line_number} has no camera_observations list"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Schema-v2 JSONL contains no capture records: {schema_path}")
    return records


def _capture_group_id(record: dict, record_index: int) -> str:
    capture = record.get("capture")
    capture = capture if isinstance(capture, dict) else {}
    for key in ("capture_id", "scene_index"):
        if key in capture and capture[key] is not None:
            return str(capture[key])
    for key in ("capture_id", "scene_index"):
        if key in record and record[key] is not None:
            return str(record[key])
    return f"record-{record_index}"


def _target_label_for_record(record: dict) -> str | None:
    capture = record.get("capture")
    if isinstance(capture, dict) and capture.get("target_label"):
        return str(capture["target_label"])
    for observation in record.get("camera_observations", []):
        bbox = observation.get("bbox") if isinstance(observation, dict) else None
        if isinstance(bbox, dict) and bbox.get("semantic_label"):
            return str(bbox["semantic_label"])
    return None


def _prepare_output_dir(output_dir: Path, *, overwrite: bool, append: bool) -> None:
    if overwrite and append:
        raise ValueError("--overwrite and --append cannot be used together")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"YOLO output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite and not append:
        raise FileExistsError(
            f"YOLO output directory is not empty: {output_dir}; use --overwrite or --append"
        )
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    for split_name in SPLIT_NAMES:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)


def _read_existing_class_names(output_dir: Path) -> list[str] | None:
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.is_file():
        return None
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "names":
            parsed = ast.literal_eval(value.strip())
            if isinstance(parsed, list) and all(isinstance(name, str) for name in parsed):
                return parsed
            raise ValueError(f"unsupported names entry in {data_yaml}: {value}")
    return None


def _read_existing_manifest(output_dir: Path) -> tuple[dict[str, str], set[str], int]:
    manifest_path = output_dir / "manifest.jsonl"
    group_splits: dict[str, str] = {}
    source_keys: set[str] = set()
    next_index = 0
    if not manifest_path.is_file():
        return group_splits, source_keys, next_index
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on existing manifest line {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"existing manifest line {line_number} is not an object")
            group_id = item.get("capture_group_id")
            split_name = item.get("split")
            if group_id is not None and split_name is not None:
                group_id = str(group_id)
                split_name = str(split_name)
                if split_name not in SPLIT_NAMES:
                    raise ValueError(f"existing manifest has unsupported split {split_name!r}")
                previous = group_splits.setdefault(group_id, split_name)
                if previous != split_name:
                    raise ValueError(f"capture group {group_id!r} has conflicting existing splits")
            source_key = item.get("source_key")
            if source_key:
                source_keys.add(str(source_key))
            try:
                next_index = max(next_index, int(item.get("dataset_index", -1)) + 1)
            except (TypeError, ValueError):
                pass
    return group_splits, source_keys, next_index


def _write_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    names = json.dumps(class_names, ensure_ascii=False)
    data_yaml = (
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names: {names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def _relative_output_path(output_dir: Path, path: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def export_dataset(
    schema_path: Path,
    output_dir: Path,
    *,
    target_label: str | None = None,
    split_seed: int = 42,
    train_prob: float = 0.70,
    val_prob: float = 0.20,
    test_prob: float = 0.10,
    min_area_px: float = DEFAULT_MIN_BOX_AREA_PX,
    overwrite: bool = False,
    append: bool = False,
) -> dict:
    """Export all camera observations and return a serializable summary."""
    schema_path = schema_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    records = _load_records(schema_path)
    probabilities = normalize_split_probabilities(train_prob, val_prob, test_prob)
    if target_label is not None:
        target_label = str(target_label)
        if not target_label:
            raise ValueError("target_label must not be empty")

    existing_classes: list[str] | None = None
    existing_group_splits: dict[str, str] = {}
    existing_source_keys: set[str] = set()
    next_index = 0
    if append and output_dir.exists():
        existing_classes = _read_existing_class_names(output_dir)
        existing_group_splits, existing_source_keys, next_index = _read_existing_manifest(output_dir)
        if not (output_dir / "manifest.jsonl").is_file() and any(output_dir.iterdir()):
            raise ValueError("--append requires an existing manifest.jsonl for a nonempty dataset")

    _prepare_output_dir(output_dir, overwrite=overwrite, append=append)
    if existing_classes is not None:
        if len(existing_classes) != 1:
            raise ValueError(
                f"this exporter supports exactly one class; existing classes are {existing_classes}"
            )
        if target_label is None:
            target_label = existing_classes[0]
        if target_label is not None and existing_classes != [target_label]:
            raise ValueError(
                f"existing classes {existing_classes} do not match target label {target_label!r}"
            )
        class_names = existing_classes
    else:
        inferred_labels = {
            label
            for record in records
            if (label := _target_label_for_record(record)) is not None
        }
        if target_label is None:
            if len(inferred_labels) == 1:
                target_label = next(iter(inferred_labels))
            elif not inferred_labels:
                target_label = DEFAULT_TARGET_LABEL
            else:
                raise ValueError(f"schema contains multiple target labels: {sorted(inferred_labels)}")
        class_names = [target_label]

    manifest_path = output_dir / "manifest.jsonl"
    manifest_mode = "a" if append and manifest_path.exists() else "w"
    exported_count = 0
    skipped_or_empty_count = 0
    split_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    group_assignments = dict(existing_group_splits)
    seen_new_source_keys: set[str] = set()

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_stream:
        for record_index, record in enumerate(records):
            record_label = _target_label_for_record(record)
            if record_label is not None and record_label != target_label:
                raise ValueError(
                    f"capture record {record_index} target label {record_label!r} "
                    f"does not match exporter label {target_label!r}"
                )
            capture = record.get("capture") if isinstance(record.get("capture"), dict) else {}
            group_id = _capture_group_id(record, record_index)
            split_name = group_assignments.get(group_id)
            if split_name is None:
                split_name = choose_split(
                    group_id,
                    seed=split_seed,
                    probabilities=probabilities,
                )
                group_assignments[group_id] = split_name

            observations = record["camera_observations"]
            for camera_index, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    raise ValueError(
                        f"capture record {record_index} camera {camera_index} is not an object"
                    )
                training_image_value = observation.get("training_image_path")
                raw_image_value = observation.get("raw_image_path")
                source_image_value = training_image_value or raw_image_value
                source_image_field = (
                    "training_image_path" if training_image_value else "raw_image_path"
                )
                if not source_image_value:
                    raise ValueError(
                        f"capture record {record_index} camera {camera_index} has neither "
                        "training_image_path nor raw_image_path; regenerate the capture"
                    )
                source_image_path = _resolve_source_path(source_image_value, schema_path)
                if not source_image_path.is_file():
                    raise FileNotFoundError(
                        f"{source_image_field} for capture {record_index} camera "
                        f"{camera_index} does not exist: "
                        f"{source_image_path}"
                    )
                image_width, image_height = _read_image_size(source_image_path)
                bbox_export = convert_bbox_to_yolo(
                    observation.get("bbox"),
                    image_width=image_width,
                    image_height=image_height,
                    class_id=0,
                    min_area_px=min_area_px,
                )

                camera_path = str(observation.get("camera_path", f"camera_{camera_index + 1:02d}"))
                source_key = "\x1f".join(
                    (group_id, camera_path, str(source_image_path))
                )
                if source_key in existing_source_keys or source_key in seen_new_source_keys:
                    raise ValueError(
                        f"source capture/camera is already present in the YOLO dataset: {source_key!r}"
                    )
                seen_new_source_keys.add(source_key)

                dataset_index = next_index
                next_index += 1
                source_suffix = source_image_path.suffix.lower() or ".png"
                sample_stem = f"capture_{dataset_index:08d}_camera_{camera_index + 1:02d}"
                destination_image = output_dir / split_name / "images" / f"{sample_stem}{source_suffix}"
                destination_label = output_dir / split_name / "labels" / f"{sample_stem}.txt"
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                if source_image_path.resolve() != destination_image.resolve():
                    shutil.copy2(source_image_path, destination_image)
                label_text = "" if bbox_export.yolo_box is None else bbox_export.yolo_box.as_line() + "\n"
                destination_label.write_text(label_text, encoding="utf-8")

                manifest_item = {
                    "manifest_version": 1,
                    "dataset_index": dataset_index,
                    "capture_group_id": group_id,
                    "split": split_name,
                    "scene_index": capture.get("scene_index", record_index),
                    "capture_id": capture.get("capture_id", group_id),
                    "camera_index": camera_index,
                    "camera_path": camera_path,
                    "target_label": target_label,
                    "class_id": 0,
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_path": _relative_output_path(output_dir, destination_image),
                    "label_path": _relative_output_path(output_dir, destination_label),
                    "source_key": source_key,
                    "source_image_field": source_image_field,
                    "source_image_path": str(source_image_path),
                    "source_bbox_path": observation.get("raw_bbox_path"),
                    "source_camera_params_path": observation.get("raw_camera_params_path"),
                    "annotation": bbox_export.as_dict(),
                }
                manifest_stream.write(
                    json.dumps(manifest_item, separators=(",", ":"), allow_nan=False) + "\n"
                )
                split_counts[split_name] += 1
                exported_count += 1
                if bbox_export.yolo_box is None:
                    skipped_or_empty_count += 1
                manifest_stream.flush()

    _write_data_yaml(output_dir, class_names)
    if exported_count == 0:
        raise ValueError("schema contained no camera observations to export")
    return {
        "output_dir": str(output_dir),
        "class_names": class_names,
        "capture_count": len(records),
        "sample_count": exported_count,
        "empty_label_count": skipped_or_empty_count,
        "split_counts": split_counts,
        "group_count": len(group_assignments),
    }


def main() -> None:
    args = parse_args()
    summary = export_dataset(
        args.schema_v2_output,
        args.output_dir,
        target_label=args.target_label,
        split_seed=args.split_seed,
        train_prob=args.train_prob,
        val_prob=args.val_prob,
        test_prob=args.test_prob,
        min_area_px=args.min_box_area_px,
        overwrite=args.overwrite,
        append=args.append,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
