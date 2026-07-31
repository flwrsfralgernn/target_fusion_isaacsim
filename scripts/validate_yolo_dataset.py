"""Validate a YOLO dataset produced by ``export_yolo_dataset.py``.

Validation covers the file contract most likely to break synthetic-data
training runs: data.yaml paths/classes, image-label pairing, finite normalized
rows, empty-label negatives, manifest consistency, and capture-group split
leakage.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "outputs" / "yolo_mannequin"
SPLIT_NAMES = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloLabelRow:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ValidationReport:
    dataset_dir: str
    class_names: list[str] = field(default_factory=list)
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    manifest_sample_count: int = 0
    capture_group_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir,
            "valid": self.valid,
            "class_names": self.class_names,
            "split_counts": self.split_counts,
            "manifest_sample_count": self.manifest_sample_count,
            "capture_group_count": self.capture_group_count,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"YOLO dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Treat a missing manifest.jsonl as a warning instead of an error",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings, including empty splits, as validation failures",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )
    return parser.parse_args()


def _parse_yaml_value(value: str):
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("'\"")


def load_dataset_config(dataset_dir: Path) -> dict:
    """Parse the small data.yaml contract emitted by the exporter."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing YOLO data.yaml: {yaml_path}")
    config = {}
    for line_number, line in enumerate(yaml_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Malformed data.yaml line {line_number}: {line}")
        config[key.strip()] = _parse_yaml_value(value)

    names = config.get("names")
    if isinstance(names, dict):
        try:
            names = [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported names mapping in {yaml_path}") from exc
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"data.yaml names must be a nonempty list: {yaml_path}")
    config["names"] = names
    return config


def resolve_split_directory(dataset_dir: Path, config: dict, split_name: str) -> Path:
    """Resolve the image directory configured for one split from data.yaml."""
    if split_name not in config:
        raise ValueError(f"data.yaml is missing the {split_name!r} split path")
    root_value = config.get("path", ".")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = (dataset_dir / root).resolve()
    split_path = Path(str(config[split_name])).expanduser()
    if not split_path.is_absolute():
        split_path = root / split_path
    return split_path.resolve()


def resolve_split_label_directory(dataset_dir: Path, config: dict, split_name: str) -> Path:
    """Resolve the labels sibling for a configured split image directory."""
    images_dir = resolve_split_directory(dataset_dir, config, split_name)
    if images_dir.name == "images":
        return (images_dir.parent / "labels").resolve()
    return (images_dir / "labels").resolve()


def collect_split_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO images directory: {images_dir}")
    images = {}
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        previous = images.get(path.stem)
        if previous is not None:
            raise ValueError(f"multiple image files share stem {path.stem!r}: {previous}, {path}")
        images[path.stem] = path
    return images


def collect_split_labels(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO labels directory: {labels_dir}")
    labels = {}
    for path in sorted(labels_dir.glob("*.txt"), key=lambda item: item.name.lower()):
        previous = labels.get(path.stem)
        if previous is not None:
            raise ValueError(f"multiple label files share stem {path.stem!r}: {previous}, {path}")
        labels[path.stem] = path
    return labels


def read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("YOLO validation requires Pillow") from exc
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"could not read image {image_path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"image has invalid dimensions {width}x{height}: {image_path}")
    return int(width), int(height)


def read_yolo_label_rows(
    label_path: Path,
    *,
    class_count: int,
) -> tuple[list[YoloLabelRow], list[str]]:
    """Read rows and return row-level errors without stopping the dataset scan."""
    rows: list[YoloLabelRow] = []
    errors: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [], [f"could not read label file {label_path}: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        prefix = f"{label_path}:{line_number}"
        if len(parts) != 5:
            errors.append(f"{prefix}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{prefix}: malformed numeric value: {exc}")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{prefix}: class id {class_id} is outside [0, {class_count})")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{prefix}: coordinates must be finite")
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"{prefix}: coordinates must be in [0, 1]")
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{prefix}: width and height must be positive")
            continue
        rows.append(YoloLabelRow(class_id, *values))
    return rows, errors


def _relative_dataset_path(dataset_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_manifest_path(dataset_dir: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (dataset_dir / path).resolve()


def _validate_manifest(
    dataset_dir: Path,
    report: ValidationReport,
    split_pairs: dict[str, dict[str, tuple[Path, Path]]],
    *,
    allow_missing_manifest: bool,
) -> None:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        message = f"missing manifest.jsonl: {manifest_path}"
        if allow_missing_manifest:
            report.warnings.append(message)
        else:
            report.errors.append(message)
        return

    manifest_images: set[str] = set()
    manifest_labels: set[str] = set()
    groups_to_splits: dict[str, str] = {}
    source_keys: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"manifest line {line_number} is invalid JSON: {exc}")
                continue
            if not isinstance(item, dict):
                report.errors.append(f"manifest line {line_number} is not an object")
                continue
            report.manifest_sample_count += 1
            split_name = item.get("split")
            if split_name not in SPLIT_NAMES:
                report.errors.append(
                    f"manifest line {line_number} has unsupported split {split_name!r}"
                )
                continue
            group_id = item.get("capture_group_id")
            if group_id is None or str(group_id) == "":
                report.errors.append(f"manifest line {line_number} has no capture_group_id")
            else:
                group_id = str(group_id)
                previous_split = groups_to_splits.setdefault(group_id, split_name)
                if previous_split != split_name:
                    report.errors.append(
                        f"capture group {group_id!r} appears in both {previous_split!r} and {split_name!r}"
                    )
            source_key = item.get("source_key")
            if source_key:
                source_key = str(source_key)
                if source_key in source_keys:
                    report.errors.append(f"duplicate manifest source_key: {source_key!r}")
                source_keys.add(source_key)

            image_value = item.get("image_path")
            label_value = item.get("label_path")
            if not image_value or not label_value:
                report.errors.append(f"manifest line {line_number} is missing image_path or label_path")
                continue
            image_path = _resolve_manifest_path(dataset_dir, image_value)
            label_path = _resolve_manifest_path(dataset_dir, label_value)
            image_rel = _relative_dataset_path(dataset_dir, image_path)
            label_rel = _relative_dataset_path(dataset_dir, label_path)
            if image_rel in manifest_images:
                report.errors.append(f"duplicate manifest image_path: {image_rel}")
            if label_rel in manifest_labels:
                report.errors.append(f"duplicate manifest label_path: {label_rel}")
            manifest_images.add(image_rel)
            manifest_labels.add(label_rel)
            if not image_path.is_file():
                report.errors.append(f"manifest image does not exist: {image_path}")
            if not label_path.is_file():
                report.errors.append(f"manifest label does not exist: {label_path}")

            pair = split_pairs.get(split_name, {}).get(image_path.stem)
            if pair is None:
                report.errors.append(
                    f"manifest image is not a discovered {split_name} sample: {image_rel}"
                )
            else:
                discovered_image, discovered_label = pair
                if discovered_image.resolve() != image_path.resolve():
                    report.errors.append(f"manifest image path mismatch: {image_rel}")
                if discovered_label.resolve() != label_path.resolve():
                    report.errors.append(f"manifest label path mismatch: {label_rel}")

            annotation = item.get("annotation")
            if isinstance(annotation, dict) and label_path.is_file():
                label_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected_line = annotation.get("yolo_line")
                if expected_line is None and label_lines:
                    report.errors.append(
                        f"manifest annotation says empty but label has rows: {label_rel}"
                    )
                elif expected_line is not None and label_lines != [str(expected_line)]:
                    report.errors.append(
                        f"manifest annotation does not match label rows: {label_rel}"
                    )

    report.capture_group_count = len(groups_to_splits)
    discovered_images = {
        _relative_dataset_path(dataset_dir, image_path)
        for pairs in split_pairs.values()
        for image_path, _ in pairs.values()
    }
    discovered_labels = {
        _relative_dataset_path(dataset_dir, label_path)
        for pairs in split_pairs.values()
        for _, label_path in pairs.values()
    }
    if manifest_images != discovered_images:
        missing = sorted(discovered_images - manifest_images)
        extra = sorted(manifest_images - discovered_images)
        if missing:
            report.errors.append(f"images missing from manifest: {missing[:5]}")
        if extra:
            report.errors.append(f"manifest references unknown images: {extra[:5]}")
    if manifest_labels != discovered_labels:
        missing = sorted(discovered_labels - manifest_labels)
        extra = sorted(manifest_labels - discovered_labels)
        if missing:
            report.errors.append(f"labels missing from manifest: {missing[:5]}")
        if extra:
            report.errors.append(f"manifest references unknown labels: {extra[:5]}")


def validate_dataset(
    dataset_dir: Path,
    *,
    allow_missing_manifest: bool = False,
    strict: bool = False,
) -> ValidationReport:
    """Validate a YOLO dataset and return all discovered issues."""
    dataset_dir = dataset_dir.expanduser().resolve()
    report = ValidationReport(dataset_dir=str(dataset_dir))
    try:
        config = load_dataset_config(dataset_dir)
        report.class_names = list(config["names"])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        report.errors.append(str(exc))
        return report

    split_pairs: dict[str, dict[str, tuple[Path, Path]]] = {}
    for split_name in SPLIT_NAMES:
        try:
            split_root = resolve_split_directory(dataset_dir, config, split_name)
            labels_root = resolve_split_label_directory(dataset_dir, config, split_name)
            images = collect_split_images(split_root)
            labels = collect_split_labels(labels_root)
        except (FileNotFoundError, ValueError) as exc:
            report.errors.append(str(exc))
            report.split_counts[split_name] = {
                "images": 0,
                "labels": 0,
                "boxes": 0,
                "empty_labels": 0,
            }
            split_pairs[split_name] = {}
            continue

        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        if missing_labels:
            report.errors.append(f"{split_name}: images without labels: {missing_labels[:5]}")
        if orphan_labels:
            report.errors.append(f"{split_name}: labels without images: {orphan_labels[:5]}")

        pairs: dict[str, tuple[Path, Path]] = {}
        box_count = 0
        empty_label_count = 0
        for stem in sorted(set(images) & set(labels)):
            image_path = images[stem]
            label_path = labels[stem]
            pairs[stem] = (image_path, label_path)
            try:
                image_width, image_height = read_image_size(image_path)
                rows, row_errors = read_yolo_label_rows(
                    label_path,
                    class_count=len(report.class_names),
                )
            except (ValueError, RuntimeError) as exc:
                report.errors.append(str(exc))
                continue
            report.errors.extend(row_errors)
            box_count += len(rows)
            if not rows:
                empty_label_count += 1
            for row in rows:
                if row.x_center - row.width / 2.0 < 0.0 or row.x_center + row.width / 2.0 > 1.0:
                    report.errors.append(
                        f"{label_path}: bbox extends outside horizontal image bounds"
                    )
                if row.y_center - row.height / 2.0 < 0.0 or row.y_center + row.height / 2.0 > 1.0:
                    report.errors.append(
                        f"{label_path}: bbox extends outside vertical image bounds"
                    )
            if image_width <= 0 or image_height <= 0:
                report.errors.append(f"{image_path}: invalid image dimensions")

        report.split_counts[split_name] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": box_count,
            "empty_labels": empty_label_count,
        }
        if not images:
            report.warnings.append(f"{split_name}: split contains no images")
        split_pairs[split_name] = pairs

    _validate_manifest(
        dataset_dir,
        report,
        split_pairs,
        allow_missing_manifest=allow_missing_manifest,
    )
    if strict and report.warnings:
        report.errors.extend(f"warning treated as error: {warning}" for warning in report.warnings)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = validate_dataset(
            args.dataset_dir,
            allow_missing_manifest=args.allow_missing_manifest,
            strict=args.strict,
        )
    except Exception as exc:
        report = ValidationReport(dataset_dir=str(args.dataset_dir), errors=[str(exc)])
    payload = report.as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report_path is not None:
        report_path = args.report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
