"""Train a normal YOLO11 bounding-box detector locally.

This is the local equivalent of the attached Colab workflow.  It expects an
already split YOLO detection dataset and never creates a train/validation/test
split.  The generated SDG dataset can be used directly:

    python3 scripts/train_yolo_local.py \
        --data outputs/autovalidated_sdg_final/yolo/data.yaml

The default mode is GPU-first: an unavailable CUDA device is an error unless
the caller explicitly selects ``--device cpu`` or passes ``--allow-cpu``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_YAML = PROJECT_DIR / "outputs" / "autovalidated_sdg_final" / "yolo" / "data.yaml"
DEFAULT_TRAINING_PROJECT = PROJECT_DIR / "outputs" / "yolo_training_runs"
DEFAULT_ARCHIVE_DIR = PROJECT_DIR / "outputs" / "yolo_training_archives"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitAudit:
    """Dataset counts and errors for one YOLO split."""

    name: str
    image_dir: str
    label_dir: str
    images: int
    labels: int
    missing_labels: int
    extra_labels: int
    empty_labels: int
    objects: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetAudit:
    """Full pre-training dataset audit."""

    data_yaml: str
    dataset_root: str
    class_names: tuple[str, ...]
    splits: tuple[SplitAudit, ...]
    errors: tuple[str, ...] = ()

    @property
    def total_objects(self) -> int:
        return sum(split.objects for split in self.splits)

    @property
    def valid(self) -> bool:
        return not self.errors and all(not split.errors for split in self.splits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_yaml": self.data_yaml,
            "dataset_root": self.dataset_root,
            "class_names": list(self.class_names),
            "splits": [asdict(split) for split in self.splits],
            "total_objects": self.total_objects,
            "valid": self.valid,
            "errors": list(self.errors),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and train a normal YOLO11 bbox detector locally."
    )
    parser.add_argument(
        "--data",
        "--data-yaml",
        dest="data_yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help=f"YOLO data.yaml/data.yml path (default: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local checkpoint (default: yolo11n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size; use -1 for Ultralytics auto-batch (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Dataloader workers (default: min(8, CPU count))",
    )
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TRAINING_PROJECT,
        help=f"Local Ultralytics run directory (default: {DEFAULT_TRAINING_PROJECT})",
    )
    parser.add_argument(
        "--name",
        default="mannequin_yolo11n_bbox",
        help="Training run name (default: mannequin_yolo11n_bbox)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, 0, or a comma-separated CUDA device list (default: auto)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow auto mode to fall back to CPU when CUDA is unavailable",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="ram",
        help="Image cache mode (default: ram)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic training where supported",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision (default: enabled)",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write Ultralytics training plots (default: enabled)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a previous last.pt checkpoint",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Run an explicit Ultralytics evaluation on the test split after training",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Zip the completed run under --archive-dir",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Archive output directory (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse an existing run directory",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit the dataset and write the patched YAML without starting training",
    )
    return parser.parse_args()


def _require_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required. Install training dependencies with: "
            "python3 -m pip install --user ultralytics pyyaml"
        ) from exc
    return yaml


def load_data_yaml(data_yaml_path: Path) -> tuple[dict[str, Any], Path, tuple[str, ...]]:
    """Load a YOLO YAML and resolve its dataset root and class names."""
    data_yaml_path = Path(data_yaml_path).expanduser().resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"YOLO data YAML does not exist: {data_yaml_path}")

    yaml = _require_yaml()
    with data_yaml_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YOLO data YAML must contain a mapping: {data_yaml_path}")

    names = config.get("names")
    if isinstance(names, dict):
        try:
            class_names = tuple(str(names[key]) for key in sorted(names, key=lambda item: int(item)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported names mapping in {data_yaml_path}") from exc
    elif isinstance(names, list):
        class_names = tuple(str(name) for name in names)
    else:
        raise ValueError(f"YOLO data YAML requires a names list or mapping: {data_yaml_path}")
    if not class_names:
        raise ValueError(f"YOLO data YAML contains no classes: {data_yaml_path}")

    root_value = config.get("path", ".")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = data_yaml_path.parent / root
    return config, root.resolve(), class_names


def _config_paths(root: Path, value: Any) -> list[Path]:
    """Resolve one YOLO split value, including list-valued entries."""
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


def split_image_paths(config: dict[str, Any], root: Path) -> dict[str, list[Path]]:
    """Return canonical train/val/test image directories from a YOLO YAML."""
    if "train" not in config:
        raise ValueError("YOLO data YAML is missing the train split")
    val_key = "val" if "val" in config else "valid" if "valid" in config else None
    if val_key is None:
        raise ValueError("YOLO data YAML is missing the val/valid split")

    paths = {
        "train": _config_paths(root, config["train"]),
        "val": _config_paths(root, config[val_key]),
    }
    if "test" in config and config["test"] is not None:
        paths["test"] = _config_paths(root, config["test"])
    return paths


def label_dir_for_image_dir(image_dir: Path) -> Path:
    """Resolve the conventional YOLO labels sibling for an images folder."""
    image_dir = Path(image_dir).resolve()
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir / "labels"


def _collect_images(image_dirs: list[Path]) -> tuple[dict[str, Path], list[str]]:
    images: dict[str, Path] = {}
    errors: list[str] = []
    for image_dir in image_dirs:
        if not image_dir.is_dir():
            errors.append(f"missing image directory: {image_dir}")
            continue
        for path in sorted(image_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            previous = images.get(path.stem)
            if previous is not None:
                errors.append(f"duplicate image stem {path.stem!r}: {previous}, {path}")
            else:
                images[path.stem] = path.resolve()
    return images, errors


def _collect_labels(label_dirs: list[Path]) -> tuple[dict[str, Path], list[str]]:
    labels: dict[str, Path] = {}
    errors: list[str] = []
    for label_dir in label_dirs:
        if not label_dir.is_dir():
            errors.append(f"missing label directory: {label_dir}")
            continue
        for path in sorted(label_dir.rglob("*.txt"), key=lambda item: item.as_posix().lower()):
            previous = labels.get(path.stem)
            if previous is not None:
                errors.append(f"duplicate label stem {path.stem!r}: {previous}, {path}")
            else:
                labels[path.stem] = path.resolve()
    return labels, errors


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for dataset auditing. Install training dependencies with: "
            "python3 -m pip install --user ultralytics pyyaml pillow"
        ) from exc
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"could not read image {image_path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"image has invalid dimensions {width}x{height}: {image_path}")
    return int(width), int(height)


def _audit_label_file(
    label_path: Path,
    *,
    image_size: tuple[int, int],
    class_count: int,
) -> tuple[int, bool, list[str]]:
    """Return object count, empty status, and row-level errors."""
    errors: list[str] = []
    try:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return 0, False, [f"could not read label file {label_path}: {exc}"]
    if not lines:
        return 0, True, []

    image_width, image_height = image_size
    objects = 0
    for line_number, line in enumerate(lines, start=1):
        prefix = f"{label_path}:{line_number}"
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{prefix}: expected 5 YOLO bbox fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{prefix}: malformed numeric value: {exc}")
            continue
        values = (x_center, y_center, width, height)
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{prefix}: class id {class_id} is outside [0, {class_count})")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{prefix}: bbox values must be finite")
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"{prefix}: bbox values must be in [0, 1]")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{prefix}: bbox width and height must be positive")
            continue
        if x_center - width / 2.0 < 0.0 or x_center + width / 2.0 > 1.0:
            errors.append(f"{prefix}: bbox extends outside horizontal image bounds")
            continue
        if y_center - height / 2.0 < 0.0 or y_center + height / 2.0 > 1.0:
            errors.append(f"{prefix}: bbox extends outside vertical image bounds")
            continue
        objects += 1
    return objects, False, errors


def audit_dataset(data_yaml_path: Path) -> DatasetAudit:
    """Validate split structure and normal YOLO bbox labels before training."""
    config, root, class_names = load_data_yaml(data_yaml_path)
    split_paths = split_image_paths(config, root)
    split_audits: list[SplitAudit] = []
    dataset_errors: list[str] = []

    for split_name in ("train", "val", "test"):
        image_dirs = split_paths.get(split_name)
        if image_dirs is None:
            if split_name == "test":
                continue
            dataset_errors.append(f"missing required {split_name} split")
            continue
        label_dirs = [label_dir_for_image_dir(path) for path in image_dirs]
        images, image_errors = _collect_images(image_dirs)
        labels, label_errors = _collect_labels(label_dirs)
        errors = [*image_errors, *label_errors]
        missing_labels = sorted(set(images) - set(labels))
        extra_labels = sorted(set(labels) - set(images))
        errors.extend(f"missing label for image stem {stem!r}" for stem in missing_labels)
        errors.extend(f"label has no matching image stem {stem!r}" for stem in extra_labels)

        empty_labels = 0
        objects = 0
        for stem in sorted(set(images) & set(labels)):
            image_path = images[stem]
            label_path = labels[stem]
            try:
                image_size = _read_image_size(image_path)
                row_objects, is_empty, row_errors = _audit_label_file(
                    label_path,
                    image_size=image_size,
                    class_count=len(class_names),
                )
            except (RuntimeError, ValueError) as exc:
                row_objects, is_empty, row_errors = 0, False, [str(exc)]
            objects += row_objects
            empty_labels += int(is_empty)
            errors.extend(row_errors)

        split_audits.append(
            SplitAudit(
                name=split_name,
                image_dir=", ".join(str(path) for path in image_dirs),
                label_dir=", ".join(str(path) for path in label_dirs),
                images=len(images),
                labels=len(labels),
                missing_labels=len(missing_labels),
                extra_labels=len(extra_labels),
                empty_labels=empty_labels,
                objects=objects,
                errors=tuple(errors),
            )
        )

    required_by_name = {split.name: split for split in split_audits}
    for required_split in ("train", "val"):
        split = required_by_name.get(required_split)
        if split is None or split.images == 0:
            dataset_errors.append(f"required {required_split} split contains no images")
    if sum(split.objects for split in split_audits) == 0:
        dataset_errors.append("no labeled objects found in the dataset")

    return DatasetAudit(
        data_yaml=str(Path(data_yaml_path).expanduser().resolve()),
        dataset_root=str(root),
        class_names=class_names,
        splits=tuple(split_audits),
        errors=tuple(dataset_errors),
    )


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_patched_data_yaml(
    data_yaml_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write a local-path YAML without modifying the source YAML."""
    config, root, _ = load_data_yaml(data_yaml_path)
    split_paths = split_image_paths(config, root)
    patched = dict(config)
    patched["path"] = str(root)
    for split_name, paths in split_paths.items():
        if len(paths) == 1:
            patched[split_name] = _relative_or_absolute(paths[0], root)
        else:
            patched[split_name] = [_relative_or_absolute(path, root) for path in paths]
    patched.pop("valid", None)

    yaml = _require_yaml()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(patched, stream, sort_keys=False)
    return output_path, patched


def resolve_device(requested: str, *, allow_cpu: bool) -> str | int:
    """Resolve and validate the requested Torch/Ultralytics device."""
    requested = str(requested).strip().lower()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install a CUDA-enabled PyTorch build "
            "for this machine, then install ultralytics."
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        if cuda_available:
            return 0
        if allow_cpu:
            return "cpu"
        raise RuntimeError(
            "CUDA is unavailable to this Python environment. Training was stopped "
            "instead of silently using CPU. Check the NVIDIA driver/CUDA PyTorch "
            "installation, or explicitly pass --device cpu/--allow-cpu."
        )
    if requested == "cpu":
        return "cpu"
    if not cuda_available:
        raise RuntimeError(
            f"Requested CUDA device {requested!r}, but torch.cuda.is_available() is False."
        )

    device_ids = [part.strip() for part in requested.split(",") if part.strip()]
    if not device_ids or not all(part.isdigit() for part in device_ids):
        raise ValueError("--device must be auto, cpu, an integer GPU id, or comma-separated GPU ids")
    device_count = int(torch.cuda.device_count())
    invalid = [part for part in device_ids if int(part) >= device_count]
    if invalid:
        raise ValueError(
            f"Requested CUDA device(s) {invalid}, but only {device_count} device(s) are visible"
        )
    return ",".join(device_ids) if len(device_ids) > 1 else int(device_ids[0])


def _cache_value(cache_mode: str) -> bool | str:
    return False if cache_mode == "none" else cache_mode


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch must be positive or -1 for auto-batch")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    if args.patience < 0:
        raise ValueError("--patience must be nonnegative")
    if not args.name or Path(args.name).name != args.name:
        raise ValueError("--name must be a non-empty filename-safe run name")
    if args.resume is not None and not args.resume.expanduser().is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Audit, train, optionally evaluate, and optionally archive one run."""
    _validate_args(args)
    audit = audit_dataset(args.data_yaml)
    if not audit.valid:
        raise RuntimeError(
            "Dataset audit failed:\n" + "\n".join(
                [*audit.errors]
                + [error for split in audit.splits for error in split.errors]
            )
        )

    project_dir = args.project.expanduser().resolve()
    patched_yaml_path, patched_config = write_patched_data_yaml(
        args.data_yaml,
        project_dir / f"{args.name}_data.yaml",
    )

    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires ultralytics and PyTorch. Install with: "
            "python3 -m pip install --user ultralytics pyyaml"
        ) from exc

    if isinstance(device, int) and torch.cuda.is_available():
        print(f"[INFO] CUDA device {device}: {torch.cuda.get_device_name(device)}")
    else:
        print(f"[INFO] Training device: {device}")

    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        model = YOLO(str(checkpoint))
        resume = True
        print(f"[INFO] Resuming checkpoint: {checkpoint}")
    else:
        model = YOLO(args.model)
        resume = False
        print(f"[INFO] Starting model: {args.model}")

    project_dir.mkdir(parents=True, exist_ok=True)
    train_kwargs = {
        "data": str(patched_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": str(project_dir),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "cache": _cache_value(args.cache),
        "save": True,
        "plots": args.plots,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "exist_ok": args.exist_ok,
        "resume": resume,
    }
    print("[INFO] Dataset audit:")
    print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    print(f"[INFO] Patched training YAML: {patched_yaml_path}")
    print(f"[INFO] Run output directory: {project_dir / args.name}")

    model.train(**train_kwargs)
    run_dir = Path(model.trainer.save_dir).expanduser().resolve()
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training completed but best.pt was not found: {best_path}")
    if not last_path.is_file():
        raise FileNotFoundError(f"Training completed but last.pt was not found: {last_path}")

    test_metrics = None
    if args.eval_test:
        _, root, _ = load_data_yaml(args.data_yaml)
        split_paths = split_image_paths(load_data_yaml(args.data_yaml)[0], root)
        if "test" not in split_paths:
            raise RuntimeError("--eval-test was requested, but the data YAML has no test split")
        print("[INFO] Running explicit test-split evaluation...")
        metrics = model.val(
            data=str(patched_yaml_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch if args.batch > 0 else 1,
            device=device,
            plots=args.plots,
            project=str(project_dir),
            name=f"{args.name}_test",
            exist_ok=True,
        )
        results_dict = getattr(metrics, "results_dict", {})
        test_metrics = {str(key): float(value) for key, value in results_dict.items()}

    archive_path = None
    if args.archive:
        archive_dir = args.archive_dir.expanduser().resolve()
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_base = archive_dir / run_dir.name
        archive_path = Path(
            shutil.make_archive(
                base_name=str(archive_base),
                format="zip",
                root_dir=run_dir.parent,
                base_dir=run_dir.name,
            )
        ).resolve()

    summary = {
        "data_yaml": str(Path(args.data_yaml).expanduser().resolve()),
        "patched_data_yaml": str(patched_yaml_path),
        "dataset_root": audit.dataset_root,
        "class_names": list(audit.class_names),
        "device": str(device),
        "model": str(args.resume if args.resume is not None else args.model),
        "run_dir": str(run_dir),
        "best_pt": str(best_path),
        "last_pt": str(last_path),
        "test_metrics": test_metrics,
        "archive_path": None if archive_path is None else str(archive_path),
        "train_args": {
            key: value
            for key, value in train_kwargs.items()
            if key not in {"data", "project"}
        },
        "audit": audit.as_dict(),
        "patched_config": patched_config,
    }
    summary_path = run_dir / "local_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    print("[DONE] YOLO training complete.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    args = parse_args()
    if args.check_only:
        _validate_args(args)
        audit = audit_dataset(args.data_yaml)
        print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
        if not audit.valid:
            return 1
        patched_path, _ = write_patched_data_yaml(
            args.data_yaml,
            args.project.expanduser().resolve() / f"{args.name}_data.yaml",
        )
        print(f"[INFO] Patched YAML: {patched_path}")
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
