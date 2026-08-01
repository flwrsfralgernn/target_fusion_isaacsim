"""Validate and train a YOLO11 bounding-box detector locally.

The trainer expects an already split YOLO detection dataset and never creates
a train/validation/test split. The schema-v2 export can be used directly:

    python3 scripts/train_yolo_local.py \
        --data outputs/yolo_mannequin/data.yaml

The default mode is GPU-first: an unavailable CUDA device is an error unless
the caller explicitly selects ``--device cpu`` or passes ``--allow-cpu``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from validate_yolo_dataset import (
        ValidationReport,
        load_data_yaml,
        split_image_paths,
        validate_data_yaml,
    )
except ModuleNotFoundError:
    from scripts.validate_yolo_dataset import (
        ValidationReport,
        load_data_yaml,
        split_image_paths,
        validate_data_yaml,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_YAML = PROJECT_DIR / "outputs" / "yolo_mannequin" / "data.yaml"
DEFAULT_TRAINING_PROJECT = PROJECT_DIR / "outputs" / "yolo_training_runs"
DEFAULT_ARCHIVE_DIR = PROJECT_DIR / "outputs" / "yolo_training_archives"


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


def audit_dataset(data_yaml_path: Path) -> ValidationReport:
    """Run the shared validator with the pre-training contract."""
    return validate_data_yaml(
        data_yaml_path,
        allow_missing_manifest=True,
        required_splits=("train", "val"),
        nonempty_splits=("train", "val"),
        require_labeled_objects=True,
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
            "Dataset audit failed:\n" + "\n".join(audit.errors)
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
        "dataset_root": audit.dataset_dir,
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
