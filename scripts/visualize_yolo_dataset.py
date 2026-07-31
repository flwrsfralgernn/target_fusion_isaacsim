"""Write visual previews for labels in an exported YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from validate_yolo_dataset import (
        SPLIT_NAMES,
        collect_split_images,
        collect_split_labels,
        load_dataset_config,
        read_yolo_label_rows,
        resolve_split_directory,
        resolve_split_label_directory,
    )
except ModuleNotFoundError:
    from scripts.validate_yolo_dataset import (
        SPLIT_NAMES,
        collect_split_images,
        collect_split_labels,
        load_dataset_config,
        read_yolo_label_rows,
        resolve_split_directory,
        resolve_split_label_directory,
    )


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_DIR / "outputs" / "yolo_mannequin"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO boxes on dataset images.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"YOLO dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Preview output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def _class_color(class_id: int) -> tuple[int, int, int]:
    colors = ((40, 220, 40), (255, 180, 40), (80, 160, 255), (255, 80, 180))
    return colors[class_id % len(colors)]


def draw_yolo_preview(
    image: Image.Image,
    label_path: Path,
    *,
    class_names: list[str],
) -> tuple[Image.Image, int]:
    """Return a copy with pixel-space boxes drawn and the row count."""
    image = image.convert("RGB")
    width, height = image.size
    rows, errors = read_yolo_label_rows(label_path, class_count=len(class_names))
    if errors:
        raise ValueError("; ".join(errors))
    draw = ImageDraw.Draw(image)
    for row in rows:
        x_min = int(round((row.x_center - row.width / 2.0) * width))
        y_min = int(round((row.y_center - row.height / 2.0) * height))
        x_max = int(round((row.x_center + row.width / 2.0) * width))
        y_max = int(round((row.y_center + row.height / 2.0) * height))
        color = _class_color(row.class_id)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)
        label = class_names[row.class_id]
        draw.text((max(0, x_min), max(0, y_min - 14)), label, fill=color)
    if not rows:
        draw.text((8, 8), "empty label", fill=(255, 80, 80))
    return image, len(rows)


def visualize_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    split: str = "train",
    start: int = 0,
    limit: int = 8,
) -> dict:
    """Write up to ``limit`` sorted previews and return a summary."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported split: {split}")
    if start < 0 or limit < 0:
        raise ValueError("start and limit must be nonnegative")
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = load_dataset_config(dataset_dir)
    split_root = resolve_split_directory(dataset_dir, config, split)
    labels_root = resolve_split_label_directory(dataset_dir, config, split)
    images = collect_split_images(split_root)
    labels = collect_split_labels(labels_root)
    stems = sorted(set(images) & set(labels))
    selected = stems[start : start + limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    box_count = 0
    for stem in selected:
        image_path = images[stem]
        label_path = labels[stem]
        with Image.open(image_path) as source:
            preview, row_count = draw_yolo_preview(
                source,
                label_path,
                class_names=list(config["names"]),
            )
        output_path = output_dir / f"{split}_{stem}_preview.png"
        preview.save(output_path, format="PNG")
        written.append(str(output_path))
        box_count += row_count

    return {
        "dataset_dir": str(dataset_dir),
        "split": split,
        "requested_start": start,
        "requested_limit": limit,
        "preview_count": len(written),
        "box_count": box_count,
        "output_paths": written,
    }


def main() -> int:
    args = parse_args()
    summary = visualize_dataset(
        args.dataset_dir,
        args.output_dir,
        split=args.split,
        start=args.start,
        limit=args.limit,
    )
    import json

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
