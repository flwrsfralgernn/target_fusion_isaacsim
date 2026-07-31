import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.export_yolo_dataset import (
    choose_split,
    convert_bbox_to_yolo,
    export_dataset,
    normalize_split_probabilities,
)
from scripts.validate_yolo_dataset import validate_dataset
from scripts.visualize_yolo_dataset import visualize_dataset


class YoloConversionTests(unittest.TestCase):
    def test_clamped_bbox_is_retained_with_normalized_coordinates(self) -> None:
        result = convert_bbox_to_yolo(
            {
                "x_min": -10,
                "y_min": 10,
                "x_max": 20,
                "y_max": 30,
                "valid": False,
                "clipped": True,
                "reason": "target bbox is clipped by the image boundary",
            },
            image_width=100,
            image_height=80,
        )

        self.assertEqual(result.status, "clipped")
        self.assertIsNotNone(result.yolo_box)
        self.assertEqual(result.clamped_xyxy_px, (0.0, 10.0, 20.0, 30.0))
        self.assertEqual(result.yolo_box.as_line(), "0 0.100000 0.250000 0.200000 0.250000")
        self.assertFalse(result.source_valid)
        self.assertTrue(result.source_clipped)

    def test_missing_or_fully_outside_bbox_becomes_empty(self) -> None:
        missing = convert_bbox_to_yolo(None, image_width=100, image_height=80)
        self.assertEqual(missing.status, "empty")
        self.assertIsNone(missing.yolo_box)

        outside = convert_bbox_to_yolo(
            {"x_min": 110, "y_min": 10, "x_max": 120, "y_max": 30},
            image_width=100,
            image_height=80,
        )
        self.assertEqual(outside.status, "empty")
        self.assertIn("no visible area", outside.reason)

    def test_serialized_edge_box_stays_inside_image_after_rounding(self) -> None:
        result = convert_bbox_to_yolo(
            {
                "x_min": 0,
                "y_min": 162,
                "x_max": 14,
                "y_max": 199,
                "valid": False,
                "clipped": True,
            },
            image_width=640,
            image_height=480,
        )

        self.assertEqual(result.yolo_box.as_line(), "0 0.010938 0.376042 0.021875 0.077083")
        _, x_center, y_center, width, height = result.yolo_box.as_line().split()
        self.assertGreaterEqual(float(x_center) - float(width) / 2.0, 0.0)
        self.assertGreaterEqual(float(y_center) - float(height) / 2.0, 0.0)
        self.assertLessEqual(float(x_center) + float(width) / 2.0, 1.0)
        self.assertLessEqual(float(y_center) + float(height) / 2.0, 1.0)

    def test_group_split_is_deterministic(self) -> None:
        probabilities = normalize_split_probabilities(0.7, 0.2, 0.1)
        first = choose_split("capture-7", seed=42, probabilities=probabilities)
        second = choose_split("capture-7", seed=42, probabilities=probabilities)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})


class YoloDatasetExportTests(unittest.TestCase):
    def _write_schema(
        self,
        root: Path,
        *,
        start_capture: int = 0,
        capture_count: int = 3,
        filename: str = "schema.jsonl",
    ) -> Path:
        raw_dir = root / "raw"
        records = []
        for capture_index in range(start_capture, start_capture + capture_count):
            observations = []
            for camera_index in range(4):
                image_path = raw_dir / f"capture_{capture_index}_camera_{camera_index}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 80), color=(40, 80, 120)).save(image_path)
                if camera_index == 0:
                    bbox = {
                        "x_min": 10,
                        "y_min": 20,
                        "x_max": 40,
                        "y_max": 60,
                        "valid": True,
                        "clipped": False,
                        "semantic_label": "mannequin",
                    }
                elif camera_index == 1:
                    bbox = {
                        "x_min": -5,
                        "y_min": 10,
                        "x_max": 20,
                        "y_max": 30,
                        "valid": False,
                        "clipped": True,
                        "reason": "target bbox is clipped by the image boundary",
                        "semantic_label": "mannequin",
                    }
                elif camera_index == 2:
                    bbox = None
                else:
                    bbox = {
                        "x_min": 110,
                        "y_min": 10,
                        "x_max": 120,
                        "y_max": 30,
                        "valid": False,
                        "clipped": True,
                        "semantic_label": "mannequin",
                    }
                observations.append(
                    {
                        "camera_path": f"/World/Camera_{camera_index + 1:02d}",
                        "bbox": bbox,
                        "raw_image_path": str(image_path.relative_to(root)),
                        "raw_bbox_path": f"bbox_{capture_index}_{camera_index}.npy",
                    }
                )
            records.append(
                {
                    "schema_version": 2,
                    "capture": {
                        "scene_index": capture_index,
                        "capture_id": capture_index,
                        "target_label": "mannequin",
                        "resolution": [100, 80],
                    },
                    "camera_observations": observations,
                }
            )
        schema_path = root / filename
        schema_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return schema_path

    def test_export_writes_clean_images_empty_labels_and_grouped_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = self._write_schema(root)
            output_dir = root / "yolo"
            summary = export_dataset(schema_path, output_dir, overwrite=True, split_seed=9)

            self.assertEqual(summary["capture_count"], 3)
            self.assertEqual(summary["sample_count"], 12)
            self.assertEqual(summary["empty_label_count"], 6)
            self.assertEqual(summary["class_names"], ["mannequin"])
            self.assertTrue((output_dir / "data.yaml").is_file())
            self.assertIn("names: [\"mannequin\"]", (output_dir / "data.yaml").read_text())

            report = validate_dataset(output_dir)
            self.assertTrue(report.valid, report.errors)
            strict_report = validate_dataset(output_dir, strict=True)
            self.assertFalse(strict_report.valid)

            active_split = next(
                split_name
                for split_name, count in summary["split_counts"].items()
                if count > 0
            )
            preview_summary = visualize_dataset(
                output_dir,
                root / "previews",
                split=active_split,
                limit=12,
            )
            self.assertEqual(preview_summary["preview_count"], summary["split_counts"][active_split])
            self.assertGreater(preview_summary["box_count"], 0)

            manifest_items = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(manifest_items), 12)
            splits_by_group = {}
            for item in manifest_items:
                splits_by_group.setdefault(item["capture_group_id"], set()).add(item["split"])
                image_path = output_dir / item["image_path"]
                label_path = output_dir / item["label_path"]
                self.assertTrue(image_path.is_file())
                self.assertTrue(label_path.is_file())
                if item["annotation"]["status"] == "empty":
                    self.assertEqual(label_path.read_text(), "")
                else:
                    line = label_path.read_text().strip()
                    self.assertEqual(len(line.split()), 5)
                    values = [float(value) for value in line.split()[1:]]
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))

            invalid_label = output_dir / manifest_items[0]["label_path"]
            invalid_label.write_text(
                invalid_label.read_text() + "0 1.2 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            invalid_report = validate_dataset(output_dir)
            self.assertFalse(invalid_report.valid)
            self.assertTrue(
                any("coordinates must be in [0, 1]" in error for error in invalid_report.errors)
            )

    @staticmethod
    def _tree_digests(root: Path) -> dict[str, str]:
        digests = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative_path = path.relative_to(root).as_posix()
            digests[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digests

    def test_fixture_stress_export_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            schema_path = self._write_schema(root, capture_count=48)
            output_a = root / "yolo_a"
            output_b = root / "yolo_b"

            summary_a = export_dataset(schema_path, output_a, overwrite=True, split_seed=17)
            summary_b = export_dataset(schema_path, output_b, overwrite=True, split_seed=17)

            self.assertEqual(summary_a, {**summary_b, "output_dir": str(output_a.resolve())})
            self.assertEqual(summary_a["sample_count"], 192)
            self.assertEqual(summary_a["empty_label_count"], 96)
            self.assertEqual(self._tree_digests(output_a), self._tree_digests(output_b))

            report = validate_dataset(output_a)
            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.manifest_sample_count, 192)
            self.assertEqual(report.capture_group_count, 48)

    def test_append_preserves_existing_group_splits_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_schema = self._write_schema(
                root,
                start_capture=0,
                capture_count=4,
                filename="first.jsonl",
            )
            second_schema = self._write_schema(
                root,
                start_capture=4,
                capture_count=3,
                filename="second.jsonl",
            )
            output_dir = root / "yolo"
            export_dataset(first_schema, output_dir, overwrite=True, split_seed=11)
            first_manifest = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text().splitlines()
            ]
            first_splits = {
                item["capture_group_id"]: item["split"] for item in first_manifest
            }

            append_summary = export_dataset(
                second_schema,
                output_dir,
                append=True,
                split_seed=999,
            )
            self.assertEqual(append_summary["sample_count"], 12)
            self.assertEqual(append_summary["group_count"], 7)

            all_manifest = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text().splitlines()
            ]
            splits_by_group = {}
            for item in all_manifest:
                splits_by_group.setdefault(item["capture_group_id"], set()).add(item["split"])
            self.assertEqual(
                {group_id: next(iter(splits)) for group_id, splits in splits_by_group.items() if group_id in first_splits},
                first_splits,
            )
            self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
            self.assertTrue(validate_dataset(output_dir).valid)

            with self.assertRaisesRegex(ValueError, "already present"):
                export_dataset(first_schema, output_dir, append=True, split_seed=11)


if __name__ == "__main__":
    unittest.main()
