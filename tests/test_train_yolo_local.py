import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.train_yolo_local import (
    _cache_value,
    audit_dataset,
    label_dir_for_image_dir,
    load_data_yaml,
    resolve_device,
    split_image_paths,
    write_patched_data_yaml,
)


class LocalYoloTrainingTests(unittest.TestCase):
    def _write_dataset(self, root: Path, *, valid_key: str = "val") -> Path:
        for split in ("train", "val" if valid_key == "val" else "valid", "test"):
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), color=(30, 60, 90)).save(image_dir / f"{split}_000.png")
            (label_dir / f"{split}_000.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n" if split != "test" else "",
                encoding="utf-8",
            )
        config = {
            "path": ".",
            "train": "train/images",
            valid_key: "valid/images" if valid_key == "valid" else "val/images",
            "test": "test/images",
            "names": ["mannequin"],
        }
        yaml_path = root / "data.yaml"
        import yaml

        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return yaml_path

    def test_audit_accepts_generated_style_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            yaml_path = self._write_dataset(root)
            audit = audit_dataset(yaml_path)

            self.assertTrue(audit.valid, audit.errors)
            self.assertEqual(audit.class_names, ("mannequin",))
            self.assertEqual(audit.total_objects, 2)
            self.assertEqual(audit.splits[0].images, 1)
            self.assertEqual(audit.splits[1].empty_labels, 0)

    def test_audit_accepts_colab_valid_split_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            yaml_path = self._write_dataset(root, valid_key="valid")
            config, dataset_root, _ = load_data_yaml(yaml_path)
            paths = split_image_paths(config, dataset_root)

            self.assertIn("val", paths)
            self.assertEqual(paths["val"][0].name, "images")
            self.assertTrue(audit_dataset(yaml_path).valid)

    def test_audit_rejects_malformed_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            yaml_path = self._write_dataset(root)
            (root / "train" / "labels" / "train_000.txt").write_text(
                "0 0.5 0.5 1.2 0.25\n", encoding="utf-8"
            )

            audit = audit_dataset(yaml_path)

            self.assertFalse(audit.valid)
            self.assertTrue(any("in [0, 1]" in error for error in audit.splits[0].errors))

    def test_patched_yaml_maps_valid_to_val_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            yaml_path = self._write_dataset(root, valid_key="valid")
            original = yaml_path.read_text(encoding="utf-8")
            patched_path, patched = write_patched_data_yaml(yaml_path, root / "run" / "data_local.yaml")

            self.assertTrue(patched_path.is_file())
            self.assertEqual(patched["val"], "valid/images")
            self.assertNotIn("valid", patched)
            self.assertEqual(yaml_path.read_text(encoding="utf-8"), original)

    def test_helpers(self) -> None:
        self.assertEqual(_cache_value("none"), False)
        self.assertEqual(_cache_value("ram"), "ram")
        self.assertEqual(label_dir_for_image_dir(Path("/tmp/train/images")).name, "labels")
        self.assertEqual(resolve_device("cpu", allow_cpu=False), "cpu")


if __name__ == "__main__":
    unittest.main()
