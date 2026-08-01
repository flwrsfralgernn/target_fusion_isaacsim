import argparse
import unittest

from scripts.cycle_ground_backgrounds import (
    attach_yolo_comparison_record,
    validate_yolo_options,
    validate_yolo_runtime,
)


def yolo_args(**overrides):
    values = {
        "yolo_comparison_mode": "disabled",
        "yolo_model": None,
        "yolo_confidence_threshold": 0.25,
        "yolo_iou_threshold": 0.70,
        "yolo_image_size": 640,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LiveYoloConfigurationTests(unittest.TestCase):
    def test_default_ground_truth_only_configuration_is_valid(self) -> None:
        validate_yolo_options(yolo_args())

    def test_enabled_mode_requires_a_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --yolo-model"):
            validate_yolo_options(yolo_args(yolo_comparison_mode="same-time"))

    def test_model_requires_an_enabled_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --yolo-comparison-mode"):
            validate_yolo_options(yolo_args(yolo_model="yolo11n.pt"))

    def test_enabled_mode_accepts_local_model_and_validates_thresholds(self) -> None:
        validate_yolo_options(
            yolo_args(
                yolo_model="outputs/yolo_training_runs/mannequin_yolo11n_bbox/weights/best.pt",
                yolo_comparison_mode="after-ground-truth",
            )
        )
        with self.assertRaisesRegex(ValueError, "confidence-threshold"):
            validate_yolo_options(
                yolo_args(
                    yolo_model="model.pt",
                    yolo_comparison_mode="same-time",
                    yolo_confidence_threshold=1.1,
                )
            )

    def test_disabled_mode_does_not_require_detector_runtime(self) -> None:
        def missing_package(_name):
            return None

        validate_yolo_runtime("disabled", module_finder=missing_package)

    def test_enabled_mode_reports_missing_isaac_detector_runtime(self) -> None:
        def missing_package(_name):
            return None

        with self.assertRaisesRegex(RuntimeError, "Isaac Sim Python environment"):
            validate_yolo_runtime("same-time", module_finder=missing_package)

    def test_enabled_mode_accepts_available_detector_runtime(self) -> None:
        def available_package(_name):
            return object()

        validate_yolo_runtime("after-ground-truth", module_finder=available_package)

    def test_schema_record_receives_serializable_yolo_block(self) -> None:
        class Serializable:
            def __init__(self, value):
                self.value = value

            def as_dict(self):
                return self.value

        record = {"schema_version": 2}
        result = attach_yolo_comparison_record(
            record,
            model_info=Serializable({"path": "model.pt"}),
            comparison_mode="same-time",
            inference_results=[Serializable({"valid": True})],
            comparison=Serializable({"metrics": {"mean_bbox_iou": 1.0}}),
        )

        self.assertIs(result, record)
        self.assertEqual(result["yolo"]["comparison_mode"], "same-time")
        self.assertEqual(result["yolo"]["inference"], [{"valid": True}])
        self.assertEqual(result["yolo"]["comparison"]["metrics"]["mean_bbox_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
