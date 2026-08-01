import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.yolo_inference import (
    CameraComparison,
    FusionComparison,
    LoadedYoloModel,
    ObservationFusion,
    YoloDetection,
    YoloInferenceResult,
    bbox_iou,
    build_yolo_observations,
    compare_observation_fusions,
    fuse_observations,
    infer_yolo_frame,
    infer_yolo_frames,
    load_yolo_model,
    model_alias_paths,
    normalize_rgb_frame,
    resolve_model_path,
)
from scripts.target_fusion import (
    BoundingBox2D,
    CameraCalibration,
    CameraObservation,
    CameraRay,
    build_ground_truth_ray,
    evaluate_fusion,
    fuse_rays,
    normalize,
)


class FakeYoloModel:
    task = "detect"

    def __init__(self, names):
        self.names = names


class FakePredictModel(FakeYoloModel):
    def __init__(self, names, result):
        super().__init__(names)
        self.result = result
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self.result]


def fake_result(xyxy, confidences, class_ids):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(xyxy, dtype=np.float32),
            conf=np.asarray(confidences, dtype=np.float32),
            cls=np.asarray(class_ids, dtype=np.float32),
        )
    )


class YoloModelResolutionTests(unittest.TestCase):
    def _write_checkpoint(self, directory: Path, name: str = "custom.pt") -> Path:
        checkpoint = directory / name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"test checkpoint")
        return checkpoint

    def test_supported_aliases_resolve_under_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = {}
            for alias in ("yolo11n.pt", "yolo26n.pt"):
                expected[alias] = self._write_checkpoint(root, alias)

            self.assertEqual(model_alias_paths(project_dir=root), expected)
            for alias, path in expected.items():
                self.assertEqual(resolve_model_path(alias, project_dir=root), path.resolve())

    def test_explicit_relative_path_uses_requested_base_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = self._write_checkpoint(root, "weights/custom.pt")

            self.assertEqual(
                resolve_model_path(
                    "weights/custom.pt",
                    relative_to=root,
                ),
                checkpoint.resolve(),
            )

    def test_resolution_rejects_missing_or_non_pt_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                resolve_model_path("missing.pt", relative_to=root)

            non_checkpoint = root / "weights.onnx"
            non_checkpoint.write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "local .pt checkpoint"):
                resolve_model_path(non_checkpoint)

    def test_load_validates_detection_task_and_target_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = self._write_checkpoint(root)

            loaded = load_yolo_model(
                checkpoint,
                target_label="mannequin",
                model_factory=lambda _: FakeYoloModel({0: "person", 1: "mannequin"}),
            )

            self.assertIsInstance(loaded, LoadedYoloModel)
            self.assertEqual(loaded.info.target_class_id, 1)
            self.assertEqual(loaded.info.class_names, ("person", "mannequin"))
            self.assertEqual(loaded.info.as_dict()["target_label"], "mannequin")

    def test_load_accepts_sequence_class_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self._write_checkpoint(Path(temporary_directory))

            loaded = load_yolo_model(
                checkpoint,
                model_factory=lambda _: FakeYoloModel(["mannequin"]),
            )

            self.assertEqual(loaded.info.target_class_id, 0)

    def test_load_rejects_wrong_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self._write_checkpoint(Path(temporary_directory))

            class SegmentModel(FakeYoloModel):
                task = "segment"

            with self.assertRaisesRegex(ValueError, "not a detection model"):
                load_yolo_model(
                    checkpoint,
                    model_factory=lambda _: SegmentModel(["mannequin"]),
                )

    def test_load_rejects_missing_target_class_with_available_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self._write_checkpoint(Path(temporary_directory))

            with self.assertRaisesRegex(ValueError, "does not contain target class") as context:
                load_yolo_model(
                    checkpoint,
                    model_factory=lambda _: FakeYoloModel({0: "person"}),
                )

            self.assertIn("person", str(context.exception))

    def test_load_rejects_noncontiguous_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = self._write_checkpoint(Path(temporary_directory))

            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_yolo_model(
                    checkpoint,
                    model_factory=lambda _: FakeYoloModel({1: "mannequin"}),
                )


class YoloFrameInferenceTests(unittest.TestCase):
    def _load_fake_model(self, root: Path, result) -> tuple[LoadedYoloModel, FakePredictModel]:
        checkpoint = root / "custom.pt"
        checkpoint.write_bytes(b"test checkpoint")
        model = FakePredictModel(["person", "mannequin"], result)
        loaded = load_yolo_model(checkpoint, model_factory=lambda _: model)
        return loaded, model

    def test_normalize_rgb_frame_handles_float_rgba(self) -> None:
        frame = np.array(
            [[[0.0, 0.5, 1.0, 0.25], [1.0, -1.0, 0.25, 0.0]]],
            dtype=np.float32,
        )

        normalized = normalize_rgb_frame(frame)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, (1, 2, 3))
        np.testing.assert_array_equal(normalized[0, 0], [0, 127, 255])
        np.testing.assert_array_equal(normalized[0, 1], [255, 0, 63])

    def test_inference_filters_target_class_and_selects_highest_confidence(self) -> None:
        result = fake_result(
            [
                [1, 2, 10, 12],
                [20, 22, 30, 32],
                [3, 4, 13, 14],
                [40, 42, 50, 52],
            ],
            [0.99, 0.70, 0.90, 0.10],
            [0, 1, 1, 1],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded, model = self._load_fake_model(Path(temporary_directory), result)
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            frame[0, 0] = [10, 20, 30]

            inference = infer_yolo_frame(
                loaded,
                frame,
                confidence_threshold=0.5,
                image_size=320,
                device="cpu",
            )

        self.assertTrue(inference.valid)
        self.assertEqual(inference.total_detection_count, 4)
        self.assertEqual(inference.target_candidate_count, 3)
        self.assertEqual(inference.qualified_candidate_count, 2)
        self.assertAlmostEqual(inference.detection.confidence, 0.90, places=5)
        np.testing.assert_allclose(inference.detection.bbox.center_uv, [8.0, 9.0])
        self.assertEqual(model.predict_calls[0]["classes"], [1])
        self.assertEqual(model.predict_calls[0]["device"], "cpu")
        np.testing.assert_array_equal(model.predict_calls[0]["source"][0, 0], [30, 20, 10])

    def test_inference_reports_below_threshold_target_miss(self) -> None:
        result = fake_result([[1, 2, 10, 12]], [0.20], [1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded, _ = self._load_fake_model(Path(temporary_directory), result)
            inference = infer_yolo_frame(
                loaded,
                np.zeros((20, 30, 3), dtype=np.uint8),
                confidence_threshold=0.5,
            )

        self.assertFalse(inference.valid)
        self.assertEqual(inference.target_candidate_count, 1)
        self.assertEqual(inference.qualified_candidate_count, 0)
        self.assertIn("below confidence", inference.reason)

    def test_inference_clamps_visible_edge_box_and_marks_it_clipped(self) -> None:
        result = fake_result([[-3, 2, 10, 25]], [0.8], [1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded, _ = self._load_fake_model(Path(temporary_directory), result)
            inference = infer_yolo_frame(
                loaded,
                np.zeros((20, 30, 3), dtype=np.uint8),
            )

        self.assertTrue(inference.valid)
        self.assertTrue(inference.detection.bbox.clipped)
        self.assertEqual(
            (
                inference.detection.bbox.x_min,
                inference.detection.bbox.y_min,
                inference.detection.bbox.x_max,
                inference.detection.bbox.y_max,
            ),
            (0.0, 2.0, 10.0, 20.0),
        )

    def test_inference_reports_empty_prediction(self) -> None:
        result = fake_result([], [], [])
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded, _ = self._load_fake_model(Path(temporary_directory), result)
            inference = infer_yolo_frame(
                loaded,
                np.zeros((20, 30, 3), dtype=np.uint8),
            )

        self.assertFalse(inference.valid)
        self.assertEqual(inference.total_detection_count, 0)
        self.assertIn("target class", inference.reason)

    def test_infer_frames_preserves_camera_order(self) -> None:
        result = fake_result([[1, 2, 10, 12]], [0.8], [1])
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded, model = self._load_fake_model(Path(temporary_directory), result)
            frames = [
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.full((20, 30, 3), 17, dtype=np.uint8),
            ]

            inferences = infer_yolo_frames(loaded, frames, device="cpu")

        self.assertEqual(len(inferences), 2)
        self.assertTrue(all(inference.valid for inference in inferences))
        self.assertEqual(len(model.predict_calls), 2)
        self.assertEqual(model.predict_calls[0]["source"][0, 0].tolist(), [0, 0, 0])
        self.assertEqual(model.predict_calls[1]["source"][0, 0].tolist(), [17, 17, 17])


class YoloFusionComparisonTests(unittest.TestCase):
    def _calibrations(self, count: int = 4) -> list[CameraCalibration]:
        return [
            CameraCalibration(
                camera_path=f"camera_{index}",
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                origin_world=[float(index), 0.0, 0.0],
            )
            for index in range(count)
        ]

    @staticmethod
    def _detection(x_min: float, y_min: float, x_max: float, y_max: float, confidence: float):
        bbox = BoundingBox2D(
            x_min,
            y_min,
            x_max,
            y_max,
            resolution=(640, 480),
            semantic_id=0,
            semantic_label="mannequin",
        )
        return YoloDetection(
            bbox=bbox,
            confidence=confidence,
            class_id=0,
            class_name="mannequin",
            raw_xyxy=(x_min, y_min, x_max, y_max),
        )

    def test_build_yolo_observations_preserves_valid_and_missing_cameras(self) -> None:
        calibrations = self._calibrations(count=2)
        results = [
            YoloInferenceResult(
                detection=self._detection(10, 20, 30, 40, 0.8),
                frame_resolution=(640, 480),
                inference_time_ms=2.0,
                total_detection_count=1,
                target_candidate_count=1,
                qualified_candidate_count=1,
            ),
            YoloInferenceResult(
                detection=None,
                frame_resolution=(640, 480),
                inference_time_ms=2.0,
                total_detection_count=0,
                target_candidate_count=0,
                qualified_candidate_count=0,
                reason="no target detection",
            ),
        ]

        observations = build_yolo_observations(calibrations, results, capture_id=7)

        self.assertEqual(len(observations), 2)
        self.assertTrue(observations[0].valid)
        self.assertEqual(observations[0].bbox.semantic_label, "mannequin")
        self.assertFalse(observations[1].valid)
        self.assertEqual(observations[1].reason, "no target detection")

    def test_fuse_observations_allows_partial_available_rays(self) -> None:
        calibrations = self._calibrations(count=2)
        observations = [
            CameraObservation(
                calibration=calibrations[0],
                camera_path=calibrations[0].camera_path,
                bbox=BoundingBox2D(310, 230, 330, 250, resolution=(640, 480)),
                capture_id=1,
            ),
            CameraObservation(
                calibration=calibrations[1],
                camera_path=calibrations[1].camera_path,
                bbox=None,
                capture_id=1,
                valid=False,
                reason="no target detection",
            ),
        ]

        source = fuse_observations(observations)

        self.assertEqual(source.valid_camera_count, 1)
        self.assertEqual(len(source.rays), 1)
        self.assertFalse(source.fusion.valid)
        self.assertEqual(source.fusion.reason, "at least two rays are required")

    def test_comparison_reports_bbox_ray_and_fused_position_metrics(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        origins = [
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 4.0, 0.0]),
            np.array([0.0, 0.0, 6.0]),
        ]
        calibrations = self._calibrations()
        ground_truth_observations = []
        yolo_results = []
        yolo_observations = []
        for index, calibration in enumerate(calibrations):
            ground_truth_observations.append(
                CameraObservation(
                    camera_path=calibration.camera_path,
                    calibration=calibration,
                    bbox=BoundingBox2D(10, 20, 30, 40, resolution=(640, 480)),
                    capture_id=3,
                )
            )
            yolo_result = YoloInferenceResult(
                detection=self._detection(12, 20, 32, 40, 0.75),
                frame_resolution=(640, 480),
                inference_time_ms=2.0,
                total_detection_count=1,
                target_candidate_count=1,
                qualified_candidate_count=1,
            )
            yolo_results.append(yolo_result)
        yolo_observations = build_yolo_observations(
            calibrations,
            yolo_results,
            capture_id=3,
        )

        ground_truth_rays = tuple(
            build_ground_truth_ray(calibration.camera_path, origin, target)
            for calibration, origin in zip(calibrations, origins)
        )
        yolo_rays = tuple(
            CameraRay(
                ray.camera_path,
                ray.origin_world,
                normalize(
                    ray.direction_world + (np.array([0.01, -0.005, 0.0]) if index == 0 else 0.0),
                    name="test YOLO direction",
                ),
            )
            for index, ray in enumerate(ground_truth_rays)
        )
        ground_truth_source = ObservationFusion(
            observations=tuple(ground_truth_observations),
            rays=ground_truth_rays,
            fusion=fuse_rays(ground_truth_rays),
            evaluation=evaluate_fusion(fuse_rays(ground_truth_rays), target),
            valid_camera_count=4,
        )
        yolo_fusion = fuse_rays(yolo_rays)
        yolo_source = ObservationFusion(
            observations=tuple(yolo_observations),
            rays=yolo_rays,
            fusion=yolo_fusion,
            evaluation=evaluate_fusion(yolo_fusion, target),
            valid_camera_count=4,
        )

        comparison = compare_observation_fusions(
            ground_truth_source,
            yolo_source,
            inference_results=yolo_results,
        )

        self.assertIsInstance(comparison, FusionComparison)
        self.assertIsInstance(comparison.cameras[0], CameraComparison)
        self.assertAlmostEqual(comparison.cameras[0].bbox_iou, 18.0 / 22.0)
        self.assertAlmostEqual(comparison.cameras[0].center_error_px, 2.0)
        self.assertGreater(comparison.cameras[0].ray_angle_deg, 0.0)
        self.assertGreater(comparison.fused_position_delta_m, 0.0)
        self.assertEqual(comparison.metrics()["ray_comparison_count"], 4)

    def test_bbox_iou_returns_none_for_missing_or_zero_area_boxes(self) -> None:
        box = BoundingBox2D(0, 0, 10, 10, resolution=(640, 480))
        zero_area = BoundingBox2D(0, 0, 0, 10, resolution=(640, 480), valid=False)

        self.assertIsNone(bbox_iou(None, box))
        self.assertIsNone(bbox_iou(zero_area, box))
        self.assertEqual(bbox_iou(box, box), 1.0)


if __name__ == "__main__":
    unittest.main()
