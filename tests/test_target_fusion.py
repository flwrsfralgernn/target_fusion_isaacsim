import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.target_fusion import (
    BoundingBox2D,
    CameraCalibration,
    CameraObservation,
    CameraRay,
    FusionResult,
    build_camera_ray_from_observation,
    build_camera_ray_from_pixel,
    build_rays_from_available_observations,
    extract_target_bbox,
    build_ground_truth_ray,
    build_schema_v2_record,
    evaluate_fusion,
    fuse_rays,
    normalize,
)
from scripts.report_target_fusion import summarize_records
from scripts.cycle_ground_backgrounds import (
    basic_writer_capture_paths,
    capture_synchronized_camera_views,
    save_annotated_capture_image,
)


class TargetFusionMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.origins = [
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 4.0, 0.0]),
            np.array([0.0, 0.0, 6.0]),
        ]

    def test_normalize_returns_unit_vector(self) -> None:
        result = normalize([3.0, 4.0, 0.0])
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_camera_calibration_builds_intrinsics_without_square_pixel_assumption(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=10.0,
            origin_world=[1.0, 2.0, 3.0],
        )

        np.testing.assert_allclose(
            calibration.intrinsic_matrix,
            [[640.0, 0.0, 320.0], [0.0, 960.0, 240.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(calibration.origin_world, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(calibration.forward_world, [0.0, 0.0, -1.0])

    def test_camera_calibration_rejects_unsupported_projection(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported camera projection"):
            CameraCalibration(
                camera_path="camera",
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                projection="fisheye",
            )

    def test_bbox_center_preserves_fractional_pixels(self) -> None:
        bbox = BoundingBox2D(
            x_min=10,
            y_min=20,
            x_max=21,
            y_max=41,
            resolution=(640, 480),
            semantic_id=7,
            semantic_label="mannequin",
            occlusion_ratio=0.25,
        )
        np.testing.assert_allclose(bbox.center_uv, [15.5, 30.5])
        self.assertEqual(bbox.area, 231.0)

    def test_camera_observation_requires_matching_camera_path(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera_a",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=15.0,
        )
        bbox = BoundingBox2D(0, 0, 10, 10, resolution=(640, 480))
        with self.assertRaisesRegex(ValueError, "must match calibration"):
            CameraObservation("camera_b", calibration, bbox, capture_id=1)

    def test_camera_observation_is_json_serializable(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=15.0,
            origin_world=[1.0, 2.0, 3.0],
        )
        bbox = BoundingBox2D(
            10.0,
            20.0,
            31.0,
            42.0,
            resolution=(640, 480),
            semantic_id=9,
            semantic_label="mannequin",
        )
        observation = CameraObservation("camera", calibration, bbox, capture_id=4)
        encoded = json.dumps(observation.as_dict(), allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["camera_path"], "camera")
        self.assertEqual(decoded["bbox"]["center_uv"], [20.5, 31.0])
        self.assertEqual(decoded["calibration"]["resolution"], [640, 480])

    def test_bearing_ray_does_not_require_target_distance(self) -> None:
        ray = CameraRay("camera", [0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
        encoded = json.dumps(ray.as_dict(), allow_nan=False)
        self.assertIsNone(json.loads(encoded)["target_distance_m"])

    def test_center_pixel_maps_to_camera_forward(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=20.0,
            origin_world=[4.0, 5.0, 6.0],
        )
        ray = build_camera_ray_from_pixel(calibration, [320.0, 240.0])
        np.testing.assert_allclose(ray.origin_world, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(ray.direction_world, [0.0, 0.0, -1.0])

    def test_pixel_axes_use_right_and_down_image_convention(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=20.0,
        )
        right_ray = build_camera_ray_from_pixel(calibration, [960.0, 240.0])
        top_ray = build_camera_ray_from_pixel(calibration, [320.0, -240.0])
        np.testing.assert_allclose(right_ray.direction_world, [1.0, 0.0, -1.0] / np.sqrt(2.0))
        np.testing.assert_allclose(top_ray.direction_world, [0.0, 1.0, -1.0] / np.sqrt(2.0))

    def test_pixel_ray_uses_rotated_camera_pose(self) -> None:
        rotation_world_from_camera = np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        )
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=20.0,
            rotation_world_from_camera=rotation_world_from_camera,
        )
        ray = build_camera_ray_from_pixel(calibration, [320.0, 240.0])
        np.testing.assert_allclose(ray.direction_world, [1.0, 0.0, 0.0])

    def test_observation_bbox_center_builds_one_ray(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=20.0,
        )
        observation = CameraObservation(
            camera_path="camera",
            calibration=calibration,
            bbox=BoundingBox2D(310.0, 230.0, 330.0, 250.0, resolution=(640, 480)),
            capture_id=1,
        )
        ray = build_camera_ray_from_observation(observation)
        np.testing.assert_allclose(ray.direction_world, [0.0, 0.0, -1.0])

    def test_available_observations_still_generate_rays_when_one_camera_misses(self) -> None:
        observations = []
        for index, origin in enumerate(self.origins):
            path = f"camera_{index}"
            calibration = CameraCalibration(
                camera_path=path,
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                origin_world=origin,
            )
            if index == 3:
                bbox = BoundingBox2D(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    resolution=(640, 480),
                    valid=False,
                    reason="target bbox not found",
                )
            else:
                bbox = BoundingBox2D(310.0, 230.0, 330.0, 250.0, resolution=(640, 480))
            observations.append(CameraObservation(path, calibration, bbox, capture_id=3))

        rays = build_rays_from_available_observations(observations)

        self.assertEqual([ray.camera_path for ray in rays], ["camera_0", "camera_1", "camera_2"])
        self.assertFalse(observations[3].valid)
        self.assertEqual(observations[3].bbox.reason, "target bbox not found")

    def test_invalid_observation_cannot_build_ray(self) -> None:
        calibration = CameraCalibration(
            camera_path="camera",
            resolution=(640, 480),
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=20.0,
        )
        observation = CameraObservation("camera", calibration, None, capture_id=1)
        with self.assertRaisesRegex(ValueError, "has no target bbox"):
            build_camera_ray_from_observation(observation)

    def test_extract_target_bbox_unions_duplicate_semantic_rows(self) -> None:
        dtype = [
            ("semanticId", np.uint32),
            ("x_min", np.int32),
            ("y_min", np.int32),
            ("x_max", np.int32),
            ("y_max", np.int32),
            ("occlusionRatio", np.float32),
        ]
        rows = np.array(
            [
                (5, 10, 20, 31, 42, 0.10),
                (5, 12, 18, 35, 40, 0.20),
                (8, 0, 0, 100, 100, 0.0),
            ],
            dtype=dtype,
        )
        bbox = extract_target_bbox(
            rows,
            {"idToLabels": {"5": {"class": "mannequin"}, "8": {"class": "other"}}},
            resolution=(640, 480),
        )
        self.assertTrue(bbox.valid)
        self.assertEqual((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max), (10.0, 18.0, 35.0, 42.0))
        np.testing.assert_allclose(bbox.center_uv, [22.5, 30.0])
        self.assertAlmostEqual(bbox.occlusion_ratio, 0.20, places=5)

    def test_extract_target_bbox_reports_missing_semantic_target(self) -> None:
        rows = [{"semanticId": 4, "x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20}]
        bbox = extract_target_bbox(
            rows,
            {"idToLabels": {4: {"class": "other"}}},
            resolution=(640, 480),
        )
        self.assertFalse(bbox.valid)
        self.assertIn("not found", bbox.reason)

    def test_extract_target_bbox_rejects_clipping_and_excessive_occlusion(self) -> None:
        row = [{
            "semanticId": 5,
            "x_min": 0,
            "y_min": 20,
            "x_max": 100,
            "y_max": 200,
            "occlusionRatio": 0.75,
        }]
        info = {"idToLabels": {5: {"class": "mannequin"}}}
        clipped = extract_target_bbox(row, info, resolution=(640, 480))
        self.assertFalse(clipped.valid)
        self.assertTrue(clipped.clipped)
        self.assertIn("clipped", clipped.reason)

        occluded = extract_target_bbox(
            [{**row[0], "x_min": 10, "occlusionRatio": 0.75}],
            info,
            resolution=(640, 480),
            max_occlusion_ratio=0.5,
        )
        self.assertFalse(occluded.valid)
        self.assertIn("exceeds limit", occluded.reason)

    def test_extract_target_bbox_rejects_zero_area(self) -> None:
        row = [{
            "semanticId": 5,
            "x_min": 10,
            "y_min": 20,
            "x_max": 10,
            "y_max": 40,
        }]
        bbox = extract_target_bbox(
            row,
            {"idToLabels": {5: "mannequin"}},
            resolution=(640, 480),
        )
        self.assertFalse(bbox.valid)
        self.assertIn("zero or negative area", bbox.reason)

    def test_build_ground_truth_ray(self) -> None:
        ray = build_ground_truth_ray("/World/Camera_01", [0.0, 0.0, 0.0], self.target)
        np.testing.assert_allclose(ray.origin_world, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ray.direction_world, self.target / np.linalg.norm(self.target))
        self.assertAlmostEqual(ray.target_distance_m, float(np.linalg.norm(self.target)))
        np.testing.assert_allclose(ray.point_at(ray.target_distance_m), self.target)

    def test_exact_rays_recover_target(self) -> None:
        rays = [
            build_ground_truth_ray(f"camera_{index}", origin, self.target)
            for index, origin in enumerate(self.origins)
        ]
        result = fuse_rays(rays)
        self.assertTrue(result.valid)
        self.assertEqual(result.rank, 3)
        np.testing.assert_allclose(result.fused_position_world, self.target, atol=1e-10)
        self.assertLess(result.rms_residual_m, 1e-10)
        evaluation = evaluate_fusion(result, self.target)
        self.assertTrue(evaluation.valid)
        self.assertLess(evaluation.error_m, 1e-10)
        self.assertEqual(len(result.ray_diagnostics), 4)
        self.assertEqual(len(result.pairwise_angles_deg), 6)
        self.assertTrue(all(item["forward_distance_m"] > 0.0 for item in result.ray_diagnostics))
        encoded = json.dumps(result.as_dict(), allow_nan=False)
        self.assertEqual(json.loads(encoded)["min_pairwise_angle_deg"], min(result.pairwise_angles_deg))

    def test_fusion_rejects_solution_behind_a_camera(self) -> None:
        exact_rays = [
            build_ground_truth_ray(f"camera_{index}", origin, self.target)
            for index, origin in enumerate(self.origins)
        ]
        reversed_rays = [
            CameraRay(ray.camera_path, ray.origin_world, -ray.direction_world)
            for ray in exact_rays
        ]
        result = fuse_rays(reversed_rays)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "ray solution lies behind one or more cameras")
        self.assertIsNotNone(result.fused_position_world)
        self.assertTrue(all(item["forward_distance_m"] < 0.0 for item in result.ray_diagnostics))

    def test_fusion_evaluation_is_separate_from_estimation(self) -> None:
        ray = build_ground_truth_ray("camera", self.origins[0], self.target)
        result = fuse_rays([ray])
        self.assertFalse(result.valid)
        evaluation = evaluate_fusion(result, self.target)
        self.assertFalse(evaluation.valid)
        self.assertIsNone(evaluation.error_m)
        self.assertEqual(evaluation.reason, "at least two rays are required")

    def test_parallel_rays_are_invalid(self) -> None:
        rays = [
            CameraRay("camera_a", [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0),
            CameraRay("camera_b", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0),
        ]
        result = fuse_rays(rays)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "ray geometry is rank deficient")

    def test_insufficient_rays_are_invalid(self) -> None:
        ray = build_ground_truth_ray("camera", self.origins[0], self.target)
        result = fuse_rays([ray])
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "at least two rays are required")

    def test_invalid_vectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero length"):
            normalize([0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "cannot share the target position"):
            build_ground_truth_ray("camera", [1.0, 2.0, 3.0], self.target)

    def test_schema_v2_separates_observations_rays_and_truth_evaluation(self) -> None:
        observations = []
        rays = []
        for index, origin in enumerate(self.origins):
            path = f"camera_{index}"
            calibration = CameraCalibration(
                camera_path=path,
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                origin_world=origin,
            )
            observations.append(
                CameraObservation(
                    camera_path=path,
                    calibration=calibration,
                    bbox=BoundingBox2D(300.0, 220.0, 340.0, 260.0, resolution=(640, 480)),
                    capture_id=9,
                )
            )
            rays.append(CameraRay(path, origin, self.target - origin))
        fusion_result = fuse_rays(rays)
        evaluation = evaluate_fusion(fusion_result, self.target)
        record = build_schema_v2_record(
            scene_index=9,
            capture_id=9,
            background_path="background.png",
            target_prim_path="/World/Mannequin",
            resolution=(640, 480),
            target_label="mannequin",
            rt_subframes=1,
            observations=observations,
            rays=rays,
            fusion_result=fusion_result,
            fusion_evaluation=evaluation,
            settled=True,
            image_paths=[f"annotated_{index}.png" for index in range(4)],
            raw_image_paths=[f"raw_rgb_{index}.png" for index in range(4)],
            raw_bbox_paths=[f"raw_bbox_{index}.npy" for index in range(4)],
            raw_camera_params_paths=[f"camera_params_{index}.json" for index in range(4)],
        )
        decoded = json.loads(json.dumps(record, allow_nan=False))
        self.assertEqual(decoded["schema_version"], 2)
        self.assertEqual(len(decoded["camera_observations"]), 4)
        self.assertEqual(len(decoded["inferred_rays"]), 4)
        self.assertEqual(decoded["capture"]["valid_camera_count"], 4)
        self.assertIsNone(decoded["inferred_rays"][0]["ray"]["target_distance_m"])
        self.assertEqual(decoded["camera_observations"][0]["image_path"], "annotated_0.png")
        self.assertEqual(decoded["camera_observations"][0]["raw_image_path"], "raw_rgb_0.png")
        self.assertEqual(decoded["camera_observations"][0]["raw_bbox_path"], "raw_bbox_0.npy")
        self.assertEqual(
            decoded["camera_observations"][0]["raw_camera_params_path"],
            "camera_params_0.json",
        )
        self.assertIn("ground_truth_evaluation", decoded)
        self.assertNotIn("target_center_world", decoded)

    def test_noisy_pixel_centers_still_produce_finite_fusion(self) -> None:
        target = np.array([0.0, 0.0, 4.0])
        origins = [
            np.array([-1.0, -1.0, 0.0]),
            np.array([1.0, -1.0, 0.0]),
            np.array([-1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
        ]
        noisy_offsets = [(-2.0, 1.5), (1.0, -1.0), (2.5, 0.5), (-1.5, -2.0)]
        rays = []
        for index, (origin, (noise_u, noise_v)) in enumerate(zip(origins, noisy_offsets)):
            forward = normalize(target - origin)
            camera_z = -forward
            camera_x = normalize(np.cross([0.0, 0.0, 1.0], camera_z))
            camera_y = np.cross(camera_z, camera_x)
            calibration = CameraCalibration(
                camera_path=f"camera_{index}",
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                origin_world=origin,
                rotation_world_from_camera=np.column_stack([camera_x, camera_y, camera_z]),
            )
            observation = CameraObservation(
                camera_path=f"camera_{index}",
                calibration=calibration,
                bbox=BoundingBox2D(
                    319.0 + noise_u,
                    239.0 + noise_v,
                    321.0 + noise_u,
                    241.0 + noise_v,
                    resolution=(640, 480),
                ),
                capture_id=1,
            )
            rays.append(build_camera_ray_from_observation(observation))
        result = fuse_rays(rays)
        self.assertTrue(result.valid)
        self.assertTrue(np.all(np.isfinite(result.fused_position_world)))
        self.assertTrue(np.isfinite(result.rms_residual_m))

    def test_schema_v2_dropped_detection_preserves_camera_reason(self) -> None:
        observations = []
        rays = []
        for index, origin in enumerate(self.origins):
            path = f"camera_{index}"
            calibration = CameraCalibration(
                camera_path=path,
                resolution=(640, 480),
                focal_length=20.0,
                horizontal_aperture=20.0,
                vertical_aperture=15.0,
                origin_world=origin,
            )
            if index == 3:
                bbox = BoundingBox2D(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    resolution=(640, 480),
                    valid=False,
                    reason="target bbox not found",
                )
            else:
                bbox = BoundingBox2D(310.0, 230.0, 330.0, 250.0, resolution=(640, 480))
                rays.append(CameraRay(path, origin, self.target - origin))
            observations.append(CameraObservation(path, calibration, bbox, capture_id=2))
        fusion_result = FusionResult(
            fused_position_world=None,
            rms_residual_m=None,
            rank=0,
            condition_number=float("inf"),
            valid=False,
            reason="required four valid camera observations; got 3",
        )
        record = build_schema_v2_record(
            scene_index=2,
            capture_id=2,
            background_path="background.png",
            target_prim_path="/World/Mannequin",
            resolution=(640, 480),
            target_label="mannequin",
            rt_subframes=1,
            observations=observations,
            rays=rays,
            fusion_result=fusion_result,
            fusion_evaluation=None,
            settled=False,
        )
        self.assertFalse(record["inferred_rays"][3]["valid"])
        self.assertIsNone(record["inferred_rays"][3]["ray"])
        self.assertIn("not found", record["inferred_rays"][3]["reason"])

    def test_report_summarizes_validity_and_geometry(self) -> None:
        records = [
            {
                "schema_version": 2,
                "fusion": {
                    "valid": True,
                    "rms_residual_m": 0.1,
                    "min_pairwise_angle_deg": 12.0,
                    "condition_number": 10.0,
                },
                "ground_truth_evaluation": {"error_m": 0.2},
                "capture": {"valid_camera_count": 4},
            },
            {
                "schema_version": 2,
                "fusion": {"valid": False, "reason": "missing camera"},
                "capture": {"valid_camera_count": 3},
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["capture_count"], 2)
        self.assertEqual(summary["valid_capture_count"], 1)
        self.assertEqual(summary["four_camera_observation_count"], 1)
        self.assertAlmostEqual(summary["position_error_m"]["mean"], 0.2)
        self.assertEqual(summary["invalid_fusion_reasons"], {"missing camera": 1})

    def test_annotated_capture_image_is_written_with_bbox(self) -> None:
        bbox = BoundingBox2D(4.0, 5.0, 20.0, 25.0, resolution=(32, 24))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "camera.png"
            save_annotated_capture_image(
                np.zeros((24, 32, 3), dtype=np.uint8),
                bbox,
                output_path,
                target_label="mannequin",
            )
            self.assertTrue(output_path.is_file())
            from PIL import Image

            with Image.open(output_path) as image:
                self.assertEqual(image.size, (32, 24))


class SynchronizedCaptureTests(unittest.TestCase):
    def test_basic_writer_paths_match_common_output_layout(self) -> None:
        paths = basic_writer_capture_paths(
            Path("outputs/sdg_raw"),
            "TargetFusion_Camera_01",
            7,
        )
        self.assertEqual(
            paths["rgb"],
            Path("outputs/sdg_raw/rgb/TargetFusion_Camera_01_rgb_000007.png").resolve(),
        )
        self.assertEqual(
            paths["bbox"],
            Path(
                "outputs/sdg_raw/bounding_box_2d_tight/"
                "TargetFusion_Camera_01_bounding_box_2d_tight_000007.npy"
            ).resolve(),
        )
        self.assertEqual(
            paths["camera_params"],
            Path("outputs/sdg_raw/camera_params/TargetFusion_Camera_01_camera_params_000007.json").resolve(),
        )

    def test_basic_writer_paths_reject_path_like_render_product_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "filename-safe"):
            basic_writer_capture_paths(Path("outputs"), "camera/01", 0)

    def test_all_camera_buffers_are_served_by_one_orchestrator_step(self) -> None:
        events = []

        class FakeOrchestrator:
            def step(self, **kwargs):
                events.append(("step", kwargs))

        class FakeAnnotator:
            def __init__(self, payload):
                self.payload = payload

            def get_data(self):
                events.append("read")
                return self.payload

        bbox_dtype = [
            ("semanticId", np.uint32),
            ("x_min", np.int32),
            ("y_min", np.int32),
            ("x_max", np.int32),
            ("y_max", np.int32),
            ("occlusionRatio", np.float32),
        ]
        bbox_payload = {
            "data": np.array([(5, 10, 20, 31, 42, 0.0)], dtype=bbox_dtype),
            "info": {"idToLabels": {"5": {"class": "mannequin"}}},
        }
        rgb_payload = {"data": np.zeros((2, 2, 3), dtype=np.uint8)}
        bbox_annotators = [
            FakeAnnotator(bbox_payload),
            FakeAnnotator(bbox_payload),
        ]
        rgb_annotators = [
            FakeAnnotator(rgb_payload),
            FakeAnnotator(rgb_payload),
        ]

        target_bboxes, rgb_frames = capture_synchronized_camera_views(
            FakeOrchestrator(),
            bbox_annotators,
            rgb_annotators,
            rt_subframes=2,
            render_resolution=(640, 480),
            target_label="mannequin",
            max_occlusion_ratio=None,
            border_tolerance_px=0.0,
        )

        self.assertEqual(len(target_bboxes), 2)
        self.assertEqual(len(rgb_frames), 2)
        self.assertTrue(all(bbox.valid for bbox in target_bboxes))
        self.assertEqual(events[0], ("step", {
            "delta_time": 0.0,
            "rt_subframes": 2,
            "pause_timeline": True,
            "wait_for_render": True,
        }))
        self.assertEqual(events[1:], ["read"] * 4)

    def test_capture_leaves_render_products_enabled_for_writer_lifecycle(self) -> None:
        events = []

        class FakeHydraTexture:
            def __init__(self, name):
                self.name = name

            def set_updates_enabled(self, enabled):
                events.append((self.name, enabled))

        class FakeRenderProduct:
            def __init__(self, name):
                self.hydra_texture = FakeHydraTexture(name)

        class FakeOrchestrator:
            def step(self, **kwargs):
                events.append(("step", kwargs))

        class FakeAnnotator:
            def __init__(self, payload):
                self.payload = payload

            def get_data(self):
                return self.payload

        bbox_dtype = [
            ("semanticId", np.uint32),
            ("x_min", np.int32),
            ("y_min", np.int32),
            ("x_max", np.int32),
            ("y_max", np.int32),
            ("occlusionRatio", np.float32),
        ]
        bbox_payload = {
            "data": np.array([(5, 10, 20, 31, 42, 0.0)], dtype=bbox_dtype),
            "info": {"idToLabels": {"5": {"class": "mannequin"}}},
        }
        rgb_payload = {"data": np.zeros((2, 2, 3), dtype=np.uint8)}
        capture_synchronized_camera_views(
            FakeOrchestrator(),
            [FakeAnnotator(bbox_payload), FakeAnnotator(bbox_payload)],
            [FakeAnnotator(rgb_payload), FakeAnnotator(rgb_payload)],
            render_products=[FakeRenderProduct("camera_01"), FakeRenderProduct("camera_02")],
            rt_subframes=1,
            render_resolution=(640, 480),
            target_label="mannequin",
            max_occlusion_ratio=None,
            border_tolerance_px=0.0,
        )

        self.assertEqual(events, [
            ("step", {
                "delta_time": 0.0,
                "rt_subframes": 1,
                "pause_timeline": True,
                "wait_for_render": True,
            }),
        ])

    def test_capture_keeps_all_rgb_frames_when_one_bbox_is_missing(self) -> None:
        events = []

        class FakeOrchestrator:
            def step(self, **kwargs):
                events.append(("step", kwargs))

        class FakeAnnotator:
            def __init__(self, payload):
                self.payload = payload

            def get_data(self):
                events.append("read")
                return self.payload

        bbox_dtype = [
            ("semanticId", np.uint32),
            ("x_min", np.int32),
            ("y_min", np.int32),
            ("x_max", np.int32),
            ("y_max", np.int32),
            ("occlusionRatio", np.float32),
        ]
        rows = np.array([(5, 10, 20, 31, 42, 0.0)], dtype=bbox_dtype)
        visible_payload = {
            "data": rows,
            "info": {"idToLabels": {"5": {"class": "mannequin"}}},
        }
        missing_payload = {
            "data": rows,
            "info": {"idToLabels": {"5": {"class": "other"}}},
        }
        bbox_annotators = [
            FakeAnnotator(visible_payload),
            FakeAnnotator(visible_payload),
            FakeAnnotator(missing_payload),
            FakeAnnotator(visible_payload),
        ]
        rgb_annotators = [
            FakeAnnotator({"data": np.zeros((2, 2, 3), dtype=np.uint8)})
            for _ in range(4)
        ]

        target_bboxes, rgb_frames = capture_synchronized_camera_views(
            FakeOrchestrator(),
            bbox_annotators,
            rgb_annotators,
            rt_subframes=1,
            render_resolution=(640, 480),
            target_label="mannequin",
            max_occlusion_ratio=None,
            border_tolerance_px=0.0,
        )

        self.assertEqual(len(target_bboxes), 4)
        self.assertEqual(sum(bbox.valid for bbox in target_bboxes), 3)
        self.assertEqual(len(rgb_frames), 4)
        self.assertEqual(events[0][0], "step")
        self.assertEqual(events[1:], ["read"] * 8)


if __name__ == "__main__":
    unittest.main()
