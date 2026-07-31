import json
import unittest

import numpy as np

from scripts.target_fusion import (
    CameraRay,
    build_ground_truth_ray,
    build_ground_truth_record,
    fuse_rays,
    normalize,
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
        result = fuse_rays(rays, target_world=self.target)
        self.assertTrue(result.valid)
        self.assertEqual(result.rank, 3)
        np.testing.assert_allclose(result.fused_position_world, self.target, atol=1e-10)
        self.assertLess(result.error_m, 1e-10)
        self.assertLess(result.rms_residual_m, 1e-10)

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

    def test_ground_truth_record_is_json_serializable(self) -> None:
        rays = [
            build_ground_truth_ray(f"camera_{index}", origin, self.target)
            for index, origin in enumerate(self.origins)
        ]
        camera_aims = [
            {
                "camera_path": ray.camera_path,
                "position_world": ray.origin_world,
                "forward_world": ray.direction_world,
                "desired_forward_world": ray.direction_world,
                "alignment": 1.0,
                "position_error_m": 0.0,
            }
            for ray in rays
        ]
        fusion_result = fuse_rays(rays, target_world=self.target)
        record = build_ground_truth_record(
            scene_index=3,
            background_path="background.png",
            target_prim_path="/World/Mannequin",
            target_world=self.target,
            mannequin_position_world=[1.0, 2.0, 0.5],
            camera_aims=camera_aims,
            rays=rays,
            fusion_result=fusion_result,
            settled=True,
        )

        encoded = json.dumps(record, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema_version"], 1)
        self.assertEqual(decoded["camera_count"], 4)
        self.assertEqual(decoded["fusion"]["valid"], True)
        self.assertEqual(decoded["cameras"][0]["ray"]["camera_path"], "camera_0")


if __name__ == "__main__":
    unittest.main()
