import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from scripts.target_fusion import (
    DEFAULT_GROUND_TRUTH_FUSED_COLOR,
    DEFAULT_GROUND_TRUTH_RAY_COLOR,
    DEFAULT_TRUTH_EVALUATION_COLOR,
    DEFAULT_YOLO_FUSED_COLOR,
    DEFAULT_YOLO_RAY_COLOR,
    CameraRay,
    draw_comparison_rays,
)


class FakeDrawInterface:
    def __init__(self) -> None:
        self.clear_lines_count = 0
        self.clear_points_count = 0
        self.line_calls = []
        self.point_calls = []

    def clear_lines(self) -> None:
        self.clear_lines_count += 1

    def clear_points(self) -> None:
        self.clear_points_count += 1

    def draw_lines(self, starts, ends, colors, thicknesses) -> None:
        self.line_calls.append((starts, ends, colors, thicknesses))

    def draw_points(self, positions, colors, sizes) -> None:
        self.point_calls.append((positions, colors, sizes))


class FakeDebugDraw:
    def __init__(self, interface: FakeDrawInterface) -> None:
        self.interface = interface

    def acquire_debug_draw_interface(self):
        return self.interface


def debug_draw_modules(debug_draw):
    isaacsim = types.ModuleType("isaacsim")
    util = types.ModuleType("isaacsim.util")
    debug_draw_module = types.ModuleType("isaacsim.util.debug_draw")
    isaacsim.util = util
    util.debug_draw = debug_draw_module
    debug_draw_module._debug_draw = debug_draw
    return {
        "isaacsim": isaacsim,
        "isaacsim.util": util,
        "isaacsim.util.debug_draw": debug_draw_module,
    }


class ComparisonRayVisualizationTests(unittest.TestCase):
    def test_overlay_uses_shared_length_and_distinct_source_colors(self) -> None:
        interface = FakeDrawInterface()
        debug_draw = FakeDebugDraw(interface)
        ground_truth_ray = CameraRay("gt_camera", [0, 0, 0], [1, 0, 0])
        yolo_ray = CameraRay("yolo_camera", [0, 1, 0], [0, 1, 0])

        with patch.dict(sys.modules, debug_draw_modules(debug_draw)):
            result = draw_comparison_rays(
                [ground_truth_ray],
                [yolo_ray],
                ground_truth_fused_position_world=[1, 2, 3],
                yolo_fused_position_world=[2, 3, 4],
                truth_world=[3, 4, 5],
                ray_length=5.0,
            )

        self.assertEqual(interface.clear_lines_count, 1)
        self.assertEqual(interface.clear_points_count, 1)
        self.assertEqual(len(interface.line_calls), 2)
        ground_truth_call, yolo_call = interface.line_calls
        self.assertEqual(ground_truth_call[2], [list(DEFAULT_GROUND_TRUTH_RAY_COLOR)])
        self.assertEqual(yolo_call[2], [list(DEFAULT_YOLO_RAY_COLOR)])
        np.testing.assert_allclose(ground_truth_call[1], [[5.0, 0.0, 0.0]])
        np.testing.assert_allclose(yolo_call[1], [[0.0, 6.0, 0.0]])

        self.assertEqual(len(interface.point_calls), 1)
        positions, colors, sizes = interface.point_calls[0]
        self.assertEqual(positions, [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        self.assertEqual(
            colors,
            [
                list(DEFAULT_GROUND_TRUTH_FUSED_COLOR),
                list(DEFAULT_YOLO_FUSED_COLOR),
                list(DEFAULT_TRUTH_EVALUATION_COLOR),
            ],
        )
        self.assertEqual(sizes, [12.0, 12.0, 12.0])
        self.assertEqual(result["ground_truth_ray_count"], 1)
        self.assertEqual(result["yolo_ray_count"], 1)
        self.assertEqual(result["ray_length"], 5.0)
        np.testing.assert_allclose(result["ground_truth_endpoints_world"], [[5.0, 0.0, 0.0]])
        np.testing.assert_allclose(result["yolo_endpoints_world"], [[0.0, 6.0, 0.0]])

    def test_ground_truth_only_view_can_omit_yolo_rays(self) -> None:
        interface = FakeDrawInterface()
        debug_draw = FakeDebugDraw(interface)
        ray = CameraRay("camera", [1, 2, 3], [0, 0, 1])

        with patch.dict(sys.modules, debug_draw_modules(debug_draw)):
            result = draw_comparison_rays(
                [ray],
                [],
                clear_existing=False,
                truth_world=[1, 2, 4],
            )

        self.assertEqual(interface.clear_lines_count, 0)
        self.assertEqual(interface.clear_points_count, 0)
        self.assertEqual(len(interface.line_calls), 1)
        self.assertEqual(len(interface.point_calls), 1)
        self.assertEqual(result["yolo_ray_count"], 0)
        self.assertEqual(result["yolo_fused_position_world"], None)

    def test_renderer_rejects_empty_geometry_and_invalid_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one ray or marker"):
            draw_comparison_rays()

        ray = CameraRay("camera", [0, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "ray_length"):
            draw_comparison_rays([ray], ray_length=0.0)


if __name__ == "__main__":
    unittest.main()
