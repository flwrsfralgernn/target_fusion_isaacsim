import unittest

from scripts.report_target_fusion import summarize_records, summarize_yolo_records


class YoloReportTests(unittest.TestCase):
    @staticmethod
    def _record() -> dict:
        return {
            "fusion": {"valid": True, "rms_residual_m": 0.1},
            "capture": {"valid_camera_count": 2},
            "yolo": {
                "model": {"reference": "yolo11n.pt"},
                "comparison_mode": "same-time",
                "inference": [
                    {
                        "valid": True,
                        "inference_time_ms": 4.0,
                        "detection": {"confidence": 0.8},
                    },
                    {
                        "valid": False,
                        "inference_time_ms": 5.0,
                        "reason": "no target detection",
                    },
                ],
                "comparison": {
                    "cameras": [
                        {
                            "bbox_iou": 0.5,
                            "center_error_px": 2.0,
                            "center_error_normalized": 0.01,
                            "ray_angle_deg": 1.5,
                        },
                        {
                            "bbox_iou": None,
                            "center_error_px": None,
                            "center_error_normalized": None,
                            "ray_angle_deg": None,
                        },
                    ],
                    "metrics": {"fused_position_delta_m": 0.4},
                    "yolo": {
                        "valid_camera_count": 1,
                        "fusion": {
                            "valid": True,
                            "rms_residual_m": 0.2,
                        },
                        "evaluation": {"error_m": 0.6},
                    },
                },
            },
        }

    def test_yolo_summary_reports_detection_and_comparison_metrics(self) -> None:
        summary = summarize_yolo_records([self._record()])

        self.assertEqual(summary["capture_count"], 1)
        self.assertEqual(summary["valid_detection_count"], 1)
        self.assertAlmostEqual(summary["detection_rate"], 0.5)
        self.assertAlmostEqual(summary["bbox_iou"]["mean"], 0.5)
        self.assertAlmostEqual(summary["fused_position_delta_m"]["mean"], 0.4)
        self.assertAlmostEqual(summary["position_error_m"]["mean"], 0.6)
        self.assertEqual(summary["model_references"], {"yolo11n.pt": 1})
        self.assertEqual(summary["inference_miss_reasons"], {"no target detection": 1})

    def test_ground_truth_only_summary_remains_without_yolo_section(self) -> None:
        summary = summarize_records([{"fusion": {"valid": False}}])

        self.assertIsNone(summarize_yolo_records([{"fusion": {"valid": False}}]))
        self.assertNotIn("yolo", summary)


if __name__ == "__main__":
    unittest.main()
