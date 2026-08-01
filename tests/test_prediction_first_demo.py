import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.demo_yolo_prediction_first import (
    DEMO_TRIALS,
    apply_trial_camera_condition,
    format_trial_changes,
    ground_truth_view_kwargs,
    prediction_view_kwargs,
    resolve_trial_position,
    run_display_sequence,
    run_trial_inference,
    save_demo_camera_images,
    validate_demo_preflight,
    wait_for_terminal,
)
from scripts.target_fusion import BoundingBox2D
from scripts.yolo_inference import YoloDetection, YoloInferenceResult


class TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonTtyStream(io.StringIO):
    def isatty(self) -> bool:
        return False


class FakeSimulationApp:
    def __init__(self) -> None:
        self.updates = 0

    def is_running(self) -> bool:
        return True

    def update(self) -> None:
        self.updates += 1


def detection_result() -> YoloInferenceResult:
    bbox = BoundingBox2D(
        2.0,
        1.0,
        7.0,
        5.0,
        resolution=(10, 6),
        semantic_id=0,
        semantic_label="mannequin",
    )
    detection = YoloDetection(
        bbox=bbox,
        confidence=0.875,
        class_id=0,
        class_name="mannequin",
        raw_xyxy=(2.0, 1.0, 7.0, 5.0),
    )
    return YoloInferenceResult(
        detection=detection,
        frame_resolution=(10, 6),
        inference_time_ms=2.0,
        total_detection_count=1,
        target_candidate_count=1,
        qualified_candidate_count=1,
    )


def miss_result() -> YoloInferenceResult:
    return YoloInferenceResult(
        detection=None,
        frame_resolution=(10, 6),
        inference_time_ms=2.0,
        total_detection_count=0,
        target_candidate_count=0,
        qualified_candidate_count=0,
        reason="no target detection",
    )


class PredictionFirstDemoTests(unittest.TestCase):
    def test_five_trials_have_fixed_expected_conditions(self) -> None:
        self.assertEqual(len(DEMO_TRIALS), 5)
        self.assertEqual(
            [trial.name for trial in DEMO_TRIALS],
            [
                "Clean baseline",
                "Positional challenge",
                "Reduced camera resolution",
                "Difficult illumination",
                "Combined stress",
            ],
        )
        self.assertEqual(DEMO_TRIALS[1].position_fraction_xy, (0.35, -0.20))
        self.assertEqual(DEMO_TRIALS[2].resolution_scale, 0.75)
        self.assertEqual(DEMO_TRIALS[3].color_temperature_k, 4800.0)
        self.assertEqual(DEMO_TRIALS[4].rgb_noise_std, 8.0)
        self.assertEqual(len(DEMO_TRIALS) * 4 * 2, 40)

    def test_relative_position_resolves_to_metric_offset(self) -> None:
        position, offset = resolve_trial_position(
            DEMO_TRIALS[1],
            ground_center=[10.0, 20.0, 0.0],
            ground_half_extent=[4.0, 5.0, 0.0],
            z_position=0.5,
        )
        np.testing.assert_allclose(offset, [1.4, -1.0, 0.0])
        np.testing.assert_allclose(position, [11.4, 19.0, 0.5])

    def test_clean_condition_is_identity_and_combined_is_deterministic(self) -> None:
        frame = np.full((24, 32, 3), 100, dtype=np.uint8)
        clean, clean_resolution = apply_trial_camera_condition(
            frame,
            DEMO_TRIALS[0],
            camera_index=0,
        )
        first, first_resolution = apply_trial_camera_condition(
            frame,
            DEMO_TRIALS[4],
            camera_index=2,
        )
        second, second_resolution = apply_trial_camera_condition(
            frame,
            DEMO_TRIALS[4],
            camera_index=2,
        )
        np.testing.assert_array_equal(clean, frame)
        self.assertEqual(clean_resolution, (32, 24))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, frame))
        self.assertEqual(first_resolution, (24, 18))
        self.assertEqual(second_resolution, first_resolution)

    def test_change_report_distinguishes_position_and_camera_conditions(self) -> None:
        positional = format_trial_changes(
            DEMO_TRIALS[1],
            position_offset=[1.4, -1.0, 0.0],
            intermediate_resolution=(640, 480),
        )
        combined = format_trial_changes(
            DEMO_TRIALS[4],
            position_offset=[-1.6, 1.5, 0.0],
            intermediate_resolution=(480, 360),
        )
        self.assertIn("X=+1.400 m", positional[1])
        self.assertEqual(positional[2], "Camera condition: clean")
        self.assertIn("resolution 0.75x", combined[2])
        self.assertIn("RGB Gaussian pixel noise sigma=8.0", combined[2])

    def test_saves_exact_input_and_separate_prediction_overlay(self) -> None:
        frame = np.full((6, 10, 3), (12, 34, 56), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path, prediction_path = save_demo_camera_images(
                frame,
                detection_result(),
                trial_directory=root,
                camera_index=0,
            )
            saved_input = np.asarray(Image.open(input_path).convert("RGB"))
            saved_prediction = np.asarray(Image.open(prediction_path).convert("RGB"))
            np.testing.assert_array_equal(saved_input, frame)
            self.assertFalse(np.array_equal(saved_prediction, frame))

            _, miss_path = save_demo_camera_images(
                frame,
                miss_result(),
                trial_directory=root,
                camera_index=1,
            )
            self.assertFalse(
                np.array_equal(np.asarray(Image.open(miss_path).convert("RGB")), frame)
            )

    def test_trial_inference_accepts_dummy_runner_without_checkpoint(self) -> None:
        dummy_model = object()
        frames = [np.zeros((6, 10, 3), dtype=np.uint8) for _ in range(4)]
        expected = [miss_result() for _ in frames]
        calls = []

        def dummy_runner(model, supplied_frames, **kwargs):
            calls.append((model, supplied_frames, kwargs))
            return expected

        actual = run_trial_inference(
            dummy_model,
            frames,
            confidence=0.25,
            iou=0.70,
            device="cpu",
            inference_runner=dummy_runner,
        )
        self.assertEqual(actual, expected)
        self.assertIs(calls[0][0], dummy_model)
        self.assertEqual(calls[0][2]["image_size"], 640)
        self.assertEqual(calls[0][2]["device"], "cpu")

    def test_display_order_requires_two_prompts(self) -> None:
        events = []
        responses = iter((True, True))
        completed = run_display_sequence(
            show_prediction=lambda: events.append("prediction"),
            show_ground_truth=lambda: events.append("ground_truth"),
            clear_display=lambda: events.append("clear"),
            wait=lambda prompt: events.append(prompt) or next(responses),
            report_ground_truth=lambda: events.append("report"),
        )
        self.assertTrue(completed)
        self.assertEqual(
            events,
            [
                "prediction",
                "Press Enter to reveal ground truth, or q to quit: ",
                "clear",
                "ground_truth",
                "report",
                "Press Enter for the next trial, or q to quit: ",
                "clear",
            ],
        )

    def test_quit_at_first_prompt_clears_without_revealing_truth(self) -> None:
        events = []
        completed = run_display_sequence(
            show_prediction=lambda: events.append("prediction"),
            show_ground_truth=lambda: events.append("ground_truth"),
            clear_display=lambda: events.append("clear"),
            wait=lambda _prompt: False,
            report_ground_truth=lambda: events.append("report"),
        )
        self.assertFalse(completed)
        self.assertEqual(events, ["prediction", "clear"])

    def test_prediction_draw_arguments_do_not_contain_truth(self) -> None:
        fusion = SimpleNamespace(
            rays=("yolo-ray",),
            fusion=SimpleNamespace(valid=True, fused_position_world=np.array([1.0, 2.0, 3.0])),
        )
        prediction = prediction_view_kwargs(fusion, ray_length=12.0)
        ground_truth = ground_truth_view_kwargs(
            fusion,
            target_world=np.array([4.0, 5.0, 6.0]),
            ray_length=12.0,
        )
        self.assertIsNone(prediction["truth_world"])
        self.assertEqual(prediction["ground_truth_rays"], [])
        np.testing.assert_allclose(ground_truth["truth_world"], [4.0, 5.0, 6.0])
        self.assertEqual(ground_truth["yolo_rays"], [])

    def test_terminal_wait_updates_app_and_accepts_enter_or_q(self) -> None:
        enter_app = FakeSimulationApp()
        self.assertTrue(
            wait_for_terminal(
                enter_app,
                "prompt: ",
                stream=TtyStream("\n"),
                select_fn=lambda read, _write, _error, _timeout: (read, [], []),
            )
        )
        self.assertGreater(enter_app.updates, 0)

        self.assertFalse(
            wait_for_terminal(
                FakeSimulationApp(),
                "prompt: ",
                stream=TtyStream("q\n"),
                select_fn=lambda read, _write, _error, _timeout: (read, [], []),
            )
        )

    def test_terminal_wait_rejects_noninteractive_and_eof(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminal-connected"):
            wait_for_terminal(FakeSimulationApp(), "prompt: ", stream=NonTtyStream("\n"))
        with self.assertRaisesRegex(EOFError, "closed"):
            wait_for_terminal(
                FakeSimulationApp(),
                "prompt: ",
                stream=TtyStream(""),
                select_fn=lambda read, _write, _error, _timeout: (read, [], []),
            )

    def test_preflight_requires_terminal_and_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist yet"):
                validate_demo_preflight(checkpoint, stream=TtyStream())
            checkpoint.touch()
            self.assertEqual(
                validate_demo_preflight(checkpoint, stream=TtyStream()),
                checkpoint.resolve(),
            )
            with self.assertRaisesRegex(RuntimeError, "launched from a terminal"):
                validate_demo_preflight(checkpoint, stream=NonTtyStream())


if __name__ == "__main__":
    unittest.main()
