import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.cycle_ground_backgrounds import (
    BASE_COLOR_TEMPERATURE_K,
    DEFAULT_BRIGHTNESS_NOISE_STD,
    DEFAULT_COLOR_TEMPERATURE_NOISE_STD_K,
    DEFAULT_EXPOSURE_NOISE_STD_STOPS,
    DEFAULT_POSITION_NOISE_STD_M,
    DEFAULT_RGB_PIXEL_NOISE_STD,
    DEFAULT_RESOLUTION_NOISE_STD,
    _coerce_position_noise_std,
    _coerce_photometric_noise_std,
    _coerce_resolution_noise_std,
    _coerce_rgb_pixel_noise_std,
    _color_temperature_rgb_gains,
    add_gaussian_position_noise,
    apply_gaussian_resolution_noise,
    apply_gaussian_photometric_noise,
    apply_gaussian_rgb_pixel_noise,
    parse_args,
    save_training_capture_image,
)


class SensorNoiseCliTests(unittest.TestCase):
    def test_sensor_noise_is_disabled_by_default(self) -> None:
        with patch.object(sys, "argv", ["cycle_ground_backgrounds.py"]):
            args = parse_args()

        self.assertFalse(args.sensor_noise)
        self.assertEqual(tuple(args.position_noise_std), DEFAULT_POSITION_NOISE_STD_M)
        self.assertEqual(args.resolution_noise_std, DEFAULT_RESOLUTION_NOISE_STD)
        self.assertEqual(args.brightness_noise_std, DEFAULT_BRIGHTNESS_NOISE_STD)
        self.assertEqual(args.exposure_noise_std, DEFAULT_EXPOSURE_NOISE_STD_STOPS)
        self.assertEqual(
            args.color_temperature_noise_std,
            DEFAULT_COLOR_TEMPERATURE_NOISE_STD_K,
        )
        self.assertEqual(args.rgb_pixel_noise_std, DEFAULT_RGB_PIXEL_NOISE_STD)

    def test_sensor_noise_can_be_enabled(self) -> None:
        with patch.object(sys, "argv", ["cycle_ground_backgrounds.py", "--sensor-noise"]):
            args = parse_args()

        self.assertTrue(args.sensor_noise)

    def test_position_noise_standard_deviation_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["cycle_ground_backgrounds.py", "--position-noise-std", "0.1", "0.2", "0.3"],
        ):
            args = parse_args()

        self.assertEqual(args.position_noise_std, [0.1, 0.2, 0.3])

    def test_position_noise_is_reproducible_and_axis_specific(self) -> None:
        first_position, first_offset = add_gaussian_position_noise(
            [1.0, 2.0, 3.0],
            [0.0, 0.1, 0.0],
            random.Random(7),
        )
        second_position, second_offset = add_gaussian_position_noise(
            [1.0, 2.0, 3.0],
            [0.0, 0.1, 0.0],
            random.Random(7),
        )

        np.testing.assert_allclose(first_position, second_position)
        np.testing.assert_allclose(first_offset, second_offset)
        self.assertEqual(first_offset[0], 0.0)
        self.assertNotEqual(first_offset[1], 0.0)
        self.assertEqual(first_offset[2], 0.0)
        np.testing.assert_allclose(first_position, np.asarray([1.0, 2.0, 3.0]) + first_offset)

    def test_position_noise_standard_deviation_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _coerce_position_noise_std([0.01, -0.01, 0.01])

    def test_resolution_noise_can_sample_downscaling_and_upscaling(self) -> None:
        frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)

        downscaled, down_scale, down_resolution = apply_gaussian_resolution_noise(
            frame,
            0.2,
            random.Random(5),
        )
        upscaled, up_scale, up_resolution = apply_gaussian_resolution_noise(
            frame,
            0.2,
            random.Random(1),
        )

        self.assertEqual(downscaled.shape, frame.shape)
        self.assertEqual(upscaled.shape, frame.shape)
        self.assertLess(down_scale, 1.0)
        self.assertLess(down_resolution[0], frame.shape[1])
        self.assertGreater(up_scale, 1.0)
        self.assertGreater(up_resolution[0], frame.shape[1])

    def test_resolution_noise_is_reproducible(self) -> None:
        frame = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        first = apply_gaussian_resolution_noise(frame, 0.15, random.Random(11))
        second = apply_gaussian_resolution_noise(frame, 0.15, random.Random(11))

        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])

    def test_resolution_noise_standard_deviation_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _coerce_resolution_noise_std(-0.1)

    def test_photometric_noise_cli_values_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "cycle_ground_backgrounds.py",
                "--brightness-noise-std",
                "0.08",
                "--exposure-noise-std",
                "0.4",
                "--color-temperature-noise-std",
                "750",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.brightness_noise_std, 0.08)
        self.assertEqual(args.exposure_noise_std, 0.4)
        self.assertEqual(args.color_temperature_noise_std, 750.0)

    def test_photometric_noise_is_reproducible_and_preserves_alpha(self) -> None:
        frame = np.full((12, 16, 4), 100, dtype=np.uint8)
        frame[:, :, 3] = 211
        kwargs = {
            "brightness_standard_deviation": 0.04,
            "exposure_standard_deviation_stops": 0.25,
            "color_temperature_standard_deviation_k": 500.0,
        }
        first = apply_gaussian_photometric_noise(
            frame,
            randomizer=random.Random(19),
            **kwargs,
        )
        second = apply_gaussian_photometric_noise(
            frame,
            randomizer=random.Random(19),
            **kwargs,
        )

        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(first[0][:, :, 3], frame[:, :, 3])
        self.assertFalse(np.array_equal(first[0][:, :, :3], frame[:, :, :3]))

    def test_zero_photometric_noise_is_identity(self) -> None:
        frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        result, sample = apply_gaussian_photometric_noise(
            frame,
            brightness_standard_deviation=0.0,
            exposure_standard_deviation_stops=0.0,
            color_temperature_standard_deviation_k=0.0,
            randomizer=random.Random(2),
        )

        np.testing.assert_array_equal(result, frame)
        self.assertEqual(sample["brightness_offset"], 0.0)
        self.assertEqual(sample["exposure_stops"], 0.0)
        self.assertEqual(sample["color_temperature_k"], BASE_COLOR_TEMPERATURE_K)

    def test_color_temperature_gains_are_warm_and_cool(self) -> None:
        warm = _color_temperature_rgb_gains(3500.0)
        neutral = _color_temperature_rgb_gains(BASE_COLOR_TEMPERATURE_K)
        cool = _color_temperature_rgb_gains(9000.0)

        self.assertGreater(warm[0], warm[2])
        np.testing.assert_allclose(neutral, [1.0, 1.0, 1.0])
        self.assertGreater(cool[2], cool[0])

    def test_photometric_standard_deviations_reject_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _coerce_photometric_noise_std(-0.1, option_name="--brightness-noise-std")

    def test_rgb_pixel_noise_cli_value_can_be_configured(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["cycle_ground_backgrounds.py", "--rgb-pixel-noise-std", "12.5"],
        ):
            args = parse_args()

        self.assertEqual(args.rgb_pixel_noise_std, 12.5)

    def test_rgb_pixel_noise_is_reproducible_and_independent_per_channel(self) -> None:
        frame = np.full((32, 40, 3), 128, dtype=np.uint8)
        first = apply_gaussian_rgb_pixel_noise(frame, 8.0, np.random.default_rng(31))
        second = apply_gaussian_rgb_pixel_noise(frame, 8.0, np.random.default_rng(31))

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, frame))
        self.assertFalse(np.array_equal(first[:, :, 0], first[:, :, 1]))

    def test_rgb_pixel_noise_preserves_alpha_and_zero_sigma_is_identity(self) -> None:
        frame = np.full((10, 12, 4), 90, dtype=np.uint8)
        frame[:, :, 3] = 203
        noisy = apply_gaussian_rgb_pixel_noise(frame, 6.0, np.random.default_rng(4))
        identity = apply_gaussian_rgb_pixel_noise(frame, 0.0, np.random.default_rng(4))

        np.testing.assert_array_equal(noisy[:, :, 3], frame[:, :, 3])
        np.testing.assert_array_equal(identity, frame)

    def test_rgb_pixel_noise_rejects_out_of_range_sigma(self) -> None:
        with self.assertRaisesRegex(ValueError, "between"):
            _coerce_rgb_pixel_noise_std(-0.1)
        with self.assertRaisesRegex(ValueError, "between"):
            _coerce_rgb_pixel_noise_std(65.0)

    def test_training_image_is_unannotated_rgb_png(self) -> None:
        frame = np.zeros((6, 8, 4), dtype=np.uint8)
        frame[:, :, :3] = (12, 34, 56)
        frame[:, :, 3] = 17
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "training.png"
            save_training_capture_image(frame, output_path)
            with Image.open(output_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (8, 6))
                self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))


if __name__ == "__main__":
    unittest.main()
