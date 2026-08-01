import random
import sys
import unittest
from unittest.mock import patch

import numpy as np

from scripts.cycle_ground_backgrounds import (
    DEFAULT_POSITION_NOISE_STD_M,
    DEFAULT_RESOLUTION_NOISE_STD,
    _coerce_position_noise_std,
    _coerce_resolution_noise_std,
    add_gaussian_position_noise,
    apply_gaussian_resolution_noise,
    parse_args,
)


class SensorNoiseCliTests(unittest.TestCase):
    def test_sensor_noise_is_disabled_by_default(self) -> None:
        with patch.object(sys, "argv", ["cycle_ground_backgrounds.py"]):
            args = parse_args()

        self.assertFalse(args.sensor_noise)
        self.assertEqual(tuple(args.position_noise_std), DEFAULT_POSITION_NOISE_STD_M)
        self.assertEqual(args.resolution_noise_std, DEFAULT_RESOLUTION_NOISE_STD)

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


if __name__ == "__main__":
    unittest.main()
