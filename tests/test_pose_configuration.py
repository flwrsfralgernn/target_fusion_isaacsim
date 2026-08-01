import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cycle_ground_backgrounds import (
    _coerce_pose_orientation,
    _coerce_pose_position,
    load_pose_scenarios,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
    validate_pose_options,
)


def pose_args(**overrides):
    values = {
        "pose_mode": "random",
        "pose_position": None,
        "pose_orientation": None,
        "pose_scenarios": None,
        "settle_mode": "physics",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PoseConfigurationTests(unittest.TestCase):
    def test_random_mode_remains_the_default_configuration(self) -> None:
        validate_pose_options(pose_args())

    def test_fixed_mode_requires_a_position(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --pose-position"):
            validate_pose_options(pose_args(pose_mode="fixed"))

    def test_fixed_mode_accepts_position_and_optional_orientation(self) -> None:
        validate_pose_options(
            pose_args(
                pose_mode="fixed",
                pose_position=[1.0, -2.0, 0.5],
                pose_orientation=[0.0, 0.0, 2.0, 0.0],
                settle_mode="none",
            )
        )

    def test_pose_arguments_are_rejected_in_the_wrong_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --pose-mode fixed"):
            validate_pose_options(pose_args(pose_position=[0.0, 0.0, 0.5]))
        with self.assertRaisesRegex(ValueError, "requires --pose-mode scenario"):
            validate_pose_options(pose_args(pose_scenarios=Path("scenarios.json")))

    def test_scenario_mode_requires_a_scenario_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --pose-scenarios"):
            validate_pose_options(pose_args(pose_mode="scenario"))

    def test_position_and_orientation_are_finite_and_normalized(self) -> None:
        self.assertEqual(_coerce_pose_position([1, 2, 3]), (1.0, 2.0, 3.0))
        np.testing.assert_allclose(
            _coerce_pose_orientation([0.0, 0.0, 2.0, 0.0]),
            [0.0, 0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(_xyzw_to_wxyz([0.1, 0.2, 0.3, 0.4]), [0.4, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(_wxyz_to_xyzw([0.4, 0.1, 0.2, 0.3]), [0.1, 0.2, 0.3, 0.4])
        with self.assertRaisesRegex(ValueError, "finite"):
            _coerce_pose_position([0.0, np.inf, 0.0])
        with self.assertRaisesRegex(ValueError, "nonzero length"):
            _coerce_pose_orientation([0.0, 0.0, 0.0, 0.0])

    def test_scenarios_load_positions_orientations_and_backgrounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backgrounds_dir = root / "backgrounds"
            backgrounds_dir.mkdir()
            background_a = backgrounds_dir / "a.png"
            background_b = backgrounds_dir / "b.png"
            background_a.write_bytes(b"not-a-real-png")
            background_b.write_bytes(b"not-a-real-png")
            scenario_path = root / "scenarios.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "scenarios": [
                            {
                                "name": "low-left",
                                "position": [-1.0, 2.0, 0.5],
                                "orientation": [0.0, 0.0, 2.0, 0.0],
                                "background": "b.png",
                            },
                            {"position": [3.0, 4.0, 0.5]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            scenarios = load_pose_scenarios(
                scenario_path,
                backgrounds_dir=backgrounds_dir,
                backgrounds=[background_a, background_b],
            )

        self.assertEqual(
            [scenario["name"] for scenario in scenarios],
            ["low-left", "scenario_0001"],
        )
        self.assertEqual(scenarios[0]["position"], (-1.0, 2.0, 0.5))
        np.testing.assert_allclose(scenarios[0]["orientation"], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(scenarios[0]["background_path"].name, "b.png")
        self.assertIsNone(scenarios[1]["background_path"])
        self.assertIsNone(scenarios[1]["orientation"])

    def test_scenario_background_must_be_one_of_discovered_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backgrounds_dir = root / "backgrounds"
            backgrounds_dir.mkdir()
            (backgrounds_dir / "known.png").write_bytes(b"not-a-real-png")
            scenario_path = root / "scenarios.json"
            scenario_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "bad-background",
                            "position": [0.0, 0.0, 0.5],
                            "background": "missing.png",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must name a PNG"):
                load_pose_scenarios(
                    scenario_path,
                    backgrounds_dir=backgrounds_dir,
                    backgrounds=[backgrounds_dir / "known.png"],
                )


if __name__ == "__main__":
    unittest.main()
