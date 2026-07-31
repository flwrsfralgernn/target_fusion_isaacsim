"""Summarize target-fusion JSONL capture quality and geometry diagnostics.

The report accepts both the schema-v1 compatibility records and schema-v2
records. It intentionally reports truth error only from the separate
``ground_truth_evaluation`` block; no ground-truth coordinate is used for
validity or fusion statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


def _finite_numbers(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values) -> dict:
    finite = _finite_numbers(values)
    if not finite:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": mean(finite),
        "median": median(finite),
        "p95": _percentile(finite, 0.95),
        "max": max(finite),
    }


def _valid_camera_count(record: dict) -> int:
    capture = record.get("capture")
    if isinstance(capture, dict) and "valid_camera_count" in capture:
        return int(capture["valid_camera_count"])
    observations = record.get("camera_observations") or record.get("bbox_observations") or []
    return sum(bool(observation.get("valid")) for observation in observations)


def summarize_records(records) -> dict:
    """Return capture-rate, error, residual, angle, and conditioning statistics."""
    records = list(records)
    valid_records = [record for record in records if bool(record.get("fusion", {}).get("valid"))]
    evaluations = [
        record.get("ground_truth_evaluation", {}).get("error_m")
        for record in valid_records
        if isinstance(record.get("ground_truth_evaluation"), dict)
    ]
    residuals = [record.get("fusion", {}).get("rms_residual_m") for record in valid_records]
    minimum_angles = [
        record.get("fusion", {}).get("min_pairwise_angle_deg") for record in valid_records
    ]
    condition_numbers = [
        record.get("fusion", {}).get("condition_number") for record in valid_records
    ]
    invalid_reasons = Counter(
        str(record.get("fusion", {}).get("reason") or "unspecified")
        for record in records
        if not record.get("fusion", {}).get("valid")
    )
    complete_camera_count = sum(_valid_camera_count(record) == 4 for record in records)
    capture_count = len(records)
    return {
        "capture_count": capture_count,
        "valid_capture_count": len(valid_records),
        "valid_capture_rate": (len(valid_records) / capture_count if capture_count else 0.0),
        "four_camera_observation_count": complete_camera_count,
        "four_camera_observation_rate": (
            complete_camera_count / capture_count if capture_count else 0.0
        ),
        "position_error_m": _stats(evaluations),
        "rms_residual_m": _stats(residuals),
        "minimum_ray_angle_deg": _stats(minimum_angles),
        "condition_number": _stats(condition_numbers),
        "invalid_fusion_reasons": dict(invalid_reasons),
    }


def load_jsonl(path: Path) -> list[dict]:
    """Load nonempty JSONL records from *path*."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="schema-v1 or schema-v2 fusion JSONL file")
    args = parser.parse_args()
    path = args.jsonl.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    print(json.dumps(summarize_records(load_jsonl(path)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
