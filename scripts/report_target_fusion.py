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


def summarize_yolo_records(records) -> dict | None:
    """Summarize optional YOLO comparison blocks embedded in schema-v2 records."""
    yolo_records = [record for record in records if isinstance(record.get("yolo"), dict)]
    if not yolo_records:
        return None

    inference_results = []
    cameras = []
    comparison_metrics = []
    yolo_fusions = []
    yolo_evaluations = []
    valid_camera_counts = []
    model_references = Counter()
    comparison_modes = Counter()
    inference_miss_reasons = Counter()
    invalid_fusion_reasons = Counter()

    for record in yolo_records:
        yolo = record["yolo"]
        model = yolo.get("model")
        if isinstance(model, dict):
            reference = model.get("reference") or model.get("path")
            if reference is not None:
                model_references[str(reference)] += 1
        mode = yolo.get("comparison_mode")
        if mode is not None:
            comparison_modes[str(mode)] += 1

        for inference in yolo.get("inference", []):
            if not isinstance(inference, dict):
                continue
            inference_results.append(inference)
            if not inference.get("valid") and inference.get("reason"):
                inference_miss_reasons[str(inference["reason"])] += 1

        comparison = yolo.get("comparison")
        if not isinstance(comparison, dict):
            continue
        cameras.extend(
            camera for camera in comparison.get("cameras", []) if isinstance(camera, dict)
        )
        metrics = comparison.get("metrics")
        if isinstance(metrics, dict):
            comparison_metrics.append(metrics)

        yolo_source = comparison.get("yolo")
        if not isinstance(yolo_source, dict):
            continue
        if yolo_source.get("valid_camera_count") is not None:
            valid_camera_counts.append(int(yolo_source["valid_camera_count"]))
        fusion = yolo_source.get("fusion")
        if isinstance(fusion, dict):
            yolo_fusions.append(fusion)
            if not fusion.get("valid"):
                invalid_fusion_reasons[str(fusion.get("reason") or "unspecified")] += 1
        evaluation = yolo_source.get("evaluation")
        if isinstance(evaluation, dict):
            yolo_evaluations.append(evaluation)

    valid_yolo_fusions = [fusion for fusion in yolo_fusions if bool(fusion.get("valid"))]
    valid_detections = [result for result in inference_results if result.get("valid")]

    def metric_values(name: str) -> list:
        return [metrics.get(name) for metrics in comparison_metrics]

    return {
        "capture_count": len(yolo_records),
        "capture_rate": len(yolo_records) / len(records) if records else 0.0,
        "valid_fusion_count": len(valid_yolo_fusions),
        "valid_fusion_rate": (
            len(valid_yolo_fusions) / len(yolo_records) if yolo_records else 0.0
        ),
        "four_camera_detection_count": sum(count == 4 for count in valid_camera_counts),
        "four_camera_detection_rate": (
            sum(count == 4 for count in valid_camera_counts) / len(yolo_records)
            if yolo_records
            else 0.0
        ),
        "camera_count": len(cameras),
        "valid_detection_count": len(valid_detections),
        "detection_rate": (
            len(valid_detections) / len(inference_results) if inference_results else 0.0
        ),
        "model_references": dict(model_references),
        "comparison_modes": dict(comparison_modes),
        "inference_time_ms": _stats(
            [result.get("inference_time_ms") for result in inference_results]
        ),
        "confidence": _stats(
            [
                result.get("detection", {}).get("confidence")
                for result in valid_detections
                if isinstance(result.get("detection"), dict)
            ]
        ),
        "bbox_iou": _stats([camera.get("bbox_iou") for camera in cameras]),
        "center_error_px": _stats([camera.get("center_error_px") for camera in cameras]),
        "center_error_normalized": _stats(
            [camera.get("center_error_normalized") for camera in cameras]
        ),
        "ray_angle_deg": _stats([camera.get("ray_angle_deg") for camera in cameras]),
        "fused_position_delta_m": _stats(
            metric_values("fused_position_delta_m")
        ),
        "rms_residual_m": _stats(
            [fusion.get("rms_residual_m") for fusion in valid_yolo_fusions]
        ),
        "position_error_m": _stats(
            [evaluation.get("error_m") for evaluation in yolo_evaluations]
        ),
        "invalid_fusion_reasons": dict(invalid_fusion_reasons),
        "inference_miss_reasons": dict(inference_miss_reasons),
    }


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
    summary = {
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
    yolo_summary = summarize_yolo_records(records)
    if yolo_summary is not None:
        summary["yolo"] = yolo_summary
    return summary


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
