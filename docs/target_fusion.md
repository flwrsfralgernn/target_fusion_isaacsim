# Bounding-box target fusion

`scripts/cycle_ground_backgrounds.py` captures the mannequin from four fixed
USD perspective cameras. Each camera produces a synchronized
`bounding_box_2d_tight` annotation. The floating-point box center is
back-projected through the camera's actual USD intrinsics and pose to form one
world-space bearing ray. A scene is valid only when all four boxes are valid
and the ray geometry passes the rank, conditioning, and forward-distance
checks.

The default output is schema v2:

```bash
/home/rog/Downloads/isaacsim/python.sh scripts/cycle_ground_backgrounds.py \
  --headless --schema-v2-output outputs/target_fusion_bbox_v2.jsonl
```

Each synchronized camera view is also saved as an annotated PNG under
`outputs/target_fusion_bbox_v2_images/` by default. Override that location with
`--image-output-dir PATH`; the corresponding path is recorded in each schema-v2
camera observation.

The GUI cycle is deliberately ordered as: randomize the background, randomize
the mannequin pose and settle it, fire the synchronized four-camera capture,
pause the timeline for inspection, then clear all transient rays and markers
before the next cycle.

The previous exact-coordinate schema-v1 file is not overwritten by default.
Pass `--fusion-output PATH` only when a compatibility schema-v1 record is also
needed. Summarize either format with:

```bash
python3 scripts/report_target_fusion.py outputs/target_fusion_bbox_v2.jsonl
```

## Interpreting the estimate

The estimate is a visual-center estimate, not a guaranteed geometric center
of the mannequin. Silhouette and bounding-box centers can differ between
views, so valid rays can retain a nonzero RMS residual and position error.
Boxes that are clipped, missing, zero-area, malformed, or too occluded are
rejected because their centers are systematically displaced. Poor camera
geometry is reported through minimum pairwise ray angle, matrix rank, and
condition number rather than silently accepted.

Ground-truth bounds are used only for camera setup validation and the separate
`ground_truth_evaluation` output block. The observation interface keeps
semantic boxes interchangeable with future detector outputs.

Noisy-pixel and dropped-observation tests should be used to quantify the gap
between perfect Isaac truth boxes and a real detector. Real-camera deployment
also requires lens undistortion, timestamp synchronization, target association,
and monitoring for extrinsic-camera drift.
