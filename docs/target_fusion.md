# Bounding-box target fusion

The commands below run from the repository root and assume Isaac Sim's Python
launcher has been configured as:

```bash
export ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh
```

`scripts/cycle_ground_backgrounds.py` captures the mannequin from four fixed
USD perspective cameras. Each camera produces a synchronized
`bounding_box_2d_tight` annotation. The floating-point box center is
back-projected through the camera's actual USD intrinsics and pose to form one
world-space bearing ray. If the mannequin is outside one camera's view, that
camera remains part of the synchronized capture while rays are still built
from the visible cameras. A scene is marked valid only when all four boxes are
valid and the ray geometry passes the rank, conditioning, and forward-distance
checks.

Schema v2 is the only supported capture-record format:

```bash
"$ISAAC_SIM_PYTHON" scripts/cycle_ground_backgrounds.py \
  --headless --schema-v2-output outputs/target_fusion_bbox_v2.jsonl
```

Each synchronized camera view is also saved as an annotated PNG under
`outputs/target_fusion_bbox_v2_images/` by default. Override that location with
`--image-output-dir PATH`; the corresponding path is recorded in each schema-v2
camera observation.

The same synchronized step is attached to an Isaac Replicator `BasicWriter`.
Clean RGB frames and Isaac-native tight bbox artifacts are written under
`outputs/sdg_raw/` by default. Override that location with
`--raw-output-dir PATH`. Pass `--sensor-noise` to add independent Gaussian noise
to the requested mannequin X/Y/Z position. The default standard deviation is
`0.02 0.02 0.02` metres and can be changed with
`--position-noise-std STD_X STD_Y STD_Z`. Sampling is reproducible from `--seed`;
the nominal position, sampled offset, and applied position are recorded in each capture.
The same flag applies per-camera resolution noise: a scale is drawn from
`N(1.0, 0.15)`, clamped to `0.5-1.5`, and used as an intermediate resampling
resolution before returning the frame to its configured dimensions. This preserves
bbox/intrinsic geometry while modeling downscaling and upscaling artifacts. Change
the scale standard deviation with `--resolution-noise-std FLOAT`; sampled scales and
intermediate resolutions are included in capture metadata.
Use `--frames N` to generate exactly `N` captures;
backgrounds repeat in stable filename order when `N` exceeds the number of
available PNGs.

The GUI cycle is deliberately ordered as: randomize the background, randomize
the mannequin pose and settle it, fire the synchronized four-camera capture,
pause the timeline for inspection, then clear all transient rays and markers
before the next cycle.

## Deterministic pose controls

The default `--pose-mode random` retains the seeded random placement. Use
`--pose-mode fixed` with `--pose-position X Y Z` to capture one exact
world-space position. `--pose-orientation QX QY QZ QW` is optional and uses
XYZW quaternion order (converted to Isaac's internal WXYZ order); it defaults
to the authored mannequin orientation.

```bash
"$ISAAC_SIM_PYTHON" scripts/cycle_ground_backgrounds.py \
  --pose-mode fixed --pose-position 0 0 0.5 \
  --pose-orientation 0 0 0 1 --settle-mode none --frames 1
```

Use `--pose-mode scenario --pose-scenarios PATH` for a JSON list (or an object
with a `scenarios` list). Each item requires `position: [x, y, z]` and may
include `name`, `orientation: [qx, qy, qz, qw]` in XYZW order, and `background`.
Background references must point to one of the PNGs discovered in
`--backgrounds-dir`.
Without `--frames`, scenario mode captures one scene per item; `--frames N`
repeats the list in order.

`--settle-mode physics` is the default and preserves the normal settle loop.
`--settle-mode none` leaves the timeline paused after applying the requested
transform, which is useful for testing exact edge-case positions and
orientations. Each output record includes `requested_position_world`,
`requested_orientation_xyzw`, `settled_position_world`,
`settled_orientation_xyzw`, the settle mode, and whether physics reported a
stable pose.

## Compare a local YOLO model

YOLO is disabled unless both a model and comparison mode are selected. For
example, compare a locally trained checkpoint using the same synchronized RGB
capture as the ground truth:

```bash
"$ISAAC_SIM_PYTHON" scripts/cycle_ground_backgrounds.py \
  --yolo-model outputs/yolo_training_runs/mannequin_yolo11n_bbox/weights/best.pt \
  --yolo-comparison-mode same-time \
  --headless
```

The available modes are `after-ground-truth` and `same-time`. In these option
names, “ground truth” means the Isaac semantic-bbox observation source. Both
modes produce per-camera YOLO detections, rays, fusion, and comparison metrics
in the schema-v2 record. The former performs YOLO after Isaac-annotation
fusion; the latter runs both computations from the same synchronized RGB
frames. In GUI mode, Isaac-annotation rays are green and YOLO rays are blue.

In GUI `after-ground-truth` mode, the sources are displayed sequentially for
five seconds each: green Isaac-annotation rays first, then blue YOLO rays.
`same-time` displays both sources together. Headless captures still compute
and record the same comparison data without the visual pauses.

`yolo11n.pt` and `yolo26n.pt` are supported aliases for checkpoints at the
repository root. Any explicit local `.pt` path is also accepted; the model
must be an Ultralytics detection checkpoint containing the configured target
label, normally `mannequin`. Additional tuning options are
`--yolo-confidence-threshold`, `--yolo-iou-threshold`, `--yolo-image-size`,
and `--yolo-device`.

Live YOLO capture requires `ultralytics` in the Python environment used by
Isaac Sim. The standalone Isaac Sim 6.0 package already bundles PyTorch
`2.11.0+cu128`; it becomes available after `SimulationApp` starts. Install
Ultralytics without dependencies so pip does not download a second Torch copy:

```bash
"$ISAAC_SIM_PYTHON" -m pip install \
  --no-deps --no-cache-dir "ultralytics==8.4.80"
```

Verify the environment with the same launcher used for capture, for example:

```bash
"$ISAAC_SIM_PYTHON" -c \
  "from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); import torch, ultralytics; print(torch.__version__, ultralytics.__version__); app.close()"
```

Do not add packages from a different system Python's `site-packages` directory.

Summarize the schema-v2 output with:

```bash
python3 scripts/report_target_fusion.py outputs/target_fusion_bbox_v2.jsonl
```

If the JSONL contains YOLO comparison blocks, the report also includes model
and mode coverage, detection rate, inference latency, confidence, bbox IoU,
center/ray errors, fused-position deltas, YOLO target error, and miss reasons.

The raw SDG directory keeps training images separate from visual diagnostics:

```text
sdg_raw/
  manifest.jsonl
  rgb/TargetFusion_Camera_01_rgb_000000.png
  bounding_box_2d_tight/TargetFusion_Camera_01_bounding_box_2d_tight_000000.npy
  bounding_box_2d_tight/TargetFusion_Camera_01_bounding_box_2d_tight_labels_000000.json
  bounding_box_2d_tight/TargetFusion_Camera_01_bounding_box_2d_tight_prim_paths_000000.json
  camera_params/TargetFusion_Camera_01_camera_params_000000.json
```

The four camera render products are attached to one `BasicWriter`, with RGB,
tight 2D boxes, and camera parameters enabled. The schema-v2 observation
stores both `image_path` (annotated preview) and the corresponding raw paths
(`raw_image_path`, `raw_bbox_path`, and `raw_camera_params_path`). The raw
manifest records the same pairing for later YOLO export.

The schema-v2 `capture.pose` block records the requested and read-back pose,
the selected pose mode/scenario, the settle mode, and the physics-settled
flag.

## Schema-v2 record contract

Every JSONL line is one synchronized capture with this top-level structure:

```text
schema_version: 2
capture
camera_observations[4]
inferred_rays[4]
fusion
ground_truth_evaluation
yolo                         # present only when comparison is enabled
```

`camera_observations` always contains all four configured cameras. A missed,
clipped, malformed, or overly occluded observation remains in its camera slot
with `valid: false` and a reason. `inferred_rays` has the same camera ordering;
an observation that cannot produce a ray has `ray: null` and its rejection
reason. This fixed cardinality prevents missing cameras from silently changing
the meaning of array positions.

The `fusion` block contains only estimator output and geometry diagnostics:
the fused position, validity and rejection reason, RMS residual, matrix rank,
condition number, pairwise angles, and per-ray forward/residual values. The
true mannequin bounds center is not an estimator input. It appears only in
`ground_truth_evaluation`, where it is used to measure the completed estimate.

When enabled, `yolo` records model metadata, ordered inference results, YOLO
observations and rays, detector fusion, per-camera comparisons, and aggregate
metrics. It does not replace the primary Isaac-annotation observations.

## Export a YOLO dataset

After capture, export the clean RGB frames and selected mannequin boxes with:

```bash
python3 scripts/export_yolo_dataset.py \
  --schema-v2-output outputs/target_fusion_bbox_v2.jsonl \
  --output-dir outputs/yolo_mannequin \
  --overwrite
```

The exporter writes the standard layout:

```text
yolo_mannequin/
  data.yaml
  manifest.jsonl
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Each label row is `class_id x_center y_center width height`, normalized to
`[0, 1]`, with class `0` equal to `mannequin`. Empty camera views still get a
copied image and an empty label file. Positive clipped boxes are clamped to
the image bounds and retained; the original and clamped pixel boxes, export
status, and source validity are recorded in `manifest.jsonl`.

Splits are assigned from `capture_id` as a group, so the four synchronized
camera views from one capture always remain in the same split. The default
split probabilities are train/val/test = 70/20/10. Use `--append` to add new
captures while preserving existing class and group assignments.

Validate the exported dataset and generate visual previews with:

```bash
python3 scripts/validate_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin \
  --report-path outputs/yolo_mannequin/validation.json

python3 scripts/visualize_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin \
  --split train --limit 16 \
  --output-dir outputs/yolo_mannequin/previews
```

Validation reports missing pairs, malformed or out-of-range rows, manifest
drift, and any capture group assigned to more than one split. Empty `val` or
`test` splits are warnings during smoke tests and can be made failures with
`--strict`.

## Train a local YOLO model

Training reuses the same YAML parsing, image discovery, image checks, label
parsing, and bbox validation code as the standalone validator. Its policy is
adapted for ordinary YOLO datasets: `train` and `val` (or `valid`) must be
nonempty, at least one labeled object must exist, and `test` and
`manifest.jsonl` are optional.

Audit without training:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/yolo_mannequin/data.yaml \
  --check-only
```

Start GPU-first training:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/yolo_mannequin/data.yaml
```

Automatic device selection stops if CUDA is unavailable. Select
`--device cpu` or pass `--allow-cpu` only when CPU training is intentional.
Use `--eval-test` for an explicit post-training test-split evaluation and
`--archive` to zip the completed run.

The architecture-only stress suite exercises a 192-view synthetic fixture,
compares two fresh exports for deterministic output, verifies append/resume
split preservation, and rejects duplicate source captures:

```bash
python3 -B -m unittest discover -s tests -v
```

## Interpreting the estimate

The estimate is a visual-center estimate, not a guaranteed geometric center
of the mannequin. Silhouette and bounding-box centers can differ between
views, so valid rays can retain a nonzero RMS residual and position error.
Boxes that are clipped, missing, zero-area, malformed, or too occluded are
rejected because their centers are systematically displaced. Rejected cameras
remain in the record, while rays from the other valid cameras are retained for
diagnostics and downstream use. Poor camera geometry is reported through
minimum pairwise ray angle, matrix rank, and condition number rather than
silently accepted.

Ground-truth bounds are used only for camera setup validation and the separate
`ground_truth_evaluation` output block. The observation interface keeps
semantic boxes interchangeable with future detector outputs.

Noisy-pixel and dropped-observation tests should be used to quantify the gap
between perfect Isaac truth boxes and a real detector. Real-camera deployment
also requires lens undistortion, timestamp synchronization, target association,
and monitoring for extrinsic-camera drift.
