# UAVFORGE UCI target fusion

This repository generates multi-camera mannequin localization data using a
realistic target-fusion pipeline. The estimator uses the mannequin's 2D
bounding-box center from four cameras, back-projects each center into a world
bearing ray, and estimates the visual center from ray convergence.

The source USD is never modified. Camera aiming and semantic labels are
runtime, in-memory changes.

## Run a capture

From the repository root:

```bash
cd /home/rog/Downloads/UAVFORGEUCIsim

/home/rog/Downloads/isaacsim/python.sh \
  scripts/cycle_ground_backgrounds.py \
  --headless
```

For a GUI run, omit `--headless`. The default configuration is:

- four fixed cameras: `/World/Camera_01` through `/World/Camera_04`
- render resolution: `640x480`
- one Replicator subframe
- semantic target label: `mannequin`
- all four valid observations required for a valid scene; available camera
  observations still produce rays when another camera misses the mannequin

Useful options:

```text
--resolution WIDTH HEIGHT
--rt-subframes N
--seed N
--max-occlusion-ratio 0.5
--bbox-border-tolerance-px 0
--schema-v2-output PATH
--image-output-dir PATH
--raw-output-dir PATH
--frames N
--scene-hold-seconds SECONDS
--yolo-model PATH_OR_ALIAS
--yolo-comparison-mode {disabled,after-ground-truth,same-time}
--yolo-confidence-threshold FLOAT
--yolo-iou-threshold FLOAT
--yolo-image-size PIXELS
--yolo-device DEVICE
```

## Capture lifecycle

Each background cycle follows this sequence:

1. Randomize the background material.
2. Randomize the mannequin position and settle physics.
3. Capture synchronized RGB and `bounding_box_2d_tight` views from all four cameras.
4. Resolve semantic IDs, validate boxes, compute floating-point box centers, and fire bearing rays from every camera that sees the mannequin. Cameras that miss it remain in the capture with an invalid observation.
5. Pause the timeline, fuse the rays, and display the valid estimate.
6. Save annotated camera images and JSONL diagnostics.
7. Clear all transient rays and markers before the next cycle.

YOLO comparison is disabled by default. To compare a local detector against
the same synchronized RGB frames, select a checkpoint and a timing mode:

```bash
/home/rog/Downloads/isaacsim/python.sh \
  scripts/cycle_ground_backgrounds.py \
  --yolo-model outputs/yolo_training_runs/mannequin_yolo11n_bbox/weights/best.pt \
  --yolo-comparison-mode same-time \
  --headless
```

The supported aliases `yolo11n.pt` and `yolo26n.pt` resolve from the repository
root; explicit relative checkpoint paths resolve from the repository root as
well. The checkpoint must be an Ultralytics detection model containing the
selected `--target-label` (normally `mannequin`). `after-ground-truth` runs
YOLO after the ground-truth fusion calculation, while `same-time` runs both
from the same synchronized RGB capture. GUI comparison views draw ground-truth
rays in green, YOLO rays in blue, and separate fused-position markers.

Live YOLO capture requires `ultralytics` in Isaac Sim's own Python environment.
The standalone Isaac Sim 6.0 package already bundles PyTorch
`2.11.0+cu128`; it becomes available after `SimulationApp` starts. Install
Ultralytics without dependencies so pip does not download a second Torch copy:

```bash
/home/rog/Downloads/isaacsim/python.sh -m pip install \
  --no-deps --no-cache-dir "ultralytics==8.4.80"
```

Verify with the same launcher used for capture:

```bash
/home/rog/Downloads/isaacsim/python.sh -c \
  "from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); import torch, ultralytics; print(torch.__version__, ultralytics.__version__); app.close()"
```

Do not reuse a different system Python's `site-packages` directory.

The four render products and annotators are created once. Capture uses one
blocking `rep.orchestrator.step(..., pause_timeline=True, wait_for_render=True)`
call, establishing the synchronized multi-camera boundary.

## Outputs

By default:

```text
outputs/target_fusion_bbox_v2.jsonl
outputs/target_fusion_bbox_v2_images/
outputs/sdg_raw/
```

There is one annotated PNG per camera per scene, named like:

```text
scene_0000_camera_01.png
```

Valid images contain the mannequin bbox and center coordinates. Invalid images
contain the rejection reason. Image paths are recorded in the corresponding
schema-v2 camera observation.

`outputs/sdg_raw/` contains clean Isaac BasicWriter RGB frames, tight bbox
arrays, camera parameters, and `manifest.jsonl`. These raw images remain
separate from annotated diagnostic previews.

## Export a YOLO dataset

```bash
python3 scripts/export_yolo_dataset.py \
  --schema-v2-output outputs/target_fusion_bbox_v2.jsonl \
  --output-dir outputs/yolo_mannequin \
  --overwrite
```

The exporter writes `data.yaml`, `train/`, `val/`, and `test/` directories with
normalized YOLO labels. Empty camera views receive empty label files. Clipped
boxes are clamped and retained when they have positive visible area. All four
camera views from one capture are assigned to the same split.

Validate and preview the exported dataset:

```bash
python3 scripts/validate_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin

python3 scripts/visualize_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin \
  --split train --limit 16 \
  --output-dir outputs/yolo_mannequin/previews
```

Architecture-only stress checks use temporary synthetic fixtures and do not
capture Isaac data:

```bash
python3 -B -m unittest discover -s tests -v
```

## Train YOLO locally

The attached Colab workflow is available as a GPU-first local script. It
audits the pre-split normal-bbox dataset, writes a local-path YAML, and then
trains YOLO11. It stops rather than silently falling back to CPU when CUDA is
not available:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/autovalidated_sdg_final/yolo/data.yaml
```

Run the audit without starting training:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/autovalidated_sdg_final/yolo/data.yaml \
  --check-only
```

Useful options include `--model yolo11s.pt`, `--batch 16`, `--cache disk`,
`--resume PATH/last.pt`, `--eval-test`, and `--archive`. Training outputs are
kept under `outputs/yolo_training_runs/` and ignored by Git.

The existing exact-coordinate baseline at
`outputs/target_fusion_ground_truth.jsonl` is preserved by default. To also
write a schema-v1 compatibility record, explicitly provide:

```bash
--fusion-output outputs/compatibility.jsonl
```

## Schema-v2 contents

Each JSONL record contains:

- capture metadata and synchronization settings;
- four camera observations with calibration, pose, bbox, semantic identity,
  clipping/occlusion state, validity, annotated image path, and clean raw
  image/bbox/camera-parameter paths;
- inferred rays from every camera with a valid target observation, even when
  another camera misses the mannequin;
- four ordered inferred-ray entries, with `null` rays and reasons for rejected
  cameras;
- fused position, residual, rank, condition number, pairwise ray angles, and
  per-ray forward/residual diagnostics;
- a separate `ground_truth_evaluation` block containing the world-bounds-center
  comparison.

Ground truth is never passed to ray construction or fusion. It is used only for
camera setup validation and post-fusion evaluation.

When YOLO comparison is enabled, each schema-v2 record additionally contains a
`yolo` block with model metadata, one inference result per camera, the YOLO
observations/rays/fusion, per-camera bbox/ray comparisons, and aggregate metrics
such as IoU, center error, ray-angle error, and fused-position delta.

## Diagnostics report

Summarize either schema-v1 or schema-v2 JSONL with:

```bash
python3 scripts/report_target_fusion.py \
  outputs/target_fusion_bbox_v2.jsonl
```

The report includes valid-capture rate, four-camera observation rate, position
error distribution, RMS residual distribution, minimum ray-angle distribution,
condition-number distribution, and invalid-fusion reasons.

For captures made with YOLO comparison enabled, the same report automatically
adds a `yolo` section with model/mode counts, detection and four-camera rates,
inference latency, confidence, bbox IoU, center and ray-angle errors,
fused-position deltas, YOLO position error, and miss/fusion reasons.

A seeded eight-scene bbox capture previously produced:

- 6/8 valid four-camera fusions (`75%`);
- mean position error: `0.117 m`;
- maximum position error: `0.258 m`;
- mean RMS residual: `0.109 m`;
- mean minimum ray angle: `56.7°`;
- mean condition number: `1.31`.

## Tests

Run the pure-Python test suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The tests cover camera intrinsics and USD coordinate conventions, rotated and
translated cameras, bbox selection and duplicate union, semantic lookup,
clipping and occlusion rejection, noisy pixels, dropped detections, invalid
fusion geometry, schema serialization, metric reporting, and annotated image
writing.

## Known limitations

This is a visual-center estimate. Different views can have different silhouette
centers, so valid rays do not necessarily intersect exactly. Clipped, heavily
occluded, malformed, missing, or zero-area boxes are rejected because their
centers are systematically biased. Rank, angle, and condition diagnostics must
be monitored rather than trusting unstable convergence.

The observation interface is intentionally detector-independent: Isaac truth
boxes can later be replaced by detector boxes without changing calibration,
ray construction, or fusion. Real-camera deployment additionally requires lens
undistortion, timestamp synchronization, target association, and extrinsic-drift
monitoring.

Isaac Sim capture requires a functioning NVIDIA/Vulkan/NVML environment. If
headless startup reports that no CUDA/NVIDIA device is available, restore the
driver/runtime before running the capture command.
