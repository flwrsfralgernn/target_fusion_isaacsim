# UAVFORGE UCI target fusion

This repository generates multi-camera mannequin localization data using a
realistic target-fusion pipeline. The estimator uses the mannequin's 2D
bounding-box center from four cameras, back-projects each center into a world
bearing ray, and estimates the visual center from ray convergence.

The source USD is never modified. Camera aiming and semantic labels are
runtime, in-memory changes.

For implementation details and guidance on interpreting the estimator, see
[the target-fusion guide](docs/target_fusion.md).

## Environments

Capture must run through Isaac Sim's Python launcher. Set its location once
for the examples below:

```bash
export ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh
```

Dataset export, validation, reporting, previews, and unit tests run with a
regular Python 3 environment containing NumPy, Pillow, and PyYAML. Local
training additionally requires Ultralytics and a suitable PyTorch build.

## Run a capture

From the repository root:

```bash
"$ISAAC_SIM_PYTHON" \
  scripts/cycle_ground_backgrounds.py \
  --headless
```

To run the complete 2,000-episode workflow—capture, export, validation,
preview generation, and YOLO training—with the tuned sensor-noise defaults:

```bash
./scripts/run_full_pipeline.sh
```

The wrapper uses `/home/rog/Downloads/isaacsim/python.sh` by default. Set
`ISAAC_SIM_PYTHON` before running it if Isaac Sim is installed elsewhere.

For a GUI run, omit `--headless`. The default configuration is:

- four fixed cameras: `/World/Camera_01` through `/World/Camera_04`
- render resolution: `640x480`
- one Replicator subframe
- semantic target label: `mannequin`
- all four valid observations required for a valid scene; available camera
  observations still produce rays when another camera misses the mannequin

Use `"$ISAAC_SIM_PYTHON" scripts/cycle_ground_backgrounds.py --help` for the
complete CLI. Common options are:

```text
--resolution WIDTH HEIGHT
--rt-subframes N
--sensor-noise
--position-noise-std STD_X STD_Y STD_Z
--resolution-noise-std FLOAT
--brightness-noise-std FLOAT
--exposure-noise-std STOPS
--color-temperature-noise-std KELVIN
--rgb-pixel-noise-std PIXEL_VALUE
--seed N
--pose-mode {random,fixed,scenario}
--pose-position X Y Z
--pose-orientation QX QY QZ QW
--pose-scenarios PATH
--settle-mode {physics,none}
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
2. Select the mannequin pose (random, fixed, or scenario) and apply the chosen settling mode.
3. Capture synchronized RGB and `bounding_box_2d_tight` views from all four cameras. Use
   `--sensor-noise` to add independent, seeded Gaussian noise to the requested mannequin
   position. The default XYZ standard deviation is 1 cm; override it with
   `--position-noise-std STD_X STD_Y STD_Z` (metres). It also samples a per-camera
   resolution scale from a Gaussian centered at 1.0 with standard deviation 0.10,
   clamped to 0.75-1.25. Each frame is resized through the sampled resolution and back
   to its configured size so bounding boxes and calibration remain aligned. Override
   the scale spread with `--resolution-noise-std FLOAT`. Per-camera photometric noise
   adds brightness offsets (`sigma=0.025`), exposure changes (`sigma=0.15` stops), and
   color-temperature changes around 6500 K (`sigma=300 K`). Configure these with
   `--brightness-noise-std`, `--exposure-noise-std`, and
   `--color-temperature-noise-std`. Finally, independent Gaussian noise is added
   to every RGB channel with an 8-bit standard deviation of 5 by default; tune it
   with `--rgb-pixel-noise-std PIXEL_VALUE`.
4. Resolve semantic IDs, validate boxes, compute floating-point box centers, and fire bearing rays from every camera that sees the mannequin. Cameras that miss it remain in the capture with an invalid observation.
5. Pause the timeline, fuse the rays, and display the valid estimate.
6. Save annotated camera images and JSONL diagnostics.
7. Clear all transient rays and markers before the next cycle.

Random placement remains the default. For a repeatable edge case, use a fixed
world-space pose and keep the timeline paused while capturing:

```bash
"$ISAAC_SIM_PYTHON" \
  scripts/cycle_ground_backgrounds.py \
  --pose-mode fixed \
  --pose-position 0.0 0.0 0.5 \
  --pose-orientation 0.0 0.0 0.0 1.0 \
  --settle-mode none \
  --frames 1
```

For multiple deterministic edge cases, use a JSON scenario file. Each item
requires `position: [x, y, z]`; `orientation: [qx, qy, qz, qw]` (the CLI uses
XYZW order) and a background filename are optional. A scenario without a
background uses the normal stable background cycle.

```json
[
  {
    "name": "left-edge",
    "position": [-2.0, 0.0, 0.5],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "background": "01_aerial_grass_rock.png"
  },
  {"name": "right-edge", "position": [2.0, 0.0, 0.5]}
]
```

Run it with:

```bash
"$ISAAC_SIM_PYTHON" \
  scripts/cycle_ground_backgrounds.py \
  --pose-mode scenario \
  --pose-scenarios scenarios.json \
  --settle-mode none
```

With scenario mode and no `--frames`, one capture is generated per scenario;
with `--frames`, scenarios repeat in file order. `--settle-mode physics`
preserves the original physics-settling behavior. Every capture records the
requested and read-back settled pose in the schema-v2 capture and raw manifest.

YOLO comparison is disabled by default. To compare a local detector against
the same synchronized RGB frames, select a checkpoint and a timing mode:

```bash
"$ISAAC_SIM_PYTHON" \
  scripts/cycle_ground_backgrounds.py \
  --yolo-model outputs/yolo_training_runs/mannequin_yolo11n_bbox/weights/best.pt \
  --yolo-comparison-mode same-time \
  --headless
```

The supported aliases `yolo11n.pt` and `yolo26n.pt` resolve from the repository
root; explicit relative checkpoint paths resolve from the repository root as
well. The checkpoint must be an Ultralytics detection model containing the
selected `--target-label` (normally `mannequin`). Here, “ground truth” means
the Isaac semantic-bbox observation source, not an older output schema.
`after-ground-truth` runs YOLO after the Isaac-annotation fusion calculation,
while `same-time` runs both from the same synchronized RGB capture. GUI
comparison views draw Isaac-annotation rays in green, YOLO rays in blue, and
separate fused-position markers.
In GUI `after-ground-truth` mode, the sources are shown separately for five
seconds each: green Isaac-annotation rays first, then blue YOLO rays.
`same-time` shows both sources together.

Live YOLO capture requires `ultralytics` in Isaac Sim's own Python environment.
The standalone Isaac Sim 6.0 package already bundles PyTorch
`2.11.0+cu128`; it becomes available after `SimulationApp` starts. Install
Ultralytics without dependencies so pip does not download a second Torch copy:

```bash
"$ISAAC_SIM_PYTHON" -m pip install \
  --no-deps --no-cache-dir "ultralytics==8.4.80"
```

Verify with the same launcher used for capture:

```bash
"$ISAAC_SIM_PYTHON" -c \
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

`outputs/target_fusion_bbox_v2_images/training/` contains unannotated frames
after resolution, photometric, and RGB pixel noise. The YOLO exporter prefers
these training frames. `outputs/sdg_raw/` contains clean Isaac BasicWriter RGB frames, tight bbox
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
camera views from one capture are assigned to the same split. It uses each
observation's noise-processed `training_image_path`, falling back to the clean
`raw_image_path` for older captures.

Validate and preview the exported dataset:

```bash
python3 scripts/validate_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin

python3 scripts/visualize_yolo_dataset.py \
  --dataset-dir outputs/yolo_mannequin \
  --split train --limit 16 \
  --output-dir outputs/yolo_mannequin/previews
```

Standalone validation requires the exported `data.yaml`, train/val/test
layout, and `manifest.jsonl`. Empty splits are warnings unless `--strict` is
used.

Architecture-only stress checks use temporary synthetic fixtures and do not
capture Isaac data:

```bash
python3 -B -m unittest discover -s tests -v
```

## Train YOLO locally

The GPU-first local trainer runs the same image, label, YAML, and bbox checks
as the standalone validator, then writes a local-path YAML and trains YOLO11.
For training, `train` and `val` must be nonempty, at least one labeled object
must exist, and `test` and `manifest.jsonl` are optional. The trainer stops
rather than silently falling back to CPU when CUDA is unavailable:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/yolo_mannequin/data.yaml
```

Run the audit without starting training:

```bash
python3 scripts/train_yolo_local.py \
  --data outputs/yolo_mannequin/data.yaml \
  --check-only
```

Useful options include `--model yolo11s.pt`, `--batch 16`, `--cache disk`,
`--resume PATH/last.pt`, `--eval-test`, and `--archive`. Training outputs are
kept under `outputs/yolo_training_runs/` and ignored by Git.

## Schema-v2 contents

Schema v2 is the only supported capture-record schema.

Each JSONL record contains:

- capture metadata and synchronization settings;
- requested and read-back deterministic pose metadata when pose controls are used;
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

Summarize schema-v2 JSONL with:

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
