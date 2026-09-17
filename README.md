# PrecisionReach-GS

Official project page and selected research code for **Robot-Aligned 3D Gaussian Splatting for Precise Reaching with Wrist-Camera VLA Policies**.

[Project page](https://001-wang.github.io/PrecisionReach-GS/) · [Paper PDF](assets/precisionreach-gs-paper.pdf)

PrecisionReach-GS is a lightweight real-to-sim-to-real pipeline for data-efficient VLA adaptation. It reconstructs a real workspace with 3D Gaussian Splatting, aligns the representation with the robot frame, and generates synchronized wrist-camera observations and smooth reaching actions.

## Released components

This repository contains the compact subset needed for the reconstruction-to-data-generation workflow:

```text
tools/
  align_colmap_to_robot.py          # Weighted COLMAP-to-robot alignment
  prepare_fastgs_robot_dataset.py   # Robot-pose-aware FastGS dataset preparation
  camera_path_to_ee_poses.py        # Camera path to end-effector supervision
  add_gripper_pose_to_camera_path.py
gs_dual_viewer/
  index.html
  main.js                            # GS viewer, reachability, path and recording tools
  assets/fr3_visual_wireframe.json
```

Large datasets, reconstructed splats, robot recordings, trained policies, and experiment outputs are intentionally excluded.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

COLMAP is also required to build the sparse reconstruction used by the alignment tool.

## 1. Prepare the reconstruction

The expected dataset layout is:

```text
dataset/
  image/
    rgb_000000.png
    rgb_000001.png
  frame_poses.jsonl
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

Each `frame_poses.jsonl` record must include an image name in `rgb_path` and a 4×4 `base_T_camera` matrix.

## 2. Align COLMAP with the robot frame

```bash
python tools/align_colmap_to_robot.py \
  --colmap-model /path/to/dataset/sparse/0 \
  --robot-poses /path/to/dataset/frame_poses.jsonl \
  --output-model ./results/aligned \
  --mode similarity_plus_camera_rotation \
  --anchor-frames 66,69 \
  --anchor-weight 50
```

The main viewer inputs are `alignment.json` and `cameras_colmap_world.json` in the output directory.

## 3. Open the GS data-generation viewer

Serve the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765/gs_dual_viewer/index.html?role=ee
```

Load the trained `.splat` or `.ply`, then load the generated `alignment.json`. The viewer supports camera-path editing, FR3 reachability checks, feasible start-pose sampling, and synchronized WebM/end-effector-pose recording. See [the viewer guide](gs_dual_viewer/README.md) for the complete controls.

## Results at a glance

- Up to 1,500 generated trajectories from short real-world captures
- Less than 10 minutes of active human capture for the largest generated dataset
- π₀.₅ precise reaching at 1,500 trajectories: 21/25 screw, 19/25 GPU, and 20/25 RAM
- Generated trajectory RMS jerk: 0.2353 m/s³, compared with 3.2657 m/s³ for teleoperation

## Citation

```bibtex
@unpublished{wang_precisionreach_gs,
  title  = {Robot-Aligned 3D Gaussian Splatting for Precise Reaching
            with Wrist-Camera VLA Policies},
  author = {Wang, Zuoxu},
  note   = {Manuscript under review},
  url    = {https://github.com/001-Wang/PrecisionReach-GS}
}
```
