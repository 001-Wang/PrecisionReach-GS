# GS Dual Viewer

Two-window prototype:

- EE camera viewer: move the robot-mounted camera view.
- Side camera viewer: keep a fixed observer view and watch the EE camera marker move live.
- The side viewer shows camera frustums/axes for inspection. The EE viewer stays as the camera rendering view.
 
Serve the repository root:
```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open the EE and optional side views:

```text
http://127.0.0.1:8765/gs_dual_viewer/index.html?role=ee
http://127.0.0.1:8765/gs_dual_viewer/index.html?role=side
```

Load the trained `.splat` or `.ply` and the `alignment.json` from the interface. You can also pass repository-relative files through `cameras=`, `alignment=`, and `url=` query parameters.

How it works:

- The EE viewer broadcasts its current camera pose using `BroadcastChannel`.
- The side viewer receives that pose and draws a live magenta camera frustum / RGB axis in the GS scene.
- Add the same `channel=...` URL parameter to an EE/side pair when running multiple pairs in parallel.
- The side camera itself is independent and fixed wherever you move it.
- The viewer still runs a first-pass numerical position IK check with FR3 joint limits for reachability/status, but the live robot skeleton is hidden from the side overlay.
- `IKerr` is not a singularity guarantee. `IKerr > 50mm` means fail/do not execute. `manip ... SINGULAR` means the solved posture is near a singular configuration.
- `Check Reachability` writes the current target gripper pose, solved `q1..q7`, joint-limit margins, and pass/warn/fail label into the JSON panel.
- The reachability result stays pinned in the JSON panel until another pose action updates it.
- If no live EE pose is available, the side viewer falls back to a static default FR3 pose.
- The live EE overlay draws the camera frustum/axes from the broadcast EE camera pose.
- It can optionally load a compact FR3 visual wireframe from `assets/fr3_visual_wireframe.json`.
- The detailed solid CAD surface was removed from this viewer because it can cover the Gaussian splat background.
- Use `Show Static Wire` only when you want to inspect the detailed CAD outline; it is a reference pose, not the live constrained robot.
- `Record WebM + EE Poses` downloads the `.webm` plus matching `_ee_poses.json` and `_ee_poses.csv` files. Each pose row has `frame_index`, `video_time_sec`, `timestamp_ms`, wall-clock time, EE/gripper center, and `base_T_gripper`.
- `Build Workspace Sweep` creates a nearest-neighbor sweep through the conservative safe workspace. It keeps the current EE/gripper orientation, rejects candidates outside `safe_inside`, and keeps only poses whose reachability label is `visual_ok`.
- `Record Start + Path` uses exactly the current chosen pose plus every pose from the loaded path JSON, then records the WebM plus EE pose logs.
- `motion` selects the interpolation used by Play, Record, Record Start + Path, and Batch Starts. `Smooth` eases in/out at every waypoint. `Linear` uses the raw segment time, giving a constant interpolation rate between waypoints with no easing.
- Saved path JSON, per-clip batch path JSON, and EE pose logs include `trajectory_mode` so linear and smooth collections can be distinguished later.
- `Output Folder` lets the browser write batch files into a selected folder. Choose the same folder as `camera_path_keyframes.json` if you want all outputs there.
- `Batch Starts` samples visual-ok safe start poses from a deterministic Gaussian cloud centered on the FR3 ready-pose gripper center, with low starts filtered out. A diversity pass spreads the final starts across the cloud. Each clip is one generated start pose followed by every pose from the loaded path JSON. Batch outputs are named in order: `000.webm`, `000_ee_poses.json`, `000_ee_poses.csv`, `000_camera_path_keyframes.json`, then `001...`.

Workflow:

1. Open the side viewer and move to a good fixed observer angle.
2. Open the EE viewer and move the EE camera.
3. The side viewer should show the moving EE camera marker.
4. In the side viewer, press `Y` to record the side-camera video and the live EE pose timeline.
5. In the EE viewer, press `Y` to record the EE-camera video and the same 30 fps pose timeline from the moving camera.

Start plus saved path collection:

1. In the EE viewer, move to a visually good starting orientation.
2. Load your saved 2-frame path JSON with `Load Path`.
3. Click `Record Start + Path` to record 3 keyframes total: chosen start plus the 2 saved path poses.

Linear start plus saved path collection:

1. Choose `Linear` in the `motion` selector.
2. Load the path and set the clip duration as usual.
3. Use `Record Start + Path` for one clip, or `Batch Starts` for a full linear-motion collection.

Batch start collection:

1. In the EE viewer, move to a visually good orientation that the starts should reuse.
2. Load the same 2-frame path JSON with `Load Path`.
3. Click `Output Folder` and choose the folder that contains `camera_path_keyframes.json`.
4. Set the batch count, for example `50`.
5. Click `Batch Starts`. The viewer records one clip per valid start pose, each with 3 keyframes total.

The current IK constrains the gripper position first. It does not yet constrain wrist orientation to match the camera exactly.
