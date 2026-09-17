import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array(
        [
            [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ],
        dtype=float,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = "<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def read_cameras_binary(path: Path) -> list[tuple[int, int, int, int, list[float]]]:
    cameras = []
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            num_params = {
                0: 3,   # SIMPLE_PINHOLE
                1: 4,   # PINHOLE
                2: 4,   # SIMPLE_RADIAL
                3: 5,   # RADIAL
                4: 8,   # OPENCV
                5: 8,   # OPENCV_FISHEYE
                6: 12,  # FULL_OPENCV
                7: 5,   # FOV
                8: 4,   # SIMPLE_RADIAL_FISHEYE
                9: 5,   # RADIAL_FISHEYE
                10: 12, # THIN_PRISM_FISHEYE
            }[model_id]
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras.append((camera_id, model_id, width, height, params))
    return cameras


def read_images_binary(path: Path) -> list[dict]:
    images = []
    with path.open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=float)
            tvec = np.array(props[5:8], dtype=float)
            camera_id = props[8]

            name_bytes = bytearray()
            while True:
                current = fid.read(1)
                if current == b"\x00":
                    break
                name_bytes.extend(current)
            name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            points2d = []
            for _pt in range(num_points2d):
                x, y, point3d_id = read_next_bytes(fid, 24, "ddq")
                points2d.append((x, y, point3d_id))

            Rcw = qvec_to_rotmat(qvec)
            C = -Rcw.T @ tvec
            images.append(
                {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                    "points2d": points2d,
                    "Rcw": Rcw,
                    "C": C,
                }
            )
    return images


def read_points3d_binary(path: Path) -> list[dict]:
    points = []
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3d_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = np.array(read_next_bytes(fid, 24, "ddd"), dtype=float)
            rgb = read_next_bytes(fid, 3, "BBB")
            error = read_next_bytes(fid, 8, "d")[0]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            track = [read_next_bytes(fid, 8, "ii") for _ in range(track_len)]
            points.append(
                {
                    "point3d_id": point3d_id,
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "track": track,
                }
            )
    return points


def load_robot_poses_jsonl(path: Path, pose_key: str = "base_T_camera", image_key: str = "rgb_path") -> dict[str, np.ndarray]:
    poses = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if image_key not in rec or pose_key not in rec:
                raise ValueError(f"{path}:{idx} missing '{image_key}' or '{pose_key}'")
            image_name = Path(rec[image_key]).name
            pose_val = rec[pose_key]
            if isinstance(pose_val, dict) and "matrix" in pose_val:
                T = np.array(pose_val["matrix"], dtype=float)
            else:
                T = np.array(pose_val, dtype=float)
            if T.shape != (4, 4):
                raise ValueError(f"{path}:{idx} pose for {image_name} has shape {T.shape}, expected (4,4)")
            poses[image_name] = T
    return poses


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.ones(len(src), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(src),):
        raise ValueError(f"weights shape {weights.shape} does not match {len(src)} points")
    if np.any(weights <= 0):
        raise ValueError("All weights must be positive")
    weights = weights / np.sum(weights)

    src_mean = np.sum(src * weights[:, None], axis=0)
    dst_mean = np.sum(dst * weights[:, None], axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = dst_centered.T @ (src_centered * weights[:, None])
    U, singular_values, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    variance = np.sum(weights * np.sum(src_centered * src_centered, axis=1))
    scale = np.trace(np.diag(singular_values) @ S) / variance
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_theta = max(-1.0, min(1.0, float((np.trace(R) - 1.0) / 2.0)))
    return math.degrees(math.acos(cos_theta))


def solve_constant_camera_rotation(
    common_names: list[str],
    world_R: np.ndarray,
    colmap_by_name: dict,
    robot_by_name: dict,
    weights_by_name: dict[str, float] | None = None,
) -> np.ndarray:
    accum = np.zeros((3, 3), dtype=float)
    for name in common_names:
        aligned_colmap_Rwc = world_R @ colmap_by_name[name]["Rcw"].T
        robot_Rwc = robot_by_name[name][:3, :3]
        weight = 1.0 if weights_by_name is None else weights_by_name.get(name, 1.0)
        accum += weight * (aligned_colmap_Rwc.T @ robot_Rwc)

    U, _, Vt = np.linalg.svd(accum)
    K = U @ Vt
    if np.linalg.det(K) < 0:
        U[:, -1] *= -1
        K = U @ Vt
    return K


def write_cameras_txt(path: Path, cameras: list[tuple[int, int, int, int, list[float]]]) -> None:
    model_names = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, model_id, width, height, params in cameras:
            params_str = " ".join(f"{p:.17g}" for p in params)
            f.write(f"{camera_id} {model_names[model_id]} {width} {height} {params_str}\n")


def write_viewer_cameras_json(path: Path, cameras: list[tuple[int, int, int, int, list[float]]], images: list[dict]) -> None:
    camera_by_id = {camera_id: (model_id, width, height, params) for camera_id, model_id, width, height, params in cameras}
    viewer_cameras = []
    for idx, image in enumerate(sorted(images, key=lambda img: img["name"])):
        model_id, width, height, params = camera_by_id[image["camera_id"]]
        if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
            fx, fy, cx, cy = params
        else:
            fx = params[0]
            fy = params[1] if len(params) > 1 else params[0]
            cx = params[2] if len(params) > 2 else width / 2.0
            cy = params[3] if len(params) > 3 else height / 2.0

        viewer_cameras.append(
            {
                "id": idx,
                "img_name": image["name"],
                "width": int(width),
                "height": int(height),
                "position": image["C"].tolist(),
                "rotation": image["Rcw"].T.tolist(),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
    path.write_text(json.dumps(viewer_cameras, indent=2), encoding="utf-8")


def write_images_txt(path: Path, images: list[dict], mode: str, world_R: np.ndarray, world_t: np.ndarray, scale: float, K: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: unknown\n")
        for image in images:
            if mode == "similarity_only":
                Rcw_new = image["Rcw"] @ world_R.T
                C_new = scale * (world_R @ image["C"]) + world_t
            elif mode == "similarity_plus_camera_rotation":
                Rwc_new = world_R @ image["Rcw"].T @ K
                Rcw_new = Rwc_new.T
                C_new = scale * (world_R @ image["C"]) + world_t
            else:
                raise ValueError(f"Unknown mode: {mode}")

            t_new = -Rcw_new @ C_new
            qvec_new = rotmat_to_qvec(Rcw_new)
            f.write(
                f"{image['image_id']} "
                f"{qvec_new[0]:.17g} {qvec_new[1]:.17g} {qvec_new[2]:.17g} {qvec_new[3]:.17g} "
                f"{t_new[0]:.17g} {t_new[1]:.17g} {t_new[2]:.17g} "
                f"{image['camera_id']} {image['name']}\n"
            )
            pts = " ".join(f"{x:.17g} {y:.17g} {pid}" for x, y, pid in image["points2d"])
            f.write(pts + "\n")


def write_points3d_txt(path: Path, points: list[dict], world_R: np.ndarray, world_t: np.ndarray, scale: float) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: unknown\n")
        for point in points:
            xyz_new = scale * (world_R @ point["xyz"]) + world_t
            track_str = " ".join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in point["track"])
            f.write(
                f"{point['point3d_id']} "
                f"{xyz_new[0]:.17g} {xyz_new[1]:.17g} {xyz_new[2]:.17g} "
                f"{point['rgb'][0]} {point['rgb'][1]} {point['rgb'][2]} "
                f"{point['error']:.17g} {track_str}\n"
            )


def save_transform_json(
    path: Path,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    common_names: list[str],
    pos_err: np.ndarray,
    ang_err: np.ndarray | None,
    weights_by_name: dict[str, float] | None = None,
) -> None:
    data = {
        "scale_colmap_to_robot": float(scale),
        "rotation_colmap_world_to_robot_world": R.tolist(),
        "translation_colmap_world_to_robot_world": t.tolist(),
        "constant_camera_rotation_colmap_to_robot": K.tolist(),
        "common_frames": len(common_names),
        "frame_names": common_names,
        "position_error_mean_m": float(np.mean(pos_err)),
        "position_error_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "position_error_max_m": float(np.max(pos_err)),
    }
    if weights_by_name:
        data["alignment_weights"] = weights_by_name
        anchors = [name for name, weight in weights_by_name.items() if weight > 1.0]
        data["anchor_frames"] = anchors
        if anchors:
            idx = [common_names.index(name) for name in anchors if name in common_names]
            if idx:
                anchor_err = pos_err[idx]
                data["anchor_position_error_mean_m"] = float(np.mean(anchor_err))
                data["anchor_position_error_max_m"] = float(np.max(anchor_err))
    if ang_err is not None:
        data["orientation_error_mean_deg"] = float(np.mean(ang_err))
        data["orientation_error_max_deg"] = float(np.max(ang_err))
        if weights_by_name:
            anchors = [name for name, weight in weights_by_name.items() if weight > 1.0]
            idx = [common_names.index(name) for name in anchors if name in common_names]
            if idx:
                anchor_ang = ang_err[idx]
                data["anchor_orientation_error_mean_deg"] = float(np.mean(anchor_ang))
                data["anchor_orientation_error_max_deg"] = float(np.max(anchor_ang))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align COLMAP model to robot world using corresponding camera centers.")
    parser.add_argument("--colmap-model", required=True, help="Path to COLMAP sparse model directory containing cameras.bin, images.bin, points3D.bin")
    parser.add_argument("--robot-poses", required=True, help="Path to frame_poses.jsonl")
    parser.add_argument("--output-model", required=True, help="Directory to write aligned COLMAP text model")
    parser.add_argument(
        "--mode",
        choices=["similarity_only", "similarity_plus_camera_rotation"],
        default="similarity_only",
        help="Whether to only align world coordinates, or also absorb a constant camera-frame rotation into image orientations.",
    )
    parser.add_argument("--skip-first-frame", action="store_true", help="Skip the lexicographically first common frame when fitting the transform")
    parser.add_argument(
        "--anchor-frames",
        default="",
        help="Comma-separated important image names or indices, e.g. rgb_000012.png,rgb_000027.png or 12,27.",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=1.0,
        help="Weight applied to anchor frames during similarity fitting. Try 20-100.",
    )
    args = parser.parse_args()

    colmap_model = Path(args.colmap_model)
    output_model = Path(args.output_model)
    output_model.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras_binary(colmap_model / "cameras.bin")
    images = read_images_binary(colmap_model / "images.bin")
    points = read_points3d_binary(colmap_model / "points3D.bin")
    robot_poses = load_robot_poses_jsonl(Path(args.robot_poses))

    colmap_by_name = {img["name"]: img for img in images}
    common_names = sorted(set(colmap_by_name) & set(robot_poses))
    if len(common_names) < 3:
        raise RuntimeError("Need at least 3 common frames to align COLMAP to robot poses.")
    if args.skip_first_frame:
        common_names = common_names[1:]
    if len(common_names) < 3:
        raise RuntimeError("Need at least 3 common frames after skipping.")

    anchor_names = set()
    if args.anchor_frames.strip():
        for token in args.anchor_frames.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                token = f"rgb_{int(token):06d}.png"
            anchor_names.add(Path(token).name)
        missing = sorted(anchor_names - set(common_names))
        if missing:
            raise ValueError(f"Anchor frames are not common COLMAP/robot frames: {missing}")
    if args.anchor_weight <= 0:
        raise ValueError("--anchor-weight must be positive")

    colmap_centers = np.stack([colmap_by_name[name]["C"] for name in common_names], axis=0)
    robot_centers = np.stack([robot_poses[name][:3, 3] for name in common_names], axis=0)
    weights = np.ones(len(common_names), dtype=float)
    for i, name in enumerate(common_names):
        if name in anchor_names:
            weights[i] = args.anchor_weight
    weights_by_name = {name: float(weights[i]) for i, name in enumerate(common_names) if weights[i] != 1.0}
    scale, world_R, world_t = umeyama_similarity(colmap_centers, robot_centers, weights=weights)
    K = solve_constant_camera_rotation(common_names, world_R, colmap_by_name, robot_poses, weights_by_name)

    aligned_centers = (scale * (world_R @ colmap_centers.T)).T + world_t
    pos_err = np.linalg.norm(aligned_centers - robot_centers, axis=1)

    if args.mode == "similarity_plus_camera_rotation":
        ang_err = []
        for name in common_names:
            Rwc_aligned = world_R @ colmap_by_name[name]["Rcw"].T @ K
            Rwc_robot = robot_poses[name][:3, :3]
            ang_err.append(rotation_angle_deg(Rwc_aligned @ Rwc_robot.T))
        ang_err = np.array(ang_err, dtype=float)
    else:
        ang_err = None

    write_cameras_txt(output_model / "cameras.txt", cameras)
    write_images_txt(output_model / "images.txt", images, args.mode, world_R, world_t, scale, K)
    write_points3d_txt(output_model / "points3D.txt", points, world_R, world_t, scale)
    write_viewer_cameras_json(output_model / "cameras_colmap_world.json", cameras, images)

    save_transform_json(output_model / "alignment.json", scale, world_R, world_t, K, common_names, pos_err, ang_err, weights_by_name)

    # Keep binary sidecars if they exist and the caller wants to inspect original rig/frame info nearby.
    for extra_name in ["frames.bin", "rigs.bin"]:
        src = colmap_model / extra_name
        dst = output_model / extra_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    print(f"[INFO] Wrote aligned model to: {output_model}")
    print(f"[INFO] Common frames used: {len(common_names)}")
    print(f"[INFO] Scale (COLMAP -> robot): {scale:.9f}")
    print(f"[INFO] Position RMSE (m): {np.sqrt(np.mean(pos_err ** 2)):.6f}")
    print(f"[INFO] Position mean/max (m): {np.mean(pos_err):.6f} / {np.max(pos_err):.6f}")
    if anchor_names:
        anchor_idx = [i for i, name in enumerate(common_names) if name in anchor_names]
        anchor_err = pos_err[anchor_idx]
        print(f"[INFO] Anchor frames: {', '.join(sorted(anchor_names))}")
        print(f"[INFO] Anchor weight: {args.anchor_weight:g}")
        print(f"[INFO] Anchor position mean/max (m): {np.mean(anchor_err):.6f} / {np.max(anchor_err):.6f}")
    if ang_err is not None:
        print(f"[INFO] Orientation mean/max (deg): {np.mean(ang_err):.6f} / {np.max(ang_err):.6f}")
        if anchor_names:
            anchor_idx = [i for i, name in enumerate(common_names) if name in anchor_names]
            anchor_ang = ang_err[anchor_idx]
            print(f"[INFO] Anchor orientation mean/max (deg): {np.mean(anchor_ang):.6f} / {np.max(anchor_ang):.6f}")
    print(f"[INFO] Saved transform summary to: {output_model / 'alignment.json'}")
    print(f"[INFO] Saved viewer cameras to: {output_model / 'cameras_colmap_world.json'}")


if __name__ == "__main__":
    main()
