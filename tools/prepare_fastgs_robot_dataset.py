import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


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


def parse_images_txt(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_lines = []
    body_started = False
    images = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not body_started and (not stripped or stripped.startswith("#")):
            header_lines.append(line)
            i += 1
            continue

        body_started = True
        if not stripped:
            i += 1
            continue

        meta = stripped.split()
        if len(meta) < 10:
            raise ValueError(f"Malformed image line in {path}: {line}")
        points_line = lines[i + 1] if i + 1 < len(lines) else ""
        qvec = np.array(meta[1:5], dtype=float)
        tvec = np.array(meta[5:8], dtype=float)
        Rcw = qvec_to_rotmat(qvec)
        images.append(
            {
                "image_id": int(meta[0]),
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": int(meta[8]),
                "name": meta[9],
                "points_line": points_line,
                "Rcw": Rcw,
            }
        )
        i += 2
    return header_lines, images


def write_images_txt(path: Path, images: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: unknown\n")
        for image in images:
            qvec = image["qvec"]
            tvec = image["tvec"]
            f.write(
                f"{image['image_id']} "
                f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
                f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
                f"{image['camera_id']} {image['name']}\n"
            )
            f.write(image["points_line"].rstrip() + "\n")


def infer_crop_box(source_image: Path, undistorted_image: Path, max_dx: int = 20, max_dy: int = 20) -> tuple[int, int, int, int]:
    src = np.asarray(Image.open(source_image))
    und = np.asarray(Image.open(undistorted_image))
    h, w = und.shape[:2]

    best = None
    for dy in range(max_dy + 1):
        for dx in range(max_dx + 1):
            crop = src[dy : dy + h, dx : dx + w]
            if crop.shape != und.shape:
                continue
            mad = float(np.mean(np.abs(crop.astype(np.int16) - und.astype(np.int16))))
            cand = (mad, dx, dy)
            if best is None or cand < best:
                best = cand

    if best is None:
        raise RuntimeError("Failed to infer crop box")

    _, dx, dy = best
    return dx, dy, w, h


def copy_and_fill_images(undistorted_dir: Path, source_dir: Path, output_dir: Path, required_names: list[str]) -> tuple[int, int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    available = {p.name for p in undistorted_dir.glob("*.png")}
    source_names = {p.name for p in source_dir.glob("*.png")}
    common_names = sorted(available & source_names)
    if not common_names:
        raise RuntimeError("No common images found between source and undistorted directories")

    crop_box = infer_crop_box(source_dir / common_names[0], undistorted_dir / common_names[0])
    dx, dy, w, h = crop_box

    for name in required_names:
        src_und = undistorted_dir / name
        dst = output_dir / name
        if src_und.exists():
            shutil.copy2(src_und, dst)
            continue

        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source image for {name}: {src}")
        if name not in source_names:
            raise FileNotFoundError(f"{name} is not present in {source_dir}")
        img = Image.open(src)
        crop = img.crop((dx, dy, dx + w, dy + h))
        crop.save(dst)

    return crop_box


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a FastGS dataset in robot reference, filling missing COLMAP views from robot poses.")
    parser.add_argument("--aligned-model", required=True, help="Path to aligned COLMAP text model directory containing cameras.txt/images.txt/points3D.txt")
    parser.add_argument("--robot-poses", required=True, help="Path to frame_poses.jsonl")
    parser.add_argument("--undistorted-images", required=True, help="Path to FastGS/COLMAP undistorted image directory")
    parser.add_argument("--source-images", required=True, help="Path to original RGB images")
    parser.add_argument("--output-root", required=True, help="Output dataset root. Will contain images/ and sparse/0/")
    args = parser.parse_args()

    aligned_model = Path(args.aligned_model)
    output_root = Path(args.output_root)
    sparse_dir = output_root / "sparse" / "0"
    images_dir = output_root / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    header_lines, aligned_images = parse_images_txt(aligned_model / "images.txt")
    robot_poses = load_robot_poses_jsonl(Path(args.robot_poses))
    required_names = sorted(robot_poses)

    camera_ids = {image["camera_id"] for image in aligned_images}
    if len(camera_ids) != 1:
        raise RuntimeError(f"Expected one shared camera in aligned model, got: {sorted(camera_ids)}")
    camera_id = next(iter(camera_ids))

    existing_names = {image["name"] for image in aligned_images}
    next_image_id = max(image["image_id"] for image in aligned_images) + 1

    completed_images = list(aligned_images)
    for name in required_names:
        if name in existing_names:
            continue
        T = robot_poses[name]
        Rwc = T[:3, :3]
        C = T[:3, 3]
        Rcw = Rwc.T
        tvec = -Rcw @ C
        qvec = rotmat_to_qvec(Rcw)
        completed_images.append(
            {
                "image_id": next_image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "points_line": "",
                "Rcw": Rcw,
            }
        )
        next_image_id += 1

    completed_images.sort(key=lambda image: image["name"])

    shutil.copy2(aligned_model / "cameras.txt", sparse_dir / "cameras.txt")
    shutil.copy2(aligned_model / "points3D.txt", sparse_dir / "points3D.txt")
    write_images_txt(sparse_dir / "images.txt", completed_images)
    crop_box = copy_and_fill_images(
        undistorted_dir=Path(args.undistorted_images),
        source_dir=Path(args.source_images),
        output_dir=images_dir,
        required_names=required_names,
    )

    print(f"[INFO] Input aligned model: {aligned_model}")
    print(f"[INFO] Output dataset root: {output_root}")
    print(f"[INFO] Existing aligned images: {len(aligned_images)}")
    print(f"[INFO] Robot frames: {len(required_names)}")
    print(f"[INFO] Missing views filled from robot poses: {len(required_names) - len(aligned_images)}")
    print(f"[INFO] Inferred crop box for filled views (x, y, w, h): {crop_box}")
    print(f"[INFO] Wrote: {sparse_dir / 'images.txt'}")
    print(f"[INFO] Wrote images: {images_dir}")


if __name__ == "__main__":
    main()
