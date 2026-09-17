import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_matrix_field(data: dict, key: str) -> np.ndarray:
    value = data[key]
    if isinstance(value, dict) and "matrix" in value:
        value = value["matrix"]
    mat = np.array(value, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"{key} must be a 4x4 matrix, got {mat.shape}")
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert saved viewer camera keyframes into robot EE/gripper poses using hand-eye calibration."
    )
    parser.add_argument("--path-json", required=True, help="camera_path_keyframes.json exported by gs_pose_viewer")
    parser.add_argument("--run-metadata", required=True, help="Dataset run_metadata.json containing camera_T_gripper")
    parser.add_argument("--output-json", required=True, help="Output JSON with base_T_gripper poses")
    parser.add_argument("--output-csv", default=None, help="Optional CSV with flattened poses")
    args = parser.parse_args()

    path_json = Path(args.path_json)
    run_metadata = Path(args.run_metadata)
    out_json = Path(args.output_json)

    path_data = json.loads(path_json.read_text(encoding="utf-8"))
    metadata = json.loads(run_metadata.read_text(encoding="utf-8"))

    # The viewer exports robot_camera_to_world as base_T_camera.
    # The metadata stores camera_T_gripper, which converts a gripper-frame point
    # into the camera frame. Therefore:
    #
    #   base_T_camera = base_T_gripper @ gripper_T_camera
    #   base_T_gripper = base_T_camera @ camera_T_gripper
    #
    camera_T_gripper = load_matrix_field(metadata, "camera_T_gripper")

    output_keyframes = []
    for keyframe in path_data["keyframes"]:
        if "robot_camera_to_world" not in keyframe:
            raise ValueError(
                f"Keyframe {keyframe.get('keyframe_index')} is missing robot_camera_to_world. "
                "Load alignment.json in the viewer before exporting the path."
            )
        base_T_camera = np.array(keyframe["robot_camera_to_world"], dtype=float)
        if base_T_camera.shape != (4, 4):
            raise ValueError(f"robot_camera_to_world must be 4x4, got {base_T_camera.shape}")

        base_T_gripper = base_T_camera @ camera_T_gripper
        output_keyframes.append(
            {
                "keyframe_index": keyframe.get("keyframe_index"),
                "name": keyframe.get("name"),
                "created_at": keyframe.get("created_at"),
                "base_T_camera": base_T_camera.tolist(),
                "base_T_gripper": base_T_gripper.tolist(),
                "ee_position_xyz": base_T_gripper[:3, 3].tolist(),
            }
        )

    result = {
        "source_path_json": str(path_json),
        "source_run_metadata": str(run_metadata),
        "duration_sec": path_data.get("duration_sec"),
        "hand_eye_used": "camera_T_gripper",
        "camera_T_gripper": camera_T_gripper.tolist(),
        "keyframes": output_keyframes,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["keyframe_index", "name", "ee_x", "ee_y", "ee_z"] + [f"T{i}{j}" for i in range(4) for j in range(4)]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for keyframe in output_keyframes:
                T = np.array(keyframe["base_T_gripper"], dtype=float)
                row = {
                    "keyframe_index": keyframe["keyframe_index"],
                    "name": keyframe["name"],
                    "ee_x": T[0, 3],
                    "ee_y": T[1, 3],
                    "ee_z": T[2, 3],
                }
                row.update({f"T{i}{j}": T[i, j] for i in range(4) for j in range(4)})
                writer.writerow(row)

    print(f"Wrote {len(output_keyframes)} EE poses to {out_json}")
    if args.output_csv:
        print(f"Wrote CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
