import argparse
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
        description="Add base_T_gripper to each keyframe in a saved viewer camera path JSON."
    )
    parser.add_argument("--path-json", required=True, help="Input camera_path_keyframes.json")
    parser.add_argument("--run-metadata", required=True, help="run_metadata.json containing camera_T_gripper")
    parser.add_argument("--output-json", required=True, help="Output enriched camera path JSON")
    args = parser.parse_args()

    path_json = Path(args.path_json)
    run_metadata = Path(args.run_metadata)
    output_json = Path(args.output_json)

    data = json.loads(path_json.read_text(encoding="utf-8"))
    metadata = json.loads(run_metadata.read_text(encoding="utf-8"))
    camera_T_gripper = load_matrix_field(metadata, "camera_T_gripper")

    for i, keyframe in enumerate(data.get("keyframes", [])):
        if "robot_camera_to_world" not in keyframe:
            raise ValueError(
                f"Keyframe {i} is missing robot_camera_to_world. "
                "Load alignment.json in the viewer before saving the path."
            )
        base_T_camera = np.array(keyframe["robot_camera_to_world"], dtype=float)
        if base_T_camera.shape != (4, 4):
            raise ValueError(f"Keyframe {i} robot_camera_to_world must be 4x4, got {base_T_camera.shape}")

        base_T_gripper = base_T_camera @ camera_T_gripper
        keyframe["robot_ee_frame"] = "franka_gripper"
        keyframe["hand_eye_used"] = "camera_T_gripper"
        keyframe["camera_T_gripper"] = camera_T_gripper.tolist()
        keyframe["base_T_gripper"] = base_T_gripper.tolist()
        keyframe["gripper_center"] = base_T_gripper[:3, 3].tolist()

    data["contains_base_T_gripper"] = True
    data["hand_eye_used"] = "camera_T_gripper"
    data["source_path_json"] = str(path_json)
    data["source_run_metadata"] = str(run_metadata)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote enriched path with {len(data.get('keyframes', []))} keyframes to {output_json}")


if __name__ == "__main__":
    main()
