"""Audit camera geometry and FK using paired offline diagnostic samples."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.piper_kinematics import fk
from demo.rgbd.perception import CAMERA_POS, R_WORLD_CAMERA, TABLE_Z, backproject_depth


def evaluate(directory):
    records = []
    for path in sorted(directory.glob("sample_*.npz")):
        with np.load(path) as sample:
            points = backproject_depth(sample["video_depth"])
            patch = points[310:355, 210:350]
            heights = patch[..., 2]
            heights = heights[np.isfinite(heights)]
            joints = np.asarray(sample["joint_positions"]).reshape(-1)[:6]
            predicted = np.diag([-1., -1., 1.]) @ fk(joints)[:3, 3]
            predicted += np.array([1.404254, 0., 0.8266427])
            measured = np.asarray(sample["gripper_world"]).reshape(-1)[:3]
            records.append({
                "sample": path.name,
                "table_valid_pixels": int(heights.size),
                "table_median_error_mm": float((np.median(heights) - TABLE_Z) * 1000) if heights.size else None,
                "table_absolute_p95_mm": float(np.quantile(np.abs(heights - TABLE_Z), .95) * 1000) if heights.size else None,
                "fk_error_mm": float(np.linalg.norm(predicted - measured) * 1000),
            })
    if not records:
        raise ValueError("No paired calibration samples found")
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = R_WORLD_CAMERA
    world_from_camera[:3, 3] = CAMERA_POS
    base_from_world = np.diag([-1., -1., 1., 1.])
    base_from_world[:3, 3] = [1.404254, 0., -.8266427]
    return {
        "scope": "Fixed external camera and gripper-base position; not wrist-camera or TCP contact calibration",
        "depth_convention": "metres, axial depth along camera local +X",
        "table_patch_caveat": "Fixed image patch; occluded frames must be rejected by visual inspection",
        "world_from_camera": world_from_camera.tolist(),
        "base_from_camera": (base_from_world @ world_from_camera).tolist(),
        "fk_max_error_mm": max(record["fk_error_mm"] for record in records),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = evaluate(arguments.directory)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
