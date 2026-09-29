"""Summarize paired offline frames without exposing truth to the policy."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def summarize(directory, output):
    paths = sorted(directory.glob("sample_*.npz"))
    if not paths:
        raise ValueError("No paired frames available")
    records = []
    reference_path = directory / "sample_00100.npz"
    if not reference_path.exists():
        reference_path = paths[0]
    with np.load(reference_path) as sample:
        reference = np.stack([sample[f"object_{index}_world"].reshape(-1)[:3] for index in (1, 2, 3)])
    images = []
    for path in paths:
        with np.load(path) as sample:
            positions = np.stack([sample[f"object_{index}_world"].reshape(-1)[:3] for index in (1, 2, 3)])
            records.append({"frame": path.name, "positions_world_m": positions.tolist(),
                            "reference_frame": reference_path.name,
                            "displacement_from_reference_mm": (1000 * np.linalg.norm(positions - reference, axis=1)).tolist()})
            frame = cv2.cvtColor(sample["video_rgb"].astype(np.uint8), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))
            cv2.putText(frame, path.stem, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            images.append(frame)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contacts.json").write_text(json.dumps(records, indent=2) + "\n")
    for start in range(0, len(images), 6):
        selected = images[start:start + 6]
        selected += [np.zeros_like(images[0]) for _ in range(6 - len(selected))]
        montage = np.concatenate([np.concatenate(selected[offset:offset + 2], axis=1) for offset in (0, 2, 4)], axis=0)
        if not cv2.imwrite(str(output / f"frames_{start:03d}.jpg"), montage):
            raise OSError("Could not save diagnostic montage")
    print(json.dumps(records[-3:], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    summarize(arguments.directory, arguments.output)
