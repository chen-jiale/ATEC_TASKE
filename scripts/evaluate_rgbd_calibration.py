"""Evaluate stored, paired RGB-D calibration samples outside the policy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.rgbd.perception import TaskEObjectLocator


def evaluate(paths):
    records = []
    for path in paths:
        with np.load(path) as sample:
            locator = TaskEObjectLocator()
            for method in ("locate", "locate_pointcloud", "locate_banded_cloud"):
                estimates = locator.locate_banded_cloud(sample["video_depth"], sample["video_rgb"]) if method == "locate_banded_cloud" else getattr(locator, method)(sample["video_depth"])
                for object_id in (1, 2, 3):
                    estimate = estimates.get(object_id)
                    measured = sample[f"object_{object_id}_world"]
                    delta = None if estimate is None else np.asarray(estimate) - measured
                    records.append({"sample": str(path), "method": method, "object": object_id,
                                    "delta_mm": None if delta is None else (delta * 1000).tolist(),
                                    "error_mm": None if delta is None else float(np.linalg.norm(delta) * 1000)})
    summaries = []
    for method in ("locate", "locate_pointcloud", "locate_banded_cloud"):
        for object_id in (1, 2, 3):
            selected = [record for record in records if record["method"] == method and record["object"] == object_id]
            errors = [record["error_mm"] for record in selected if record["error_mm"] is not None]
            summaries.append({"method": method, "object": object_id, "samples": len(selected),
                              "misses": len(selected) - len(errors),
                              "median_mm": float(np.median(errors)) if errors else None,
                              "p95_mm": float(np.quantile(errors, 0.95)) if errors else None})
    return {"summary": summaries, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    samples = sorted(arguments.directory.glob("sample_*.npz"))
    if not samples:
        parser.error("No paired calibration samples found")
    report = evaluate(samples)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
