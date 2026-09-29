"""Observable RGB-D and hierarchical grasp planning primitives for Task E."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import torch


class TaskPhase(str, Enum):
    SEARCH = "search"; PRE_GRASP = "pre_grasp"; APPROACH = "approach"; CLOSE = "close"
    VERIFY_LIFT = "verify_lift"; TRANSPORT = "transport"; PLACE = "place"
    CLEARANCE = "clearance"; CROSSOVER = "crossover"
    VERIFY_PLACE = "verify_place"; RETRY = "retry"; DONE = "done"
    RETRACT = "retract"; HOME = "home"


@dataclass(frozen=True)
class CameraModel:
    fx: float; fy: float; cx: float; cy: float

    def backproject(self, u, v, depth):
        return torch.stack(((u - self.cx) * depth / self.fx,
                            (v - self.cy) * depth / self.fy, depth), dim=-1)


@dataclass(frozen=True)
class GraspTemplate:
    grasp_offset: tuple[float, float, float]
    approach_offset: tuple[float, float, float]
    approach_axis: str
    jaw_open: float
    lift_distance: float


# Mustard is intentionally grasped around the bottle body from the side.
GRASP_TEMPLATES = {
    "sugar_box": GraspTemplate((0.0, 0.0, 0.04), (0.0, 0.0, 0.16), "z", 0.070, 0.12),
    "mustard_bottle": GraspTemplate((0.0, 0.0, 0.075), (0.0, -0.14, 0.075), "y", 0.060, 0.10),
    "banana": GraspTemplate((0.0, 0.0, 0.035), (0.0, 0.0, 0.15), "z", 0.085, 0.12),
}


def _as_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.float()
    if depth.ndim == 3 and depth.shape[-1] == 1: depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1: depth = depth[0]
    finite = depth[torch.isfinite(depth) & (depth > 0)]
    if finite.numel() and float(finite.median()) > 20.0: depth = depth / 1000.0
    return depth


def estimate_centroid(mask: torch.Tensor, depth: torch.Tensor, camera: CameraModel):
    """Median 3-D point of valid mask pixels; returns None for an invalid mask."""
    mask = mask.squeeze().bool(); depth = _as_depth(depth)
    valid = mask & torch.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    v, u = torch.where(valid)
    if u.numel() < 8: return None
    points = camera.backproject(u.float(), v.float(), depth[v, u])
    return points.median(dim=0).values


class TaskEPlanner:
    """ALORE-style ordered subtasks with bounded re-planning."""
    def __init__(self, order=("banana", "mustard_bottle", "sugar_box"), retries=1):
        self.order, self.max_retries = order, retries; self.reset()

    def reset(self):
        self.index = 0; self.phase = TaskPhase.SEARCH; self.retries = 0; self.target = None

    @property
    def finished(self): return self.phase is TaskPhase.DONE

    def begin_target(self, label):
        if label not in GRASP_TEMPLATES: raise KeyError(f"unknown Task E object label: {label}")
        self.target, self.phase, self.retries = label, TaskPhase.PRE_GRASP, 0
        return GRASP_TEMPLATES[label]

    def advance(self, success=None):
        if self.phase in (TaskPhase.VERIFY_LIFT, TaskPhase.VERIFY_PLACE) and success is False:
            return self._retry()
        sequence = [TaskPhase.PRE_GRASP, TaskPhase.APPROACH, TaskPhase.CLOSE,
                    TaskPhase.VERIFY_LIFT, TaskPhase.CLEARANCE, TaskPhase.CROSSOVER,
                    TaskPhase.TRANSPORT, TaskPhase.PLACE,
                    TaskPhase.VERIFY_PLACE, TaskPhase.RETRACT, TaskPhase.HOME]
        if self.phase in sequence:
            pos = sequence.index(self.phase)
            if pos + 1 < len(sequence): self.phase = sequence[pos + 1]
            else:
                self.index += 1; self.target = None
                self.phase = TaskPhase.DONE if self.index >= len(self.order) else TaskPhase.SEARCH
        return self.phase

    def _retry(self):
        if self.retries >= self.max_retries: self.phase = TaskPhase.DONE
        else: self.retries += 1; self.phase = TaskPhase.RETRY
        return self.phase

    def after_retry(self):
        self.phase = TaskPhase.PRE_GRASP if self.target else TaskPhase.SEARCH
        return self.phase


__all__ = ["CameraModel", "GRASP_TEMPLATES", "GraspTemplate", "TaskEPlanner", "TaskPhase", "estimate_centroid"]
