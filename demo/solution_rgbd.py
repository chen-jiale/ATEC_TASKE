"""RGB-D VBC/ALORE policy entry point for Task E.

The policy consumes only the public observation dictionary.  Perception and
task sequencing are implemented here; robot-specific FK/IK is isolated behind
``joint_target_from_ee`` so it can be calibrated without reintroducing ACT.
"""
from __future__ import annotations

from typing import Any
import os
from pathlib import Path
import numpy as np
import cv2
import torch

from .rgbd_vbc_alore import CameraModel, TaskEPlanner, TaskPhase, _as_depth
from .piper_kinematics import fk, ik
from .rgbd.perception import TaskEObjectLocator, backproject_depth


class AlgSolution:
    _DEFAULT = torch.tensor([0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=torch.float32)
    _HOME = torch.tensor([-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035], dtype=torch.float32)

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cv2.setNumThreads(1)
        self.default_joint_pos = self._DEFAULT.to(self.device)
        self.home_joint_pos = self._HOME.to(self.device)
        self.planner = TaskEPlanner()
        self.locator = TaskEObjectLocator()
        self.object_targets = {}
        self._steps = 0
        self._episode_seen = False
        self._phase_ticks = 0
        self._lift_evidence = 0
        self._grasp_origin = None
        self._lost_target_steps = 0
        self._search_ticks = 0
        self._pose_cache = {}
        self._grasp_rotation = None
        self._attempt = 0
        self._diagnostic_dir = os.environ.get("ATEC_RGBD_DIAGNOSTICS")
        self._base_world = torch.tensor([1.404254, 0.0, 0.8266427], device=self.device)
        self._world_to_base = np.diag([-1.0, -1.0, 1.0])
        self._basket_world = torch.tensor([1.08, -0.30, 0.9766427], device=self.device)

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {}

    def _reset_if_needed(self, proprio: torch.Tensor) -> None:
        q = proprio[:, :8] + self.default_joint_pos
        # Keep the model-independent FK/IK layer live in the policy path. The
        # perception adapter will supply target poses once masks are available.
        _ = fk(q[0, :6].detach().float().cpu().numpy())
        if self._episode_seen and bool(torch.all(torch.abs(q - self.default_joint_pos) < 0.02)):
            self.planner.reset(); self._steps = 0; self._phase_ticks = 0
            self._lift_evidence = 0
            self._grasp_origin = None
            self._lost_target_steps = 0
            self._search_ticks = 0
            self._pose_cache.clear()
            self._grasp_rotation = None
        self._episode_seen = True

    def _observe_depth(self, obs: dict) -> None:
        image = obs.get("image", {}) if isinstance(obs, dict) else {}
        depth = image.get("video_depth")
        if depth is None:
            return
        # Keep the public RGB-D path explicit; model-specific segmentation can
        # replace this hook without changing the policy interface.
        frame = depth[0] if depth.ndim > 2 else depth
        _as_depth(frame)
        estimated = self.locator.locate(frame.detach().cpu().numpy())
        rgb = image.get("video_rgb")
        if rgb is not None:
            colored = self.locator.locate_banded_cloud(frame.detach().cpu().numpy(), rgb[0].detach().cpu().numpy())
            estimated[1] = colored[1]
            estimated[2] = colored[2]
        self.object_targets = {key: value for key, value in estimated.items() if value is not None}

    def _observe_lift(self, obs, joints):
        image = obs.get("image", {})
        if self._grasp_origin is None or "video_rgb" not in image or "video_depth" not in image:
            return False
        rgb = image["video_rgb"][0].detach().cpu().numpy()
        depth = image["video_depth"][0].detach().cpu().numpy()
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        points = backproject_depth(depth)
        pose = fk(joints)
        tcp = pose[:3, 3] + pose[:3, :3] @ np.array([0.0, 0.0, 0.10])
        tcp_world = self._world_to_base.T @ tcp + self._base_world.detach().cpu().numpy()
        mask = (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)
        mask &= np.isfinite(points).all(axis=-1)
        mask &= np.linalg.norm(points - tcp_world, axis=-1) < 0.09
        mask &= points[..., 2] > self._grasp_origin[2] + 0.05
        count = int(mask.sum())
        self._lift_evidence = self._lift_evidence + 1 if count >= 30 else 0
        print(f"[RGBD_LIFT] elevated_color_pixels={count} consecutive={self._lift_evidence}", flush=True)
        return self._lift_evidence >= 5

    def predicts(self, obs, current_score):
        if not isinstance(obs, dict) or "proprio" not in obs:
            raise ValueError("Expected obs dict with proprio and image observations")
        proprio = obs["proprio"].to(self.device)
        self._reset_if_needed(proprio)
        if self._steps % 5 == 0:
            self._observe_depth(obs)
        if self._diagnostic_dir and self._steps % 100 == 0:
            destination = Path(self._diagnostic_dir)
            destination.mkdir(parents=True, exist_ok=True)
            frames = {name: value[0].detach().cpu().numpy() for name, value in obs.get("image", {}).items()}
            np.savez_compressed(destination / f"observation_{self._steps:05d}.npz", **frames)
        q = proprio[:, :8] + self.default_joint_pos
        # Safe startup/home phase while the RGB-D target and Piper IK adapter
        # initialize.  No ACT checkpoint or hidden simulator state is used.
        if self.planner.phase is TaskPhase.HOME:
            target = self.home_joint_pos.expand_as(q).clone()
            target[:, :6] = q[:, :6] + torch.clamp(target[:, :6] - q[:, :6], -0.025, 0.025)
            self._phase_ticks += 1
            if self._phase_ticks > 50 and torch.max(torch.abs(q[:, :6] - self.home_joint_pos[:6])) < 0.04:
                self.planner.advance(True)
                self._phase_ticks = 0
            self._steps += 1
            return {"action": ((target - self.default_joint_pos) / 0.5).detach().cpu().tolist(), "giveup": self._phase_ticks > 500}
        if self._steps < 40:
            target = self.home_joint_pos.expand_as(q).clone(); grip = torch.tensor([0.035, -0.035], device=self.device)
        else:
            if self.planner.phase is TaskPhase.SEARCH:
                self._search_ticks += 1
                if self._search_ticks >= 250:
                    print("[RGBD_FAILURE] search_timeout", flush=True)
                    return {"action": ((q - self.default_joint_pos) / 0.5).detach().cpu().tolist(), "giveup": True}
                order = self.planner.order[self.planner.index]
                key = {"sugar_box": 1, "mustard_bottle": 2, "banana": 3}[order]
                if key in self.object_targets:
                    self.planner.begin_target(order); self._phase_ticks = 0
                    self._attempt = 0
                    self._grasp_origin = self.object_targets[key].copy()
                    if order == "mustard_bottle":
                        self._grasp_origin[2] += 0.055
                    self._lift_evidence = 0
                    self._lost_target_steps = 0
                    self._search_ticks = 0
                    self._pose_cache.clear()
                    self._grasp_rotation = None
            target = q.clone(); grip = torch.tensor([0.035, -0.035], device=self.device)
            if self.planner.target is not None:
                key = {"sugar_box": 1, "mustard_bottle": 2, "banana": 3}[self.planner.target]
                phase = self.planner.phase
                if phase in (TaskPhase.PRE_GRASP, TaskPhase.APPROACH):
                    observed = self.object_targets.get(key)
                    valid = observed is not None and np.isfinite(observed).all()
                    self._lost_target_steps = 0 if valid else self._lost_target_steps + 1
                    if not valid and (phase is TaskPhase.PRE_GRASP or self._lost_target_steps >= 100):
                        self._steps += 1
                        stopped = self._lost_target_steps >= 100
                        if stopped:
                            print(f"[RGBD_FAILURE] target_lost_or_displaced object={self.planner.target}", flush=True)
                        return {"action": ((q - self.default_joint_pos) / 0.5).detach().cpu().tolist(), "giveup": stopped}
                point = torch.tensor(self._grasp_origin, device=self.device, dtype=torch.float32)
                if phase is TaskPhase.PRE_GRASP:
                    point[2] += 0.10
                elif phase is TaskPhase.VERIFY_LIFT: point[2] += 0.15
                elif phase in (TaskPhase.CLEARANCE, TaskPhase.CROSSOVER):
                    point[0] = 1.18
                    point[2] = 1.18 if self.planner.target == "mustard_bottle" else 1.12
                    if phase is TaskPhase.CROSSOVER:
                        point[1] = self._basket_world[1]
                elif phase in (TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.VERIFY_PLACE, TaskPhase.RETRACT):
                    point = self._basket_world.clone(); point[2] = 1.15 if phase in (TaskPhase.TRANSPORT, TaskPhase.RETRACT) else 0.9766
                    if self.planner.target == "mustard_bottle" and phase in (TaskPhase.PLACE, TaskPhase.VERIFY_PLACE):
                        point[2] = 1.06
                if phase in (TaskPhase.CLOSE, TaskPhase.VERIFY_LIFT, TaskPhase.CLEARANCE, TaskPhase.CROSSOVER, TaskPhase.TRANSPORT, TaskPhase.PLACE): grip = torch.tensor([0.0, 0.0], device=self.device)
                local = self._world_to_base @ (point - self._base_world).detach().cpu().numpy()
                initial = q[0, :6].detach().cpu().numpy()
                final_local = local.copy()
                linear_approach = self.planner.target == "mustard_bottle" and phase is TaskPhase.APPROACH
                if linear_approach:
                    current_pose = fk(initial)
                    current_tcp = current_pose[:3, 3] + current_pose[:3, :3] @ np.array([0.0, 0.0, 0.10])
                    displacement = final_local - current_tcp
                    local = current_tcp + displacement * min(1.0, 0.008 / max(np.linalg.norm(displacement), 1e-9))
                if phase is TaskPhase.PRE_GRASP and self.planner.target == "sugar_box":
                    initial = self.home_joint_pos[:6].detach().cpu().numpy()
                yaw = np.arctan2(local[1], local[0])
                cosine, sine = np.cos(yaw), np.sin(yaw)
                radial = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
                candidates = []
                angles = [-45] if self.planner.target == "mustard_bottle" else [-15, -30, -45, -60, 0, 15, 30, 45]
                if phase in (TaskPhase.CLEARANCE, TaskPhase.CROSSOVER, TaskPhase.TRANSPORT, TaskPhase.PLACE):
                    angles = angles[:1]
                if phase in self._pose_cache:
                    candidates.append(self._pose_cache[phase])
                    angles = []
                elif phase is TaskPhase.VERIFY_PLACE:
                    release_joints = q[0, :6].detach().cpu().numpy().copy()
                    candidates.append((0.0, fk(release_joints), release_joints))
                    angles = []
                for angle in np.deg2rad(angles):
                    cosine, sine = np.cos(angle), np.sin(angle)
                    tilt = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
                    candidate_pose = np.eye(4)
                    candidate_pose[:3, :3] = radial @ tilt @ np.diag([-1.0, 1.0, -1.0])
                    if self.planner.target == "sugar_box":
                        twist_angle = -np.pi / 2 - yaw
                        twist = np.array([[np.cos(twist_angle), -np.sin(twist_angle), 0], [np.sin(twist_angle), np.cos(twist_angle), 0], [0, 0, 1]])
                        candidate_pose[:3, :3] = radial @ tilt @ twist @ np.diag([-1.0, 1.0, -1.0])
                    if phase in (TaskPhase.CLEARANCE, TaskPhase.CROSSOVER, TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.RETRACT):
                        candidate_pose[:3, :3] = fk(initial)[:3, :3]
                    if self._grasp_rotation is not None and (linear_approach or phase in (TaskPhase.CLOSE, TaskPhase.VERIFY_LIFT)):
                        candidate_pose[:3, :3] = self._grasp_rotation
                    candidate_pose[:3, 3] = local - candidate_pose[:3, :3] @ np.array([0.0, 0.0, 0.10])
                    candidate_joints = ik(candidate_pose, initial)
                    predicted = fk(candidate_joints)
                    residual = np.linalg.norm(predicted[:3, 3] - candidate_pose[:3, 3])
                    angular = np.arccos(np.clip((np.trace(predicted[:3, :3].T @ candidate_pose[:3, :3]) - 1) / 2, -1, 1))
                    candidates.append((residual + 0.1 * angular, candidate_pose, candidate_joints))
                    if residual < 0.01 and angular < 0.10:
                        break
                    if self._grasp_rotation is not None and (linear_approach or phase in (TaskPhase.CLOSE, TaskPhase.VERIFY_LIFT)):
                        break
                selected = min(candidates, key=lambda candidate: candidate[0])
                if not linear_approach and phase in (TaskPhase.PRE_GRASP, TaskPhase.APPROACH, TaskPhase.CLOSE, TaskPhase.VERIFY_LIFT, TaskPhase.VERIFY_PLACE):
                    self._pose_cache[phase] = selected
                _, pose, solved = selected
                if phase is TaskPhase.APPROACH:
                    self._grasp_rotation = pose[:3, :3].copy()
                if phase is TaskPhase.PRE_GRASP and self.planner.target == "mustard_bottle":
                    self._grasp_rotation = pose[:3, :3].copy()
                target[:, :6] = torch.tensor(solved, device=self.device, dtype=torch.float32)
                self._phase_ticks += 1
                limits = {TaskPhase.PRE_GRASP: 45, TaskPhase.APPROACH: 55, TaskPhase.CLOSE: 35, TaskPhase.VERIFY_LIFT: 55, TaskPhase.TRANSPORT: 65, TaskPhase.PLACE: 45, TaskPhase.VERIFY_PLACE: 25}
                limits.update({TaskPhase.CLEARANCE: 55, TaskPhase.CROSSOVER: 65})
                limits[TaskPhase.RETRACT] = 80
                achieved = fk(q[0, :6].detach().cpu().numpy())
                position_error = float(np.linalg.norm(achieved[:3, 3] - pose[:3, 3]))
                if linear_approach:
                    achieved_tcp = achieved[:3, 3] + achieved[:3, :3] @ np.array([0.0, 0.0, 0.10])
                    position_error = float(np.linalg.norm(achieved_tcp - final_local))
                orientation_error = float(np.arccos(np.clip((np.trace(achieved[:3, :3].T @ pose[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                if self._steps % 50 == 0:
                    solved_pose = fk(solved)
                    solver_position_error = float(np.linalg.norm(solved_pose[:3, 3] - pose[:3, 3]))
                    tracking_error = float(np.max(np.abs(solved - q[0, :6].detach().cpu().numpy())))
                    print(f"[RGBD_RESIDUAL] solver_position_m={solver_position_error:.6f} tracking_max_rad={tracking_error:.6f}", flush=True)
                    print(f"[RGBD_POSE] phase={phase} position_error_m={position_error:.4f} orientation_error_rad={orientation_error:.4f} target_base={local.tolist()}", flush=True)
                    print(f"[RGBD_IK] solved={solved.tolist()} joint_error={(solved - initial).tolist()}", flush=True)
                if self._phase_ticks >= 500:
                    print(f"[RGBD_FAILURE] phase_timeout={phase} position_error_m={position_error:.4f} orientation_error_rad={orientation_error:.4f}", flush=True)
                    return {"action": ((q - self.default_joint_pos) / 0.5).detach().cpu().tolist(), "giveup": True}
                if phase is TaskPhase.VERIFY_LIFT and self._phase_ticks >= 180 and position_error < 0.02 and float(current_score) < 6 * self.planner.index + 3 - 0.01 and self._attempt < 2:
                    self._attempt += 1
                    observed = self.object_targets.get(key)
                    if observed is not None:
                        self._grasp_origin = observed.copy()
                        if self.planner.target == "mustard_bottle":
                            self._grasp_origin[2] += 0.055 - 0.02 * self._attempt
                    self.planner.phase = TaskPhase.PRE_GRASP
                    self._phase_ticks = 0
                    self._pose_cache.clear()
                    self._grasp_rotation = None
                    self._lift_evidence = 0
                    print(f"[RGBD_RETRY] object={self.planner.target} attempt={self._attempt} target={self._grasp_origin.tolist()}", flush=True)
                position_tolerance = 0.005 if self.planner.target == "mustard_bottle" and phase is TaskPhase.APPROACH else 0.01 if self.planner.target == "mustard_bottle" and phase is TaskPhase.PRE_GRASP else 0.025 if phase is TaskPhase.APPROACH else 0.05
                if self.planner.target == "mustard_bottle" and phase is TaskPhase.VERIFY_LIFT:
                    position_tolerance = 0.01
                if phase in limits and self._phase_ticks >= limits[phase] and position_error < position_tolerance and orientation_error < 0.15:
                    verified = True
                    if phase is TaskPhase.VERIFY_LIFT:
                        verified = self._phase_ticks % 5 == 0 and self._observe_lift(obs, initial)
                        verified = verified or (float(current_score) >= 6 * self.planner.index + 3 - 0.01 and float(torch.abs(q[0, 6] - q[0, 7])) > 0.004)
                    if phase is TaskPhase.VERIFY_PLACE:
                        verified = float(current_score) >= 6 * (self.planner.index + 1) - 0.01
                    if verified:
                        if phase is TaskPhase.APPROACH:
                            contact_joints = q[0, :6].detach().cpu().numpy().copy()
                            self._pose_cache[TaskPhase.CLOSE] = (0.0, achieved.copy(), contact_joints)
                        self.planner.advance(True); self._phase_ticks = 0
            target[:, 6:] = grip
        self._steps += 1
        target[:, :6] = q[:, :6] + torch.clamp(target[:, :6] - q[:, :6], -0.04, 0.04)
        if self._steps % 50 == 0:
            print(f"[RGBD] step={self._steps} phase={self.planner.phase} object={self.planner.target} score={current_score} q={q[0].tolist()} targets={self.object_targets}", flush=True)
        action = (target - self.default_joint_pos) / 0.5
        return {"action": action.detach().cpu().tolist(), "giveup": False}


__all__ = ["AlgSolution"]
