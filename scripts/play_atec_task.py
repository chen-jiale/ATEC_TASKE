# Created by skywoodsz on 2026/02/07.

import argparse
import hashlib
import json
import os
import re
import sys
import time

from isaaclab.app import AppLauncher

# Make the repository root importable so local packages like `demo` work
# regardless of whether this script is launched from the repo root or elsewhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate ATEC Task E tabletop manipulation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Reproducible environment reset seed.")
parser.add_argument("--output_dir", type=str, default=None, help="Run-specific video output directory.")
parser.add_argument("--camera_video", action="store_true", help="Record the policy RGB camera instead of the overview viewport.")
parser.add_argument("--task", type=str, default="ATEC-TaskE-Piper", choices=["ATEC-TaskE-Piper"], help="Task E environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

from demo.solution import AlgSolution
solution = AlgSolution()


def _disable_cameras_if_needed(env_cfg, enable_cameras: bool):
    """Drop camera sensors and image observations for low-memory, proprio-only runs."""
    if enable_cameras:
        print("[INFO] Camera mode enabled. Keeping camera sensors and image observations.")
        return env_cfg

    print("[INFO] Camera mode disabled. Switching to low-memory proprio-only runtime.")

    if hasattr(env_cfg, "scene"):
        if hasattr(env_cfg.scene, "head_camera"):
            env_cfg.scene.head_camera = None
        if hasattr(env_cfg.scene, "ee_camera"):
            env_cfg.scene.ee_camera = None
        if hasattr(env_cfg.scene, "ee_dual_camera"):
            env_cfg.scene.ee_dual_camera = None

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "image"):
        env_cfg.observations.image = None

    print("[INFO] Disabled sensors: head_camera, ee_camera, ee_dual_camera")
    print("[INFO] Disabled observation group: image")
    return env_cfg


def _resolve_body_ids_from_patterns(body_names: list[str], patterns) -> list[int]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    resolved_ids: list[int] = []
    for idx, body_name in enumerate(body_names):
        for pattern in patterns:
            if re.fullmatch(pattern, body_name):
                resolved_ids.append(idx)
                break
    return resolved_ids


def _log_default_joint_pos_diagnostics(env, env_cfg):
    robot = env.unwrapped.scene["robot"]
    joint_names = list(getattr(robot, "joint_names", []))
    action_joint_names = list(env_cfg.actions.joint_arm.joint_names)
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu()

    action_order_indices = [joint_names.index(name) for name in action_joint_names]
    action_default_positions = [float(default_joint_pos[index].item()) for index in action_order_indices]
    print(f"[DEBUG] Task E native joint order: {joint_names}")
    print(f"[DEBUG] Task E action joint order: {action_joint_names}")
    print(f"[DEBUG] Task E default positions in action order: {action_default_positions}")


def _infer_termination_reason(env, env_cfg, terminated, truncated) -> str:
    reasons: list[str] = []
    robot = env.unwrapped.scene["robot"]
    root_pos_w = robot.data.root_pos_w[0].detach().cpu()

    if bool(truncated.item()):
        reasons.append("time_out")

    fall_cfg = getattr(getattr(env_cfg, "terminations", None), "fall", None)
    if fall_cfg is not None:
        min_height = float(fall_cfg.params["minimum_height"])
        if float(root_pos_w[2].item()) < min_height:
            reasons.append(f"fall(z={float(root_pos_w[2].item()):.4f}<min={min_height:.4f})")

    illegal_contact_cfg = getattr(getattr(env_cfg, "terminations", None), "illegal_contact", None)
    if illegal_contact_cfg is not None:
        contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
        body_names = list(getattr(contact_sensor, "body_names", []))
        body_patterns = illegal_contact_cfg.params["sensor_cfg"].body_names
        body_ids = _resolve_body_ids_from_patterns(body_names, body_patterns)
        threshold = float(illegal_contact_cfg.params["threshold"])
        if body_ids:
            net_forces = contact_sensor.data.net_forces_w_history[0, :, body_ids, :]
            max_force = float(net_forces.norm(dim=-1).max().item())
            if max_force > threshold:
                hit_names = [body_names[idx] for idx in body_ids]
                reasons.append(
                    "illegal_contact("
                    f"max_force={max_force:.4f}>thr={threshold:.4f}, bodies={hit_names}"
                    ")"
                )

    if not reasons and bool(terminated.item()):
        reasons.append("terminated_unknown")

    return " | ".join(reasons) if reasons else "not_terminated"


def _format_top_contact_bodies(env, top_k: int = 8) -> list[tuple[str, float]]:
    contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
    body_names = list(getattr(contact_sensor, "body_names", []))
    net_forces = contact_sensor.data.net_forces_w_history[0]
    if net_forces.numel() == 0 or len(body_names) == 0:
        return []
    body_force_max = net_forces.norm(dim=-1).max(dim=0)[0].detach().cpu()
    pairs = []
    for idx, body_name in enumerate(body_names[: body_force_max.shape[0]]):
        pairs.append((body_name, float(body_force_max[idx].item())))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:top_k]


def _log_terminal_state_diagnostics(env, env_cfg, terminated, truncated, obs):
    robot = env.unwrapped.scene["robot"]
    root_pos_w = robot.data.root_pos_w[0].detach().cpu().tolist()
    root_lin_vel_w = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
    root_ang_vel_w = robot.data.root_ang_vel_w[0].detach().cpu().tolist()
    projected_gravity_b = robot.data.projected_gravity_b[0].detach().cpu().tolist()

    print("[DEBUG] Terminal state diagnostics:")
    print(f"  terminated={bool(terminated.item())}, truncated={bool(truncated.item())}")
    print(f"  root_pos_w={root_pos_w}")
    print(f"  root_lin_vel_w={root_lin_vel_w}")
    print(f"  root_ang_vel_w={root_ang_vel_w}")
    print(f"  projected_gravity_b={projected_gravity_b}")

    if isinstance(obs, dict) and "proprio" in obs:
        proprio = obs["proprio"][0].detach().cpu()
        print(f"  final_proprio_shape={tuple(obs['proprio'].shape)}")
        print(f"  final_joint_pos_rel_obs={proprio[0:8].tolist()}")
        print(f"  final_joint_vel_rel_obs={proprio[8:16].tolist()}")
        print(f"  final_previous_action_obs={proprio[16:24].tolist()}")

    illegal_contact_cfg = getattr(getattr(env_cfg, "terminations", None), "illegal_contact", None)
    if illegal_contact_cfg is not None:
        last_illegal_contact_debug = getattr(env.unwrapped, "_last_illegal_contact_debug", None)
        if last_illegal_contact_debug:
            print("[DEBUG] Illegal-contact snapshot from termination callback:")
            for env_index, entry in enumerate(last_illegal_contact_debug):
                print(
                    f"  env[{env_index}] terminated={entry['terminated']} "
                    f"threshold={entry['threshold']} body_force_pairs={entry['body_force_pairs'][:8]}"
                )

        contact_sensor = env.unwrapped.scene.sensors["contact_sensor"]
        body_names = list(getattr(contact_sensor, "body_names", []))
        body_patterns = illegal_contact_cfg.params["sensor_cfg"].body_names
        body_ids = _resolve_body_ids_from_patterns(body_names, body_patterns)
        threshold = float(illegal_contact_cfg.params["threshold"])
        net_forces = contact_sensor.data.net_forces_w_history[0]
        if body_ids:
            selected = net_forces[:, body_ids, :].norm(dim=-1).max(dim=0)[0].detach().cpu()
            selected_pairs = [(body_names[body_ids[i]], float(selected[i].item())) for i in range(len(body_ids))]
            selected_pairs.sort(key=lambda item: item[1], reverse=True)
            print("[DEBUG] Illegal-contact body diagnostics:")
            print(f"  patterns={body_patterns}")
            print(f"  threshold={threshold}")
            print(f"  candidate_bodies={selected_pairs}")

    print(f"[DEBUG] Top contact bodies: {_format_top_contact_bodies(env)}")

    fall_cfg = getattr(getattr(env_cfg, "terminations", None), "fall", None)
    if fall_cfg is not None:
        min_height = float(fall_cfg.params["minimum_height"])
        print(f"[DEBUG] Fall threshold check: root_z={root_pos_w[2]:.4f}, minimum_height={min_height:.4f}")


def _log_termination_manager_debug(env, info):
    print("[DEBUG] Step info diagnostics:")
    if isinstance(info, dict):
        print(f"  info_keys={sorted(info.keys())}")
        for key, value in info.items():
            if hasattr(value, "shape"):
                print(f"  info[{key}] shape={tuple(value.shape)} value={value}")
            else:
                print(f"  info[{key}]={value}")
    else:
        print(f"  info_type={type(info).__name__}")

    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    if termination_manager is None:
        print("[DEBUG] termination_manager is not available on env.unwrapped")
        return

    print("[DEBUG] TerminationManager diagnostics:")
    active_terms = getattr(termination_manager, "_term_names", None)
    if active_terms is None:
        active_terms = getattr(termination_manager, "active_terms", None)
    print(f"  active_terms={active_terms}")

    for attr_name in [
        "dones",
        "terminated",
        "time_outs",
        "_truncated_buf",
        "_terminated_buf",
        "_term_dones",
        "_term_values",
        "_term_cfgs",
    ]:
        attr_value = getattr(termination_manager, attr_name, None)
        if attr_value is None:
            continue
        if hasattr(attr_value, "shape"):
            print(f"  termination_manager.{attr_name} shape={tuple(attr_value.shape)} value={attr_value}")
        else:
            print(f"  termination_manager.{attr_name}={attr_value}")

    for method_name in ["get_active_iterable_terms", "get_active_terms"]:
        method = getattr(termination_manager, method_name, None)
        if callable(method):
            try:
                print(f"  {method_name}() -> {method(0)}")
            except Exception as exc:
                print(f"  {method_name}() failed: {exc}")

def play() -> tuple[float, float]:
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # Read the participant action spec directly from AlgSolution and validate it
    # before creating the environment.
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    env_cfg = type(env_cfg)(seed=args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    print(f"[INFO] Creating environment for task: {args_cli.task}")
    print(f"[INFO] Requested action spec: {action_spec}")
    env_cfg = apply_safe_action_spec(env_cfg, action_spec)
    env_cfg = _disable_cameras_if_needed(env_cfg, getattr(args_cli, "enable_cameras", False))
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    terminal_evidence = {}
    terminal_camera_frame = None
    original_reset = env.unwrapped._reset_idx

    def audited_reset(env_ids):
        nonlocal terminal_camera_frame
        manager = env.unwrapped.termination_manager
        if bool(manager.dones.any()):
            terminal_evidence["objects_world"] = {
                f"object_{object_id}": env.unwrapped.scene[f"object_{object_id}"].data.root_pos_w[0].detach().cpu().tolist()
                for object_id in (1, 2, 3)
            }
            terminal_evidence["termination_terms"] = {
                name: bool(manager.get_term(name)[0]) for name in manager.active_terms
            }
            if args_cli.camera_video:
                terminal_camera_frame = env.unwrapped.scene.sensors["video_cam"].data.output["rgb"][0].detach().cpu().numpy().copy()
        return original_reset(env_ids)

    env.unwrapped._reset_idx = audited_reset

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video and not args_cli.camera_video:
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(args_cli.output_dir or os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()
    camera_writer = None
    if args_cli.video and args_cli.camera_video:
        import imageio_ffmpeg
        camera_directory = args_cli.output_dir or os.path.join("logs", "videos", args_cli.task, "play")
        os.makedirs(camera_directory, exist_ok=True)
        frame = obs["image"]["video_rgb"][0].detach().cpu().numpy()
        camera_writer = imageio_ffmpeg.write_frames(
            os.path.join(camera_directory, "rgbd_success.mp4"),
            size=(frame.shape[1], frame.shape[0]), fps=round(1 / env.unwrapped.step_dt),
            codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
            output_params=["-movflags", "+faststart"],
        )
        camera_writer.send(None)
    print(f"[INFO] Environment reset complete. Observation keys: {list(obs.keys())}")
    _log_default_joint_pos_diagnostics(env, env_cfg)

    _robot = env.unwrapped.scene["robot"]
    _env_origins = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
    _root_pos_w = _robot.data.root_pos_w[0].detach().cpu().tolist()
    print(f"[DEBUG] env_origins[0]={_env_origins}")
    print(f"[DEBUG] root_pos_w[0] at reset={_root_pos_w}")
    print(f"[DEBUG] root_pos local to env_origin={[a - b for a, b in zip(_root_pos_w, _env_origins)]}")

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    termination_reason = "step_limit"
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()
            if camera_writer is not None:
                camera_writer.send(obs["image"]["video_rgb"][0].detach().cpu().numpy().copy())

            # ===== Your controller goes here =====
            resp = solution.predicts(obs, total_episode_reward)
            policy_duration = time.time() - start_time
            if os.environ.get("ATEC_FK_DIAGNOSTICS") == "1" and timestep % 100 == 0:
                import numpy as np
                from demo.piper_kinematics import fk
                robot = env.unwrapped.scene["robot"]
                body_index = list(robot.body_names).index("gripper_base")
                joint_positions = robot.data.joint_pos[0, :6].detach().cpu().numpy()
                expected = fk(joint_positions)
                base_rotation = np.diag([-1.0, -1.0, 1.0])
                predicted_world = base_rotation @ expected[:3, 3] + np.array([1.404254, 0.0, 0.8266427])
                measured_world = robot.data.body_pos_w[0, body_index].detach().cpu().numpy()
                print(f"[FK_CALIBRATION] predicted={predicted_world.tolist()} measured={measured_world.tolist()} error_m={np.linalg.norm(predicted_world - measured_world):.6f}", flush=True)
                calibration_sample = {}
                for object_id in (1, 2, 3):
                    measured_object = env.unwrapped.scene[f"object_{object_id}"].data.root_pos_w[0].detach().cpu().numpy()
                    calibration_sample[f"object_{object_id}_world"] = measured_object
                    estimated_object = solution.object_targets.get(object_id)
                    if estimated_object is not None:
                        delta = np.asarray(estimated_object) - measured_object
                        print(f"[VISION_CALIBRATION] object={object_id} estimated={estimated_object.tolist()} measured={measured_object.tolist()} delta_m={delta.tolist()}", flush=True)
                calibration_directory = os.environ.get("ATEC_CALIBRATION_OUTPUT")
                if calibration_directory:
                    os.makedirs(calibration_directory, exist_ok=True)
                    calibration_sample.update({name: value[0].detach().cpu().numpy() for name, value in obs.get("image", {}).items()})
                    calibration_sample["joint_positions"] = joint_positions
                    calibration_sample["gripper_joint_positions"] = robot.data.joint_pos[0, 6:8].detach().cpu().numpy()
                    calibration_sample["gripper_world"] = measured_world
                    np.savez_compressed(os.path.join(calibration_directory, f"sample_{timestep:05d}.npz"), **calibration_sample)
            giveup = resp["giveup"]
            if giveup:
                termination_reason = "policy_giveup"
                print("[INFO] Solution requested early stop via giveup flag.")
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device="cuda").view(1, -1)
            step_start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(actions)
            if timestep % 100 == 0:
                print(f"[PERF] step={timestep} policy_ms={1000 * policy_duration:.1f} simulation_ms={1000 * (time.time() - step_start_time):.1f}", flush=True)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                print(f"[DEBUG] total_episode_reward: {total_episode_reward:.2f}")
                print(f"[DEBUG] total_elapsed_time: {total_elapsed_time:.2f}")

            done = (terminated.item() or truncated.item())
            if done:
                termination_reason = _infer_termination_reason(env, env_cfg, terminated, truncated)
                if terminal_evidence.get("termination_terms"):
                    termination_reason = " | ".join(name for name, active in terminal_evidence["termination_terms"].items() if active)
                _log_terminal_state_diagnostics(env, env_cfg, terminated, truncated, obs)
                _log_termination_manager_debug(env, info)
                print(
                    "[INFO] Episode finished: "
                    f"terminated={terminated.item()}, truncated={truncated.item()}, "
                    f"score={total_episode_reward:.2f}, elapsed_time={total_elapsed_time:.2f}, "
                    f"termination_reason={termination_reason}"
                )
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    if camera_writer is not None:
        if terminal_camera_frame is not None:
            camera_writer.send(terminal_camera_frame)
        camera_writer.close()
    if args_cli.output_dir:
        os.makedirs(args_cli.output_dir, exist_ok=True)
        filenames = ["demo/solution_rgbd.py", "demo/piper_kinematics.py", "demo/rgbd/perception.py", "demo/rgbd_vbc_alore.py", "source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/rewards.py", "source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/terminations.py"]
        report = {
            "seed": args_cli.seed,
            "score": total_episode_reward,
            "elapsed_sim_seconds": total_elapsed_time,
            "steps": timestep,
            "termination_reason": termination_reason,
            "terminal_evidence": terminal_evidence,
            "policy": "RGBD deterministic FK/IK; no ACT checkpoint",
            "order": ["banana", "mustard_bottle", "sugar_box"],
            "sha256": {filename: hashlib.sha256(open(os.path.join(REPO_ROOT, filename), "rb").read()).hexdigest() for filename in filenames},
        }
        with open(os.path.join(args_cli.output_dir, "result.json"), "w") as report_file:
            json.dump(report, report_file, indent=2)
    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
