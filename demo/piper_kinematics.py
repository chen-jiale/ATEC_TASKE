"""Self-contained Piper FK and damped least-squares IK for submission use."""
from __future__ import annotations
import numpy as np

_JOINTS = [
    ((0.0, 0.0, 0.123), (1.0, 0.0, 0.0, 0.0), -150.00035, 124.21724),
    ((0.0, 0.0, 0.0), (0.048008, -0.048013, -0.705476, -0.705474), 0.0, 179.90874),
    ((0.28503, 0.0, 0.0), (0.623996, 0.0, 0.0, -0.781427), -169.99657, 0.0),
    ((-0.021984, -0.25075, 0.0), (0.707106, 0.707108, 0.0, 0.0), -99.98113, 99.98113),
    ((0.0, 0.0, 0.0), (0.707106, -0.707108, 0.0, 0.0), -69.90085, 69.90085),
    ((8.8e-05, -0.091, 0.0), (0.707106, 0.707108, 0.0, 0.0), -120.00027, 120.00027),
]
LIMITS = np.deg2rad(np.asarray([(j[2], j[3]) for j in _JOINTS], dtype=np.float64))


def _rotation(q):
    w, x, y, z = q; n = w*w + x*x + y*y + z*z; s = 2.0 / n
    return np.array([[1-s*(y*y+z*z), s*(x*y-w*z), s*(x*z+w*y)],
                     [s*(x*y+w*z), 1-s*(x*x+z*z), s*(y*z-w*x)],
                     [s*(x*z-w*y), s*(y*z+w*x), 1-s*(x*x+y*y)]])


def _static(pos, quat):
    out = np.eye(4); out[:3, :3] = _rotation(quat); out[:3, 3] = pos; return out


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


_STATIC = [_static(j[0], j[1]) for j in _JOINTS]


def fk(q):
    transform = np.eye(4)
    for joint, angle in zip(_STATIC, np.asarray(q, dtype=np.float64)):
        transform = transform @ joint @ _rotz(angle)
    return transform


def _rot_error(current, desired):
    return 0.5 * sum(np.cross(current[:, i], desired[:, i]) for i in range(3))


def fk_jacobian(joints):
    transform = np.eye(4)
    origins = []
    axes = []
    for joint, angle in zip(_STATIC, np.asarray(joints, dtype=np.float64)):
        transform = transform @ joint
        origins.append(transform[:3, 3].copy())
        axes.append(transform[:3, 2].copy())
        transform = transform @ _rotz(angle)
    axes = np.asarray(axes)
    linear = np.cross(axes, transform[:3, 3] - np.asarray(origins))
    return transform, np.concatenate((linear.T, axes.T), axis=0)


def ik(target, initial, iterations=100, damping=0.06, step_limit=0.08):
    q = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(iterations):
        current, jacobian = fk_jacobian(q)
        error = np.concatenate((target[:3, 3] - current[:3, 3],
                                                   _rot_error(current[:3, :3], target[:3, :3])))
        if np.linalg.norm(error[:3]) < 0.003 and np.linalg.norm(error[3:]) < 0.03:
            break
        lhs = jacobian @ jacobian.T + damping * damping * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(lhs, error)
        scale = np.max(np.abs(delta))
        if scale > step_limit: delta *= step_limit / scale
        q = np.clip(q + delta, LIMITS[:, 0], LIMITS[:, 1])
    return q
