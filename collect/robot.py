"""Minimal 6-axis industrial-arm kinematics for the weld cell simulator.

Standard DH model of a generic 6R welding robot (~1.4 m reach, spherical wrist)
plus a straight MIG/TIG torch. Inverse kinematics is damped least squares with
warm starts, which is plenty for the small, smooth weave trajectories we need.
"""
from __future__ import annotations

import numpy as np

# (a [m], alpha [rad], d [m], theta offset [rad])
DH = np.array([
    [0.150, -np.pi / 2, 0.450, 0.0],
    [0.610, 0.0, 0.000, -np.pi / 2],
    [0.115, -np.pi / 2, 0.000, 0.0],
    [0.000, np.pi / 2, 0.640, 0.0],
    [0.000, -np.pi / 2, 0.000, 0.0],
    [0.000, 0.0, 0.100, 0.0],
])
TORCH_LEN_M = 0.38          # flange -> contact tip / TCP
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
Q_HOME = np.array([0.0, 0.35, 0.25, 0.0, 1.05, 0.0])


def _dh(a, alpha, d, theta):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics: 4x4 TCP pose in the robot base frame."""
    T = np.eye(4)
    for i in range(6):
        a, alpha, d, off = DH[i]
        T = T @ _dh(a, alpha, d, q[i] + off)
    tool = np.eye(4)
    tool[2, 3] = TORCH_LEN_M
    return T @ tool


def _pose_err(T, p_t, R_t):
    dp = p_t - T[:3, 3]
    R = T[:3, :3]
    dr = 0.5 * (np.cross(R[:, 0], R_t[:, 0]) + np.cross(R[:, 1], R_t[:, 1])
                + np.cross(R[:, 2], R_t[:, 2]))
    return np.concatenate([dp, dr])


def ik(p_t: np.ndarray, R_t: np.ndarray, q0: np.ndarray, iters: int = 50,
       tol: float = 1e-6, lam: float = 1e-3) -> tuple[np.ndarray, float]:
    """Damped-least-squares IK. Returns (q, final error norm)."""
    q = q0.astype(float).copy()
    eps = 1e-6
    err = np.inf
    for _ in range(iters):
        T = fk(q)
        e = _pose_err(T, p_t, R_t)
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        J = np.zeros((6, 6))
        for j in range(6):
            dq = q.copy()
            dq[j] += eps
            J[:, j] = (_pose_err(T, p_t, R_t) - _pose_err(fk(dq), p_t, R_t)) / eps
        JJt = J @ J.T + lam * np.eye(6)
        q = q + J.T @ np.linalg.solve(JJt, e)
    return q, err


def torch_rotation(push_deg: float = 10.0, work_deg: float = 0.0) -> np.ndarray:
    """Tool frame: z = torch approach (pointing at the work), tilted by push/work angles."""
    ph, wk = np.radians(push_deg), np.radians(work_deg)
    a = np.array([np.sin(ph), np.sin(wk), -np.cos(ph) * np.cos(wk)])
    a /= np.linalg.norm(a)
    o = np.array([0.0, 1.0, 0.0])
    o = o - a * (o @ a)
    o /= np.linalg.norm(o)
    n = np.cross(o, a)
    return np.column_stack([n, o, a])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def fk_batch(Q: np.ndarray) -> np.ndarray:
    """Vectorised FK for joint arrays (N,6) -> (N,4,4)."""
    n = len(Q)
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(6):
        a, alpha, d, off = DH[i]
        th = Q[:, i] + off
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(alpha), np.sin(alpha)
        A = np.zeros((n, 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = ct, -st * ca, st * sa, a * ct
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st, ct * ca, -ct * sa, a * st
        A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = sa, ca, d
        A[:, 3, 3] = 1.0
        T = T @ A
    tool = np.eye(4)
    tool[2, 3] = TORCH_LEN_M
    return T @ tool


def _pose_err_batch(T, P_t, R_t):
    dp = P_t - T[:, :3, 3]
    R = T[:, :3, :3]
    dr = 0.5 * sum(np.cross(R[:, :, k], R_t[:, k][None, :]) for k in range(3))
    return np.concatenate([dp, dr], axis=1)


def ik_batch(P_t: np.ndarray, R_t: np.ndarray, q0: np.ndarray, iters: int = 8,
             lam: float = 1e-4) -> np.ndarray:
    """Batched DLS IK for many nearby targets sharing one orientation."""
    n = len(P_t)
    Q = np.tile(q0, (n, 1)).astype(float)
    eps = 1e-6
    for _ in range(iters):
        T = fk_batch(Q)
        e = _pose_err_batch(T, P_t, R_t)
        J = np.zeros((n, 6, 6))
        for j in range(6):
            dQ = Q.copy()
            dQ[:, j] += eps
            J[:, :, j] = (e - _pose_err_batch(fk_batch(dQ), P_t, R_t)) / eps
        JJt = J @ np.swapaxes(J, 1, 2) + lam * np.eye(6)
        Q = Q + np.einsum("nji,nj->ni", J, np.linalg.solve(JJt, e[..., None])[..., 0])
    return Q


def solve_trajectory(positions: np.ndarray, R_t: np.ndarray,
                     q_start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IK along a TCP path. Returns joints (N,6), achieved tcp xyz (N,3), quats (N,4).

    The weld trajectories are small (mm-scale weave around a fixed point), so all
    samples are solved in one batch from a common seed, then FK gives the exact
    achieved TCP pose.
    """
    q = ik(positions[0], R_t, Q_HOME if q_start is None else q_start, iters=200)[0]
    Q = ik_batch(positions, R_t, q)
    T = fk_batch(Q)
    P = T[:, :3, 3]
    Qt = np.array([rot_to_quat(R) for R in T[:, :3, :3]])
    return Q, P, Qt
