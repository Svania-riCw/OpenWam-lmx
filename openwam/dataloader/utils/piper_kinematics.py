"""Piper forward kinematics used by the LeRobot training adapter.

The modified-DH parameters match AgileX's ``piper_sdk`` implementation. Joint
angles are radians and translations are returned in metres.
"""

from __future__ import annotations

import numpy as np

JOINT14_DIM = 14
EEF20_DIM = 20

_A = np.asarray([0.0, 0.0, 285.03, -21.98, 0.0, 0.0], dtype=np.float64)
_ALPHA = np.asarray([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2], dtype=np.float64)
_THETA_OFFSET = np.deg2rad([0.0, -172.22, -102.78, 0.0, 0.0, 0.0])
_D = np.asarray([123.0, 0.0, 0.0, 250.75, 0.0, 91.0], dtype=np.float64)


def piper_forward_kinematics(joints: np.ndarray) -> np.ndarray:
    """Return Piper tool transforms with shape ``(..., 4, 4)``.

    The input's last dimension must contain the six arm joints in radians.
    """
    values = np.asarray(joints)
    if values.ndim == 0 or values.shape[-1] != 6:
        raise ValueError(f"Piper joints must have shape (..., 6), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Piper joints contain NaN or infinity")

    original_shape = values.shape[:-1]
    flat = values.reshape(-1, 6).astype(np.float64, copy=False)
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (len(flat), 4, 4)).copy()

    for index in range(6):
        theta = flat[:, index] + _THETA_OFFSET[index]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(_ALPHA[index]), np.sin(_ALPHA[index])
        link = np.zeros((len(flat), 4, 4), dtype=np.float64)
        link[:, 0, 0] = ct
        link[:, 0, 1] = -st
        link[:, 0, 3] = _A[index]
        link[:, 1, 0] = st * ca
        link[:, 1, 1] = ct * ca
        link[:, 1, 2] = -sa
        link[:, 1, 3] = -sa * _D[index]
        link[:, 2, 0] = st * sa
        link[:, 2, 1] = ct * sa
        link[:, 2, 2] = ca
        link[:, 2, 3] = ca * _D[index]
        link[:, 3, 3] = 1.0
        result = result @ link

    result[:, :3, 3] *= 1e-3
    return result.reshape(*original_shape, 4, 4)


def _pose10(transform: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    rotation_6d = np.swapaxes(transform[..., :3, :2], -2, -1).reshape(*transform.shape[:-2], 6)
    translation = transform[..., :3, 3]
    return np.concatenate([translation, rotation_6d, gripper[..., None]], axis=-1)


def joint14_to_eef20(values: np.ndarray) -> np.ndarray:
    """Convert dual-Piper joint positions to Alpha's raw bimanual EEF20.

    Input layout is ``[L joints6, L grip, R joints6, R grip]``. Output layout
    is ``[L xyz3, L rot6d6, L grip, R xyz3, R rot6d6, R grip]``. Piper gripper
    angle increases with opening; subsequent min-max normalization therefore
    maps closed/open to Alpha's -1/+1 convention.
    """
    joint14 = np.asarray(values)
    if joint14.ndim == 0 or joint14.shape[-1] != JOINT14_DIM:
        raise ValueError(f"dual-Piper values must have shape (..., {JOINT14_DIM}), got {joint14.shape}")
    if not np.isfinite(joint14).all():
        raise ValueError("dual-Piper values contain NaN or infinity")

    left = _pose10(piper_forward_kinematics(joint14[..., :6]), joint14[..., 6])
    right = _pose10(piper_forward_kinematics(joint14[..., 7:13]), joint14[..., 13])
    return np.concatenate([left, right], axis=-1).astype(np.float32, copy=False)


__all__ = ["EEF20_DIM", "JOINT14_DIM", "joint14_to_eef20", "piper_forward_kinematics"]
