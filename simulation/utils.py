# utils.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System 
#
# Author: Saber Sakhrieh
# Date: 10 June 2026
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Utility functions for coordinate transforms and angle handling.

Conventions
-----------
- Frames:
    BODY = vessel-fixed frame (x: surge, y: sway, z: heave)
    NED  = North-East-Down (x: North, y: East, z: Down)
- Yaw is in radians.
- Rotation convention used throughout the project:
    NED = Rz(psi) @ BODY
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------
def Rz(psi: float) -> np.ndarray:
    """
    Rotation about z-axis.

    Parameters
    ----------
    psi : float
        Yaw angle [rad].

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix such that: NED = Rz(psi) @ BODY
    """
    c = float(np.cos(psi))
    s = float(np.sin(psi))
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def body_to_ned_xy(v_body_xy: np.ndarray, psi: float) -> np.ndarray:
    """
    Rotate a 2D vector from BODY to NED using yaw.

    v_ned_xy = Rz(psi)[:2,:2] @ v_body_xy
    """
    v = np.asarray(v_body_xy, dtype=float).reshape(2)
    c, s = float(np.cos(psi)), float(np.sin(psi))
    return np.array([c*v[0] - s*v[1],
                     s*v[0] + c*v[1]], dtype=float)

def ned_to_body_xy(v_ned_xy: np.ndarray, psi: float) -> np.ndarray:
    """
    Rotate a 2D vector from NED to BODY using yaw.

    v_body_xy = Rz(psi).T[:2,:2] @ v_ned_xy
    """
    v = np.asarray(v_ned_xy, dtype=float).reshape(2)
    c, s = float(np.cos(psi)), float(np.sin(psi))
    return np.array([ c*v[0] + s*v[1],
                     -s*v[0] + c*v[1]], dtype=float)

# ---------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------
def wrap_angle_pi(angle: float) -> float:
    """
    Wrap an angle to (-π, π].

    Notes
    -----
    This uses a modulo that returns -π as +π for numerical consistency.
    """
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    # normalize -pi -> +pi for a unique representation
    if np.isclose(wrapped, -np.pi):
        wrapped = np.pi
    return float(wrapped)

def wrap_angle_2pi(angle: float) -> float:
    """Wrap an angle to [0, 2π)."""
    return float(angle % (2.0 * np.pi))

def angle_diff(a: float, b: float) -> float:
    """
    Signed shortest difference a - b wrapped to (-π, π].

    Positive result means rotate CCW from b to a.
    """
    return wrap_angle_pi(a - b)

def lerp_angle(a0: float, a1: float, s: float, mode: str = "shortest") -> float:
    """
    Interpolate between two angles.

    Parameters
    ----------
    a0, a1 : float
        Start and end angles [rad].
    s : float
        Fraction in [0,1].
    mode : {"shortest","cw","ccw"}
        - "shortest": follow the minimal angular distance
        - "cw":       increasing angle direction (NED: + toward East)
        - "ccw":      decreasing angle direction

    Returns
    -------
    float
        Interpolated angle [rad], wrapped to (-π, π].
    """
    a0 = float(a0); a1 = float(a1)
    s = float(np.clip(s, 0.0, 1.0))
    if mode == "shortest":
        d = wrap_angle_pi(a1 - a0)
        return wrap_angle_pi(a0 + s * d)
    elif mode == "cw":
        d = (a1 - a0) % (2*np.pi)
        return wrap_angle_pi(a0 + s * d)
    elif mode == "ccw":
        d = -((a0 - a1) % (2*np.pi))
        return wrap_angle_pi(a0 + s * d)
    else:
        raise ValueError("mode must be 'shortest', 'cw', or 'ccw'")

# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------
def saturate(x: float, xmin: float, xmax: float) -> float:
    """Clamp scalar x to [xmin, xmax]."""
    return float(np.clip(x, xmin, xmax))

def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """
    Clip a vector's Euclidean norm to max_norm (no change if <= max_norm).
    """
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or n <= max_norm:
        return v
    return v * (max_norm / n)

def near_zero(x: float, atol: float = 1e-12) -> bool:
    """True if |x| <= atol."""
    return bool(abs(x) <= atol)


# --- 3-DOF <-> 6-DOF convention ------------------------------------------------
# Project-wide convention: generalized 6-DOF vectors are ordered
# [surge, sway, heave, roll, pitch, yaw]. The horizontal-plane 3-DOF model
# uses the components selected by SURGE_SWAY_YAW.
SURGE_SWAY_YAW = np.array([0, 1, 5])


def to_3dof(v6) -> np.ndarray:
    """Reduce a generalized 6-DOF vector to 3-DOF [surge, sway, yaw]."""
    v6 = np.asarray(v6, dtype=float).reshape(6)
    return v6[SURGE_SWAY_YAW].copy()


def to_6dof(v3) -> np.ndarray:
    """Expand a 3-DOF vector [surge, sway, yaw] into a generalized 6-DOF vector."""
    v3 = np.asarray(v3, dtype=float).reshape(3)
    v6 = np.zeros(6)
    v6[SURGE_SWAY_YAW] = v3
    return v6
