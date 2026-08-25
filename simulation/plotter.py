# simulation/plotter.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Plotting utilities for the DP simulator.

Every function takes the ``Logs`` object returned by ``DPSimulator3DOF.run``
and returns the created matplotlib figure. Generalized vectors in the logs
are 6-DOF; these plots show the 3-DOF components [surge, sway, yaw]
(indices [0, 1, 5]).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def plot_time_histories(logs, figsize=(11, 7)):
    """Position/heading vs. setpoint (left) and BODY velocities (right)."""
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    t = logs.t

    pos = [("North [m]", logs.eta[:, 0], logs.sp[:, 0]),
           ("East [m]", logs.eta[:, 1], logs.sp[:, 1]),
           ("Heading [deg]", np.degrees(_wrap_pi(logs.eta[:, 5])),
            np.degrees(_wrap_pi(logs.sp[:, 5])))]
    for ax, (label, y, y_sp) in zip(axes[:, 0], pos):
        ax.plot(t, y, label="actual")
        ax.plot(t, y_sp, "--", label="setpoint")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")

    vel = [("u [m/s]", logs.nu[:, 0], logs.nu_ref[:, 0]),
           ("v [m/s]", logs.nu[:, 1], logs.nu_ref[:, 1]),
           ("r [deg/s]", np.degrees(logs.nu[:, 5]), np.degrees(logs.nu_ref[:, 5]))]
    for ax, (label, y, y_ref) in zip(axes[:, 1], vel):
        ax.plot(t, y, label="actual")
        if not np.all(np.isnan(y_ref)):
            ax.plot(t, y_ref, "--", label="reference")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Position, heading, and velocities")
    fig.tight_layout()
    return fig


def plot_xy(logs, figsize=(6, 6)):
    """North–East trajectory with start/end markers and setpoints."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(logs.eta[:, 1], logs.eta[:, 0], label="trajectory")
    ax.plot(logs.eta[0, 1], logs.eta[0, 0], "o", label="start")
    ax.plot(logs.eta[-1, 1], logs.eta[-1, 0], "s", label="end")
    # commanded setpoints (logs.cmd), not the smoothed reference (logs.sp) —
    # with a real reference model sp traces the whole path
    sp = np.unique(logs.cmd[:, [1, 0]], axis=0)
    ax.plot(sp[:, 0], sp[:, 1], "x", ms=10, label="setpoint")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Horizontal-plane trajectory")
    fig.tight_layout()
    return fig


def plot_thrusters(logs, figsize=(10, 6)):
    """Actual thrust and azimuth angle for each thruster."""
    fig, (ax_u, ax_a) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for i, name in enumerate(logs.thruster_names):
        ax_u.plot(logs.t, logs.u[:, i] * 1e-3, label=name)
        ax_a.plot(logs.t, np.degrees(logs.alpha[:, i]), label=name)
    ax_u.set_ylabel("Thrust [kN]")
    ax_a.set_ylabel("Azimuth [deg]")
    ax_a.set_xlabel("Time [s]")
    for ax in (ax_u, ax_a):
        ax.grid(True, alpha=0.3)
    ax_u.legend(loc="best")
    fig.suptitle("Thrusters")
    fig.tight_layout()
    return fig


def plot_wrench(logs, figsize=(10, 7)):
    """Commanded vs. applied BODY wrench (surge, sway, yaw)."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    rows = [("Fx [kN]", 0, 1e-3), ("Fy [kN]", 1, 1e-3), ("Mz [kNm]", 5, 1e-3)]
    for ax, (label, idx, scale) in zip(axes, rows):
        ax.plot(logs.t, logs.tau_d[:, idx] * scale, label="commanded (controller)")
        ax.plot(logs.t, logs.tau_thr[:, idx] * scale, "--", label="applied (thrusters)")
        ax.plot(logs.t, logs.tau_total[:, idx] * scale, ":", label="total on hull")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("BODY wrench")
    fig.tight_layout()
    return fig


def plot_current(logs, figsize=(10, 6)):
    """Ambient current: speed, direction, and BODY-frame components."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[0].plot(logs.t, logs.Uc)
    axes[0].set_ylabel("Speed [m/s]")
    axes[1].plot(logs.t, np.degrees(_wrap_pi(logs.beta_c)))
    axes[1].set_ylabel("Direction\n(towards) [deg]")
    axes[2].plot(logs.t, logs.cur_body[:, 0], label="$u_c$ (surge)")
    axes[2].plot(logs.t, logs.cur_body[:, 1], label="$v_c$ (sway)")
    axes[2].set_ylabel("BODY comp. [m/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Ambient current")
    fig.tight_layout()
    return fig


def plot_wind(logs, figsize=(10, 6)):
    """Wind: speed, directions, and the resulting BODY loads."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[0].plot(logs.t, logs.U_w)
    axes[0].set_ylabel("Speed [m/s]")
    axes[1].plot(logs.t, np.degrees(_wrap_pi(logs.beta_w)), label="NED (towards)")
    axes[1].plot(logs.t, np.degrees(_wrap_pi(logs.alpha_w)), "--", label="relative (BODY)")
    axes[1].set_ylabel("Direction [deg]")
    axes[1].legend(loc="best")
    axes[2].plot(logs.t, logs.tau_w6[:, 0] * 1e-3, label="Fx [kN]")
    axes[2].plot(logs.t, logs.tau_w6[:, 1] * 1e-3, label="Fy [kN]")
    axes[2].plot(logs.t, logs.tau_w6[:, 5] * 1e-3, label="Mz [kNm]")
    axes[2].set_ylabel("BODY loads [kN, kNm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Wind")
    fig.tight_layout()
    return fig


def plot_dashboard(logs, figsize=(13, 8)):
    """Compact overview: trajectory, tracking errors, wrench, and thrust."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    t = logs.t

    ax = axes[0, 0]
    ax.plot(logs.eta[:, 1], logs.eta[:, 0])
    ax.plot(logs.eta[0, 1], logs.eta[0, 0], "o")
    sp = np.unique(logs.cmd[:, [1, 0]], axis=0)   # commanded, not reference
    ax.plot(sp[:, 0], sp[:, 1], "x", ms=10)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectory")

    ax = axes[0, 1]
    ax.plot(t, logs.eta[:, 0] - logs.sp[:, 0], label="N err [m]")
    ax.plot(t, logs.eta[:, 1] - logs.sp[:, 1], label="E err [m]")
    ax.set_title("Position error"); ax.legend(loc="best")

    ax = axes[0, 2]
    ax.plot(t, np.degrees(_wrap_pi(logs.eta[:, 5] - logs.sp[:, 5])))
    ax.set_title("Heading error [deg]")

    ax = axes[1, 0]
    for i, name in enumerate(logs.thruster_names):
        ax.plot(t, logs.u[:, i] * 1e-3, label=name)
    ax.set_title("Thrust [kN]"); ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, logs.tau_d[:, 0] * 1e-3, label="Fx cmd [kN]")
    ax.plot(t, logs.tau_d[:, 1] * 1e-3, label="Fy cmd [kN]")
    ax.plot(t, logs.tau_d[:, 5] * 1e-3, label="Mz cmd [kNm]")
    ax.set_title("Controller wrench"); ax.legend(loc="best", fontsize=8)

    ax = axes[1, 2]
    ax.plot(t, logs.nu[:, 0], label="u [m/s]")
    ax.plot(t, logs.nu[:, 1], label="v [m/s]")
    ax.plot(t, np.degrees(logs.nu[:, 5]), label="r [deg/s]")
    ax.set_title("BODY velocities"); ax.legend(loc="best", fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")
    fig.tight_layout()
    return fig
