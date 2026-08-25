# thrusters.py
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
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from simulation.utils import wrap_angle_pi as wrap


@dataclass
class ThrusterConfig:
    """Thruster geometry and actuator limits used by the engine."""
    name: str
    kind: str       # "tunnel" or "azimuth"
    x: float        # position in BODY frame [m]
    y: float        # position in BODY frame [m]
    u_max: float    # maximum absolute thrust [N]
    u_rate: float   # maximum thrust rate [N/s]
    rot_speed: float  # maximum rotation speed [rad/s]; 0 for tunnels
    alpha0: float   # default/fixed direction [rad]

@dataclass
class ThrusterState:
    """Per-thruster dynamic state."""
    u: float       # thrust [N] (signed, along current alpha direction)
    alpha: float   # azimuth angle [rad] (BODY x-axis reference)

class ThrusterSet:
    """
    Rate-limited thrusters + total BODY wrench computation.

    - Tunnels: fixed alpha = alpha0; thrust can reverse sign; no rotation dynamics.
    - Azimuths: angle and thrust are rate-limited.

    Optional behaviors
    ------------------
    dynamics: if True (default), thrust and azimuth commands are rate-limited
      (u_rate, rot_speed) and thrust is saturated at u_max. This is a pure
      slew-rate model — no first-order lag — so the actual value ramps
      linearly toward the command. If False, the actuators are IDEAL:
      commands are applied exactly as requested — no rate limits, no
      saturation, no mechanical angle limits (tunnels keep their fixed
      direction, which is geometry, not a limit). Project Part 1 runs with
      dynamics off, so there respecting u_max is the allocation's job.
    min_rotation: if True (default False), for azimuths we pick the command
      (alpha_cmd, u_cmd) or (alpha_cmd-π, -u_cmd) that yields the *smaller*
      rotation from current alpha before applying rate limits. This reduces
      large 180° swings when the allocator flips direction. It is OFF by
      default: resolving the thrust/angle sign ambiguity is part of the
      thrust-allocation design (see the project text, Allocation Methods),
      so it belongs in YOUR allocator, not in the engine.

    Mechanical limits
    -----------------
    If ThrusterConfig has attributes alpha_min / alpha_max (rad), they will be
    enforced after slew — with dynamics on only; ideal actuators ignore them.
    If not present, angles are only wrapped to (-π, π].

    Caveats: the limits clip the *wrapped* angle, so the allowed sector must
    lie within (-π, π] and cannot span the ±π seam (e.g. "aft only",
    [90°, 270°], is not representable). min_rotation picks its flipped
    option (alpha-π, -u) *before* the limits are applied, so combine the two
    with care.

    # ThrusterConfig(..., alpha_min=-np.pi/2, alpha_max=+np.pi/2)

    Notes
    -----
    The wrench mapping follows:

        Alpha_surge = [cos alpha_i]
        Alpha_sway  = [sin alpha_i]
        Alpha_yaw   = [sin alpha_i * x_i - cos alpha_i * y_i]

        tau = [Alpha_surge; Alpha_sway; Alpha_yaw] * u

    where u is the vector of signed thrusts [N].
    """

    def __init__(self, thrusters: List[ThrusterConfig], *,
                 dynamics: bool = True, min_rotation: bool = False):
        self.cfgs: List[ThrusterConfig] = thrusters
        self.n: int = len(thrusters)
        self.dynamics = bool(dynamics)
        self.min_rotation = bool(min_rotation)
        self.state: List[ThrusterState] = [
            ThrusterState(u=0.0, alpha=th.alpha0) for th in thrusters
        ]

    # --------- public API ---------
    def reset(self) -> None:
        for s, th in zip(self.state, self.cfgs):
            s.u = 0.0
            s.alpha = float(th.alpha0)

    def __len__(self) -> int:
        return self.n

    def get_angles(self) -> np.ndarray:
        """Current actual angles [rad] for all thrusters."""
        return np.array([s.alpha for s in self.state], dtype=float)

    def get_thrusts(self) -> np.ndarray:
        """Current actual thrusts [N] (signed) for all thrusters."""
        return np.array([s.u for s in self.state], dtype=float)

    def get_states(self) -> List[ThrusterState]:
        """Deep-copy-ish snapshot of internal states (safe to read)."""
        return [ThrusterState(u=s.u, alpha=s.alpha) for s in self.state]

    # --------- core stepping ---------
    def step(
        self,
        u_cmd: np.ndarray,
        alpha_cmd: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance thruster dynamics one step.

        Parameters
        ----------
        u_cmd : (n,) array_like
            Commanded thrusts [N] (signed).
        alpha_cmd : (n,) array_like
            Commanded angles [rad]. Ignored for tunnels (alpha fixed = alpha0).
        dt : float
            Timestep [s] (>0).

        Returns
        -------
        u_act : (n,) ndarray
            Actual thrusts [N]: rate-limited and saturated with dynamics on,
            exactly the commands with ideal actuators.
        a_act : (n,) ndarray
            Actual angles [rad], wrapped to (-π, π].
        tau_actual : (3,) ndarray
            BODY wrench [Fx, Fy, Mz].
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        u_cmd = np.asarray(u_cmd, dtype=float).reshape(self.n)
        alpha_cmd = np.asarray(alpha_cmd, dtype=float).reshape(self.n)

        u_act = np.zeros(self.n)
        a_act = np.zeros(self.n)

        if not self.dynamics:
            # Ideal actuators (Part 1): commands are applied exactly as
            # requested — no rate limits, no saturation, no angle limits.
            for i, th in enumerate(self.cfgs):
                s = self.state[i]
                if (th.rot_speed is None) or (th.rot_speed == 0.0) or (th.kind.lower() == "tunnel"):
                    s.alpha = float(th.alpha0)
                else:
                    s.alpha = float(wrap(alpha_cmd[i]))
                s.u = float(u_cmd[i])
                u_act[i] = s.u
                a_act[i] = s.alpha
            return u_act, a_act, self._wrench_from(u_act, a_act)

        for i, th in enumerate(self.cfgs):
            s = self.state[i]

            # ---- Angle dynamics ----
            if (th.rot_speed is None) or (th.rot_speed == 0.0) or (th.kind.lower() == "tunnel"):
                # Tunnels: fixed direction
                s.alpha = float(th.alpha0)
                u_target = u_cmd[i]  # sign allowed for tunnels
            else:
                # Azimuth: choose minimal rotation option if enabled
                a_des = wrap(alpha_cmd[i])
                u_des = float(u_cmd[i])

                if self.min_rotation:
                    # Compare rotation cost for (a_des, u_des) vs (a_des-π, -u_des)
                    delta1 = abs(wrap(a_des - s.alpha))
                    a_alt = wrap(a_des - np.pi)
                    delta2 = abs(wrap(a_alt - s.alpha))
                    if delta2 < delta1:
                        a_des = a_alt
                        u_des = -u_des

                # Slew angle
                max_step = float(th.rot_speed) * dt
                delta = float(wrap(a_des - s.alpha))
                step = np.clip(delta, -max_step, max_step)
                s.alpha = wrap(s.alpha + step)

                # Enforce mechanical angle limits if present on config
                if hasattr(th, "alpha_min") and hasattr(th, "alpha_max"):
                    a_min = float(getattr(th, "alpha_min"))
                    a_max = float(getattr(th, "alpha_max"))
                    # clamp then wrap for consistency
                    s.alpha = float(np.clip(s.alpha, a_min, a_max))
                    s.alpha = wrap(s.alpha)

                u_target = u_des

            # ---- Thrust dynamics ----
            du = float(u_target - s.u)
            s.u += float(np.clip(du, -th.u_rate * dt, th.u_rate * dt))
            # hard saturation
            s.u = float(np.clip(s.u, -th.u_max, th.u_max))

            # log
            u_act[i] = s.u
            a_act[i] = s.alpha

        # ---- Compute actual BODY wrench ----
        tau_actual = self._wrench_from(u_act, a_act)
        return u_act, a_act, tau_actual

    # --------- internal mapping ---------
    def _wrench_from(self, u: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Compute BODY wrench from thrusts u and angles a using current geometry.
        """
        u = np.asarray(u, dtype=float).reshape(self.n)
        a = np.asarray(a, dtype=float).reshape(self.n)

        c = np.cos(a); s = np.sin(a)
        Fx = float(np.sum(u * c))
        Fy = float(np.sum(u * s))
        Mz = 0.0
        for i, th in enumerate(self.cfgs):
            Mz += float(u[i] * (s[i] * th.x - c[i] * th.y))
        return np.array([Fx, Fy, Mz], dtype=float)
