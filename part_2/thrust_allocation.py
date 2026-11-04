"""
Thrust Allocation - pseudo-inverse with saturate-and-redistribute.

Solves B_e z = tau_c for the decision vector
z = [u_T, Fx_1, Fy_1, Fx_2, Fy_2]^T with the Moore-Penrose pseudo-inverse
(as in Part 1) and recovers the per-thruster commands. The template calls
thrust "u"; the report calls it f.

Design choices:
  * Saturation (new in Part 2): if a thruster exceeds u_max it is fixed at
    the limit, keeping the direction of its unconstrained solution, and the
    remaining wrench is re-solved on the free thrusters with the
    pseudo-inverse. This repeats until no free thruster saturates.
  * Infeasible demand: if redistribution cannot reproduce tau_c, the largest
    feasible s*tau_c (0 <= s <= 1) is found by bisection, so the delivered
    wrench keeps the requested direction.
  * Angle ambiguity: for each azimuth we pick between (alpha, +u) and
    (alpha + pi, -u) the representation nearest alpha_now.
  * Small-thrust hold: below u_min (2 kN, 2.5 % of the azimuth capacity)
    the azimuth angle is held at alpha_now and the thrust is the component
    of the demanded force along that direction, so the azimuths do not slew
    after small, rapidly changing force demands. Raised from 100 N in
    Part 1: with the 12 deg/s slew limit, the small demands of station
    keeping otherwise turned the azimuths continuously.
  * Thruster loss: set_available(i, False) removes thruster i from the
    allocation (its columns of B_e are dropped) and commands it zero thrust.

This is a steady-state allocator: thrust ramps and azimuth slew rates are
left to the actuator model (models/thruster_dynamics.py), so the applied
wrench lags the commanded one during transients.

Signed thrust is returned; the tunnel is always driven at alpha = pi/2.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi as wrap


class ThrustAllocator:
    """Saturate-and-redistribute allocator for the 1-tunnel + 2-azimuth layout."""

    def __init__(
        self,
        thrusters: List[ThrusterConfig],
        u_min: float = 2000.0,
        n_bisect: int = 20,
    ):
        self.thrusters = thrusters
        self.n = len(thrusters)
        self.u_min = float(u_min)      # angle-hold threshold [N]
        self.n_bisect = int(n_bisect)  # bisection steps for infeasible demands

        self.tunnel_idx: List[int] = []
        self.az_idx: List[int] = []
        for i, th in enumerate(thrusters):
            if th.kind.lower() == "tunnel":
                self.tunnel_idx.append(i)
            else:
                self.az_idx.append(i)

        if len(self.tunnel_idx) != 1 or len(self.az_idx) != 2:
            raise ValueError(
                "ThrustAllocator expects 1 tunnel and 2 azimuth thrusters "
                f"(got {len(self.tunnel_idx)} tunnel, {len(self.az_idx)} azimuth)."
            )

        i_T = self.tunnel_idx[0]
        j1, j2 = self.az_idx
        x_T = thrusters[i_T].x
        x1, y1 = thrusters[j1].x, thrusters[j1].y
        x2, y2 = thrusters[j2].x, thrusters[j2].y

        # Extended configuration matrix: B_e z = tau_c.
        # Tunnel at alpha_T = pi/2 contributes (0, 1, x_T) to (Fx, Fy, Mz);
        # each azimuth j contributes (1, 0, -y_j) via Fx_j and (0, 1, x_j) via Fy_j.
        self.B_e = np.array([
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [x_T, -y1,  x1,  -y2,  x2],
        ])

        # Entries of z belonging to each thruster (tunnel, azimuth 1, azimuth 2)
        # and their static thrust limits.
        self.groups = [[0], [1, 2], [3, 4]]
        self.u_max_T = float(thrusters[i_T].u_max)
        self.u_max_az = np.array(
            [thrusters[j1].u_max, thrusters[j2].u_max], dtype=float
        )
        self.limits = np.array([self.u_max_T, *self.u_max_az])

        self.alpha_T = float(thrusters[i_T].alpha0)  # pi/2
        self._pinv_cache: dict = {}
        # Thruster index (engine order) -> group in z, and which groups are usable
        self._group_of = {i_T: 0, j1: 1, j2: 2}
        self.available = [True, True, True]

        # Diagnostics from the last call (useful for logging / the report)
        self.last_scale = 1.0               # s in s * tau_c
        self.last_saturated: List[int] = []  # groups fixed at their limit

    def set_available(self, index: int, available: bool) -> None:
        """Mark thruster `index` (order of the thruster list) as available or
        lost, e.g. on a thruster fault alarm."""
        self.available[self._group_of[index]] = bool(available)

    def available_rank(self) -> int:
        """Rank of B_e restricted to the available thrusters (3 = all DOFs)."""
        cols = [c for g in range(3) if self.available[g] for c in self.groups[g]]
        return int(np.linalg.matrix_rank(self.B_e[:, cols])) if cols else 0

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 6-DOF -> 3-DOF wrench: surge, sway, yaw are indices 0, 1, 5.
        tau_c = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]]

        z, feasible, sat = self._redistribute(tau_c)
        s = 1.0
        if not feasible:
            # Largest feasible s * tau_c, so the delivered wrench keeps the
            # requested direction (s = 0 is always feasible).
            lo, hi = 0.0, 1.0
            z, sat = np.zeros(5), []
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                z_mid, ok, sat_mid = self._redistribute(mid * tau_c)
                if ok:
                    lo, z, sat = mid, z_mid, sat_mid
                else:
                    hi = mid
            s = lo
        self.last_scale = s
        self.last_saturated = sat

        u_cmd = np.zeros(self.n)
        alpha_cmd = np.zeros(self.n)

        # Tunnel: signed thrust, fixed angle
        i_T = self.tunnel_idx[0]
        u_cmd[i_T] = z[0]
        alpha_cmd[i_T] = self.alpha_T

        # Current azimuth angles for the closest-representation choice
        if alpha_now is None:
            a_now = np.array(
                [self.thrusters[j].alpha0 for j in self.az_idx], dtype=float
            )
        else:
            a_now = np.asarray(alpha_now, dtype=float)[self.az_idx]

        # Recover (u_j, alpha_j) from (Fx_j, Fy_j) for each azimuth
        for k, j in enumerate(self.az_idx):
            Fx = z[1 + 2 * k]
            Fy = z[2 + 2 * k]
            mag = float(np.hypot(Fx, Fy))

            if mag < self.u_min:
                # Small demand: hold the current angle and give the part of
                # the force that lies along it
                alpha_cmd[j] = float(a_now[k])
                u_cmd[j] = float(Fx * np.cos(a_now[k]) + Fy * np.sin(a_now[k]))
                continue

            alpha_pos = float(np.arctan2(Fy, Fx))
            alpha_neg = wrap(alpha_pos + np.pi)
            d_pos = abs(wrap(alpha_pos - float(a_now[k])))
            d_neg = abs(wrap(alpha_neg - float(a_now[k])))
            if d_neg < d_pos:
                alpha_cmd[j] = alpha_neg
                u_cmd[j] = -mag
            else:
                alpha_cmd[j] = alpha_pos
                u_cmd[j] = mag

        return u_cmd, alpha_cmd

    def _redistribute(self, tau: np.ndarray) -> Tuple[np.ndarray, bool, List[int]]:
        """Pseudo-inverse with saturate-and-redistribute.

        Returns z, whether B_e z = tau is met within the limits, and the
        groups that were fixed at their limit.
        """
        z = np.zeros(5)
        free = [g for g in range(3) if self.available[g]]  # lost thrusters stay at 0
        fixed: List[int] = []
        while free:
            cols = [c for g in free for c in self.groups[g]]
            z[cols] = 0.0
            # Wrench left for the free thrusters after the saturated ones
            tau_r = tau - self.B_e @ z
            z[cols] = self._pinv(tuple(cols)) @ tau_r

            ratios = [np.linalg.norm(z[self.groups[g]]) / self.limits[g] for g in free]
            worst = int(np.argmax(ratios))
            if ratios[worst] <= 1.0 + 1e-9:
                break
            # Fix the most saturated thruster at its limit, same direction
            g = free.pop(worst)
            z[self.groups[g]] *= 1.0 / ratios[worst]
            fixed.append(g)

        feasible = bool(np.allclose(self.B_e @ z, tau, rtol=1e-6, atol=1e-3))
        return z, feasible, fixed

    def _pinv(self, cols: Tuple[int, ...]) -> np.ndarray:
        """Pseudo-inverse of the columns of B_e for the free thrusters (cached)."""
        if cols not in self._pinv_cache:
            self._pinv_cache[cols] = np.linalg.pinv(self.B_e[:, list(cols)])
        return self._pinv_cache[cols]
