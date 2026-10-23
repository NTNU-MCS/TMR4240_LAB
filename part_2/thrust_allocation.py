"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        n_z = sum(2 if thr.kind == "azimuth" else 1 for thr in thrusters)
        self.Be = np.zeros((3, n_z))
        col = 0
        for thr in thrusters:
            if thr.kind == "tunnel":
                self.Be[:, col] = [np.cos(thr.alpha0), np.sin(thr.alpha0),
                                   thr.x * np.sin(thr.alpha0) - thr.y * np.cos(thr.alpha0)]
                col += 1
            elif thr.kind == "azimuth":
                self.Be[:, col] = [1.0, 0.0, -thr.y]
                self.Be[:, col + 1] = [0.0, 1.0, thr.x]
                col += 2
            else:
                raise ValueError(f"unsupported thruster kind: {thr.kind!r}")

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)
        z = np.linalg.pinv(self.Be) @ np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]]
        magnitudes = []
        col = 0
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                magnitudes.append(abs(z[col]))
                col += 1
            else:
                magnitudes.append(float(np.hypot(z[col], z[col + 1])))
                col += 2

        ratios = [mag / thr.u_max for mag, thr in zip(magnitudes, self.thrusters)]
        scale = max(1.0, *ratios)
        z = z / scale
        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)
        current_angles = None if alpha_now is None else np.asarray(alpha_now, dtype=float).reshape(n)
        col = 0
        for i, thr in enumerate(self.thrusters):
            if thr.kind == "tunnel":
                u_cmd[i] = z[col]
                alpha_cmd[i] = thr.alpha0
                col += 1
            else:
                fx, fy = z[col], z[col + 1]
                thrust = float(np.hypot(fx, fy))
                angle = float(np.arctan2(fy, fx)) if thrust > 0.0 else thr.alpha0
                if current_angles is not None:
                    alternate = wrap_angle_pi(angle - np.pi)
                    if abs(wrap_angle_pi(alternate - current_angles[i])) < abs(
                            wrap_angle_pi(angle - current_angles[i])):
                        angle = alternate
                        thrust = -thrust
                u_cmd[i] = thrust
                alpha_cmd[i] = angle
                col += 2
        return u_cmd, alpha_cmd
