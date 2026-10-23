"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np

# Per-axis tuning parameters live with the rest of the Part 2 configuration.
from part_2.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class ReferenceModel:
    """
    Template for student reference model.

    The default implementation is pass-through, so eta_ref = eta_cmd and the
    reference velocities/accelerations are zero.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig(wn=0.06)
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)
        n_new, ndot_new, nddot = self.filter_axis(
            self.eta_ref[0], self.nu_ref[0], eta_cmd[0], self.cfg_xy, dt)
        e_new, edot_new, eddot = self.filter_axis(
            self.eta_ref[1], self.nu_ref[1], eta_cmd[1], self.cfg_xy, dt)
        psi_sp = self.eta_ref[5] + wrap_angle_pi(eta_cmd[5] - self.eta_ref[5])
        psi_new, psidot_new, psiddot = self.filter_axis(
            self.eta_ref[5], self.nu_ref[5], psi_sp, self.cfg_psi, dt)
        psi_new = wrap_angle_pi(psi_new)

        self.eta_ref = np.zeros(6)
        self.eta_ref[0], self.eta_ref[1], self.eta_ref[5] = n_new, e_new, psi_new
        self.nu_ref = np.zeros(6)
        self.nu_ref[0], self.nu_ref[1], self.nu_ref[5] = ndot_new, edot_new, psidot_new
        self.acc_ref = np.zeros(6)
        self.acc_ref[0], self.acc_ref[1], self.acc_ref[5] = nddot, eddot, psiddot
        return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()

    @staticmethod
    def filter_axis(pos: float, vel: float, sp: float, cfg: RefAxisConfig, dt: float):
        acc = cfg.wn**2 * (sp - pos) - 2.0 * cfg.zeta * cfg.wn * vel
        vel_new = vel + acc * dt
        if cfg.rate_limit is not None:
            vel_new = float(np.clip(vel_new, -cfg.rate_limit, cfg.rate_limit))
        return pos + vel_new * dt, vel_new, acc
