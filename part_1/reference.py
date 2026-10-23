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

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig
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
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig(wn=0.15, zeta=1.0)
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig(wn=0.08, zeta=1.0)
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    @staticmethod
    def filter_axis(pos: float, vel: float, sp: float, cfg: RefAxisConfig, dt: float):
        """ One euler step for second order per single scalar axis.
        This is called in step function to solve per axis.
        1. Fetches zeta and wn from refAxisConfig
        2. Solves differential equation from project task description, equation 2.
        3. Integrates acceleration to get new velocity, then velocity to get new position.
        If we want to have a rate limit there is also added a function to keep velocity in range.
        """
        # 1
        wn, zeta = cfg.wn, cfg.zeta
        # 2
        acc = wn**2 * (sp-pos) - 2 * zeta * wn * vel
        # 3
        vel_new = vel + acc * dt
        if cfg.rate_limit is not None:
            vel_new = float(np.clip(vel_new, -cfg.rate_limit, cfg.rate_limit))
        pos_new = pos+ vel_new * dt
        return pos_new, vel_new, acc

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Advance the reference filter one time step.

        Runs the second-order filter (filter_axis) independently on each of
        the three controlled axes (N, E, psi), and packs the results into
        6-DOF NED-frame vectors.

        The heading (psi) is handled specially: the commanded setpoint is
        unwrapped relative to the filter's own current state before
        filtering (so the filter always takes the shortest path and never
        jumps by 2*pi), and the result is wrapped back to (-pi, pi] after
        filtering.

         Args:   
            t       : current simulation time [s] (unused, kept for interface
                    consistency with the other Part 1 modules)    
            dt      : time step [s]
            eta_cmd : (6,) raw, unfiltered commanded setpoint in NED
                    (N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

        Returns:
            eta_ref : (6,) filtered reference position/heading, NED
            nu_ref  : (6,) filtered reference velocity, NED
            acc_ref : (6,) filtered reference acceleration, NED
        """
        eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        # N (north / surge direction in NED)
        N_new, Ndot_new, Nddot = self.filter_axis(self.eta_ref[0],
                                                self.nu_ref[0], 
                                                eta_cmd[0], 
                                                self.cfg_xy, 
                                                dt)
        
        # E (east / sway direction in NED)
        E_new, Edot_new, Eddot = self.filter_axis(self.eta_ref[1],
                                                        self.nu_ref[1], 
                                                        eta_cmd[1], 
                                                        self.cfg_xy, 
                                                        dt)

        # psi, heading
        # Unwap the commanded heading relative to the filters own
        # current state before fitlering to avoid the filter
        # chasing the long way / jumping by 2pi when sp is near +- pi.
        psi_sp_unwrapped = self.eta_ref[5] + wrap_angle_pi(eta_cmd[5] - self.eta_ref[5])
        psi_new, psidot_new, psiddot = self.filter_axis(self.eta_ref[5],
                                                        self.nu_ref[5], 
                                                        psi_sp_unwrapped, 
                                                        self.cfg_psi, 
                                                        dt)
        psi_new = wrap_angle_pi(psi_new)
        self.eta_ref = np.zeros(6)
        self.eta_ref[0], self.eta_ref[1], self.eta_ref[5] = N_new, E_new, psi_new

        self.nu_ref = np.zeros(6)
        self.nu_ref[0], self.nu_ref[1], self.nu_ref[5] = Ndot_new, Edot_new, psidot_new

        self.acc_ref = np.zeros(6)
        self.acc_ref[0], self.acc_ref[1], self.acc_ref[5] = Nddot, Eddot, psiddot

        return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()
