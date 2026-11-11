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

======================================================================

New in Part 2: the reference model is now a third-order system per axis, so the
reference acceleration starts at zero after a setpoint step.

Model per axis (Fossen 2021, Sec 12.1 third-order reference model):
 
    eta_d / r (s) = w^3 / (s^3 + (2z+1) w s^2 + (2z+1) w^2 s + w^3)
                  = [w / (s + w)] * [w^2 / (s^2 + 2 z w s + w^2)]
 
implemented as the two factors in series:
 
    Stage 1, low-pass prefilter:      x_f'    = w (r - x_f)
    Stage 2, mass-damper-spring:      eta_d'' = w^2 (x_f - eta_d) - 2 z w eta_d'

"""

from typing import Tuple
import numpy as np

from part_2.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class ReferenceModel:
    def __init__(self, dt, cfg_xy=None, cfg_psi=None):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)
        self.x_f = np.zeros(6)

    def reset(self, eta0):
        eta0 = np.asarray(eta0, dtype=float).reshape(6).copy()
        eta0[5] = wrap_angle_pi(eta0[5])
        self.eta_ref = eta0.copy()
        self.x_f = eta0.copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)
    
    def step(
            self, t: float, dt: float, eta_cmd: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
            eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)
    
            for i, cfg in ((0, self.cfg_xy), (1, self.cfg_xy), (5, self.cfg_psi)):
                wn, zeta = cfg.wn, cfg.zeta
                err = wrap_angle_pi if i == 5 else (lambda x: x)
    
                # Stage 1: low-pass prefilter,  x_f' = wn (r - x_f)
                if cfg.prefilter:
                    self.x_f[i] += dt * wn * err(eta_cmd[i] - self.x_f[i])
                else:
                    self.x_f[i] = eta_cmd[i]             # Part 1: no prefilter
    
                # Stage 2: mass-damper-spring towards x_f (as in Part 1)
                acc = wn**2 * err(self.x_f[i] - self.eta_ref[i]) - 2.0 * zeta * wn * self.nu_ref[i]
                vel = self.nu_ref[i] + dt * acc
    
                # Velocity limit
                if cfg.rate_limit is not None:
                    vel = float(np.clip(vel, -cfg.rate_limit, cfg.rate_limit))
                    acc = (vel - self.nu_ref[i]) / dt   # Recalculate acc to match the limited vel
    
                self.eta_ref[i] += dt * vel
                self.nu_ref[i] = vel
                self.acc_ref[i] = acc
    
            # Keep both heading states in (-pi, pi]
            self.x_f[5] = wrap_angle_pi(self.x_f[5])
            self.eta_ref[5] = wrap_angle_pi(self.eta_ref[5])
    
            return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()


