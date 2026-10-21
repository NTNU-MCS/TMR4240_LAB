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
from part_2.config import RefAxisConfig

# To avoid heading wrong direction
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
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
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
        # TODO: Replace this pass-through placeholder with your reference model.

        #Extract the frequency and damping coefficient from the configuration
        wn_xy = self.cfg_xy.wn
        zeta_xy = self.cfg_xy.zeta

        wn_psi = self.cfg_psi.wn
        zeta_psi = self.cfg_psi.zeta

        

        #Calculate the acceleration refrences
        self.acc_ref[0] = (wn_xy**2)*(eta_cmd[0]-self.eta_ref[0]) - 2*zeta_xy*wn_xy*self.nu_ref[0]
        self.acc_ref[1] = (wn_xy**2)*(eta_cmd[1]-self.eta_ref[1]) - 2*zeta_xy*wn_xy*self.nu_ref[1]
        self.acc_ref[5] = (wn_psi**2)*wrap_angle_pi(eta_cmd[5] - self.eta_ref[5]) - 2*zeta_psi*wn_psi*self.nu_ref[5]

        #Integrate acceleration to velocity, then velocity to position        
        self.nu_ref[0] = self.nu_ref[0] + dt*self.acc_ref[0]        
        self.eta_ref[0] = self.eta_ref[0] + dt*self.nu_ref[0]

        self.nu_ref[1] = self.nu_ref[1] + dt*self.acc_ref[1]        
        self.eta_ref[1] = self.eta_ref[1] + dt*self.nu_ref[1]

        self.nu_ref[5] = self.nu_ref[5] + dt*self.acc_ref[5]        
        self.eta_ref[5] = self.eta_ref[5] + dt*self.nu_ref[5]
        self.eta_ref[5] = wrap_angle_pi(self.eta_ref[5])
        
        return self.eta_ref, self.nu_ref, self.acc_ref
