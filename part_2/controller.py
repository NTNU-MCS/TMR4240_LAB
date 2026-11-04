"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model.  Unlike Part 1, the
                                               Part 2 actuators saturate and
                                               rate-limit by default, so the
                                               applied wrench really does
                                               differ from the demanded one.
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py --part 2``,
``pytest``) build your controller with no arguments:

    DPController()

so your final tuned gains must be the ``__init__`` defaults.  Gains set only
in ``run_case_part_2.py`` reach your own runs but not the checks.
"""
import numpy as np

from part_2.config import ControllerConfig
from simulation.utils import wrap_angle_pi, ned_to_body_xy, body_to_ned_xy, to_3dof, to_6dof

# --- Control plant model, plant_model.ipynb Sec. 6 ---------------------
# M3: full rigid-body + added-mass inertia, [surge, sway, yaw]
M3 = np.array([
    [6.007e5,      0.0,       0.0],
    [0.0,      7.067e5,  -4.733e5],
    [0.0,     -5.712e5,   5.456e7],
])
# D3: linear damping only (diagonal)
D3 = np.array([
    [1117.6,     0.0,     0.0],
    [0.0,    2.229e4,     0.0],
    [0.0,        0.0,  1.95e6],
])

def design_gains(cfg: ControllerConfig):
    m = np.diag(M3)
    d = np.diag(D3)
    wc = np.asarray(cfg.wc, dtype=float)
    zeta = np.asarray(cfg.zeta_c, dtype=float)
    Ti = np.asarray(cfg.Ti, dtype=float)
    Kp = m * wc**2
    Kd = 2.0 * zeta * wc * m
    Ki = Kp / Ti
    return Kp, Kd, Ki


class DPController:
    """PID + feedforward DP controller, body-frame implementation."""
 
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.Kp, self.Kd, self.Ki = design_gains(self.cfg)
        Tt = np.array([self.cfg.Tt_pos, self.cfg.Tt_pos, self.cfg.Tt_psi])
        self.Kaw = 1.0 / (Tt * self.Ki)
        self.M = M3
        self.I_limit = np.asarray(self.cfg.I_limit, dtype=float)
        self._raw_int_safety = 1.0e6
        self.reset()
 
    def reset(self) -> None:
        """Called by the engine before each run."""
        self.int_ned = np.zeros(2)            # accumulated [N, E] error [m s]
        self.int_psi = 0.0                    # accumulated heading error [rad s]
        self._last_tau_unsat_3 = np.zeros(3)  # last [Fx, Fy, Mz] requested
        self._last_psi = 0.0
        self.last_pid_body = {"P": np.zeros(6), "I": np.zeros(6), "D": np.zeros(6)}

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        nu_ref = np.zeros(6) if nu_ref is None else nu_ref
        acc_ref = np.zeros(6) if acc_ref is None else acc_ref

        eta3 = to_3dof(eta)            # [N, E, psi]
        nu3 = to_3dof(nu)              # [u, v, r]
        eta_ref3 = to_3dof(eta_ref)    # [N_d, E_d, psi_d]
        nu_ref3 = to_3dof(nu_ref)      # [Ndot_d, Edot_d, psidot_d]
        acc_ref3 = to_3dof(acc_ref)    # [Nddot_d, Eddot_d, psiddot_d]
        psi = eta3[2]

        # Position / heading error, NED
        e_pos_ned = eta_ref3[:2] - eta3[:2]
        e_psi = wrap_angle_pi(eta_ref3[2] - psi)

        # Velocity error, BODY
        nu_ref_body_xy = ned_to_body_xy(nu_ref3[:2], psi)
        e_dot_body = np.array([nu_ref_body_xy[0], nu_ref_body_xy[1], nu_ref3[2]]) - nu3

        # Acceleration feedforward, BODY
        acc_ref_body_xy = ned_to_body_xy(acc_ref3[:2], psi)
        acc_ref_body = np.array([acc_ref_body_xy[0], acc_ref_body_xy[1], acc_ref3[2]])

        # Integrator update (NED, safety bound only)
        s = self._raw_int_safety
        self.int_ned = np.clip(self.int_ned + e_pos_ned * dt, -s, s)
        self.int_psi = float(np.clip(self.int_psi + e_psi * dt, -s, s))

        # Rotate error and integral to BODY, apply gains
        e_body_xy = ned_to_body_xy(e_pos_ned, psi)
        e_body = np.array([e_body_xy[0], e_body_xy[1], e_psi])
        i_body_xy = ned_to_body_xy(self.int_ned, psi)
        i_body = np.array([i_body_xy[0], i_body_xy[1], self.int_psi])

        P = self.Kp * e_body
        I = np.clip(self.Ki * i_body, -self.I_limit, self.I_limit)
        D = self.Kd * e_dot_body
        FF = self.M @ acc_ref_body if self.cfg.feedforward else np.zeros(3) # full M3, keeps sway-yaw coupling

        tau_3 = P + I + D + FF # BODY wrench [Fx, Fy, Mz]

        # Save for anti-windup comparison in apply_external_aw()
        self._last_tau_unsat_3 = tau_3.copy()
        self._last_psi = psi
        self.last_pid_body = {"P": to_6dof(P), "I": to_6dof(I), "D": to_6dof(D)}

        return to_6dof(tau_3)

    def apply_external_aw(self, tau_applied: np.ndarray, psi: float, dt: float) -> None:
        """Back-calculation anti-windup using the wrench actually applied
        after allocation and actuator dynamics.
 
            d(int)/dt = e + Kaw * (tau_sat - tau_unsat)
 
        tau_applied and the stored tau_unsat are both BODY. The correction
        is rotated back into the NED frame where the integrator lives.
        """
        if not self.cfg.anti_windup:
            return
        diff_body = to_3dof(tau_applied) - self._last_tau_unsat_3
        diff_ned_xy = body_to_ned_xy(diff_body[:2], psi)
        diff_ned = np.array([diff_ned_xy[0], diff_ned_xy[1], diff_body[2]])
        correction = self.Kaw * diff_ned * dt
        
        s = self._raw_int_safety

        self.int_ned = np.clip(self.int_ned + correction[:2], -s, s)
        self.int_psi = float(np.clip(self.int_psi + correction[2], -s, s))
        