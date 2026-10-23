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
from simulation.utils import body_to_ned_xy, ned_to_body_xy, wrap_angle_pi


class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):
        names = ("Kp", "Ki", "Kd", "tau_cap", "use_feedforward")
        defaults = {
            "Kp": np.array([24_028.00, 28_268.00, 545_600.00]),
            "Ki": np.array([480.56, 565.36, 5_456.00]),
            "Kd": np.array([240_280.00, 282_680.00, 10_912_000.00]),
            "tau_cap": np.array([160_000.0, 190_000.0, 860_000.0]),
            "use_feedforward": True,
        }
        if len(args) > len(names):
            raise TypeError(f"DPController takes at most {len(names)} positional arguments")
        values = dict(zip(names, args))
        unknown = kwargs.keys() - defaults.keys()
        if unknown:
            unexpected = next(iter(unknown))
            raise TypeError(f"DPController got an unexpected keyword argument {unexpected!r}")
        values.update(kwargs)
        defaults.update(values)
        self.Kp = np.asarray(defaults["Kp"], dtype=float)
        self.Ki = np.asarray(defaults["Ki"], dtype=float)
        self.Kd = np.asarray(defaults["Kd"], dtype=float)
        self.tau_cap = np.asarray(defaults["tau_cap"], dtype=float)
        self.use_feedforward = bool(defaults["use_feedforward"])
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self._tau_unsat = np.zeros(6)
        self.last_pid_body = {key: np.zeros(6) for key in ("P", "I", "D")}

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self._tau_unsat = np.zeros(6)
        self.last_pid_body = {key: np.zeros(6) for key in ("P", "I", "D")}

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
        if nu_ref is None:
            nu_ref = np.zeros(6)
        if acc_ref is None:
            acc_ref = np.zeros(6)

        psi = eta[5]
        e_ned = eta_ref[:2] - eta[:2]
        e_psi = wrap_angle_pi(eta_ref[5] - psi)
        e_body_xy = ned_to_body_xy(e_ned, psi)

        int_ned_prev, int_psi_prev = self.int_ned.copy(), self.int_psi 
        self.int_ned += e_ned * dt
        self.int_psi += e_psi * dt
        int_body_xy = ned_to_body_xy(self.int_ned, psi)

        nu_ref_body_xy = ned_to_body_xy(nu_ref[:2], psi)
        e_vel_body_xy = nu_ref_body_xy - nu[:2]
        e_vel_psi = nu_ref[5] - nu[5]
        if not self.use_feedforward:
            e_vel_body_xy = -nu[:2]
            e_vel_psi = -nu[5]

        p_xy = self.Kp[:2] * e_body_xy
        i_xy = self.Ki[:2] * int_body_xy
        d_xy = self.Kd[:2] * e_vel_body_xy
        p_psi = self.Kp[2] * e_psi
        i_psi = self.Ki[2] * self.int_psi
        d_psi = self.Kd[2] * e_vel_psi

        tau_d = np.zeros(6)
        tau_d[0], tau_d[1] = p_xy + i_xy + d_xy
        tau_d[5] = p_psi + i_psi + d_psi
        self._tau_unsat = tau_d.copy()
        saturated = np.abs(tau_d[[0, 1, 5]]) > self.tau_cap
        if saturated[0] or saturated[1]:
            self.int_ned = int_ned_prev
        if saturated[2]:
            self.int_psi = int_psi_prev
        self.last_pid_body = {
            "P": np.array([p_xy[0], p_xy[1], 0, 0, 0, p_psi]),
            "I": np.array([i_xy[0], i_xy[1], 0, 0, 0, i_psi]),
            "D": np.array([d_xy[0], d_xy[1], 0, 0, 0, d_psi]),
        }
        return tau_d
