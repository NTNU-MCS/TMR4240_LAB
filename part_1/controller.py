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
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
from simulation.utils import wrap_angle_pi, ned_to_body_xy, body_to_ned_xy


class DPController:

    """
    PID dynamic positioning controller, following the approach
    recommended in Section 6.1 of the project task description.
    Computes the surge force, sway force and yaw moment needed to
    drive the vessel from its actual position toward the reference
    position produced by the reference model. The position error is
    computed in the NED frame and then rotated into the body frame
    using J transpose of psi before it becomes a force command, since
    the thrusters act in the body frame while the vessel position and
    the setpoint are both defined in NED. 
    
    The heading error is always computed with atan2 so that it wraps
    correctly to the interval from minus pi to pi, as required in
    Section 6. The gains used here follow the frequency domain design
    method from Section 6.2, using the Gunnerus vessel mass and yaw
    inertia from the process plant model to compute Kp equals m times
    omega_c squared and Kd equals 2 times zeta_c times omega_c times
    m, with a chosen closed loop bandwidth omega_c and critical
    damping zeta_c equal to 1.
    """

    def __init__(
        self,
        Kp: np.ndarray = np.array([24_028.00, 28_268.00, 545_600.00]),  # surge, sway, yaw
        Ki: np.ndarray = np.array([480.56, 565.36, 5_456.00]),
        Kd: np.ndarray = np.array([240_280.00, 282_680.00, 10_912_000.00]),
        Kaw: np.ndarray = np.array([0.2, 0.2, 0.2]),
        use_feedforward: bool = True,
    ):
        
        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.asarray(Ki, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.Kaw = np.asarray(Kaw, dtype=float)
        self.use_feedforward = use_feedforward
 
        # integrator states (persist across steps, reset() clears them)
        self.int_ned = np.zeros(2)   # integral of [N, E] error, NED
        self.int_psi = 0.0           # integral of psi error
 
        # last requested wrench, kept for anti-windup back-calculation
        self._tau_unsat = np.zeros(6)
 
        # logged by the simulator if present
        self.last_pid_body = {"P": np.zeros(6), "I": np.zeros(6), "D": np.zeros(6)}

    def reset(self) -> None:
        """
        Clear the integrator states before a new simulation run. Without
        this, the integral terms would carry over accumulated error from
        a previous run, which would corrupt the first part of a new
        simulation.
        """
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self._tau_unsat = np.zeros(6)

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
       
        """
        Compute the desired generalized force tau_d for the current time
        step, following the PID structure recommended in Section 6.1 of
        the project task description. The position error in N and E is
        computed directly in NED, while the heading error is computed
        with atan2 as required in Section 6, so that it always represents
        the shortest angular distance. 
        
        The NED position error, its
        accumulated integral, and the reference velocity are all rotated
        into the body frame using ned_to_body_xy before they enter the
        force calculation, since the thrusters can only produce force in
        the body frame. 
        
        Three PID terms are computed per channel
        (surge, sway and yaw). The proportional term uses the rotated
        position error, the integral term uses the rotated accumulated
        error and removes steady state error under constant disturbances
        such as current or wind, and the derivative term uses the
        difference between the reference velocity and the measured body
        frame velocity rather than a numerical derivative of the error,
        which is more robust to measurement noise. When use_feedforward
        is True, the reference velocity from the reference model is used
        directly in the derivative term, which improves tracking of a
        moving setpoint and reduces how much work the integral term has
        to do, following the feedforward approach described in Section
        6.1. 
        
        The resulting six degree of freedom wrench only has nonzero
        entries in surge (index 0), sway (index 1) and yaw (index 5), in
        accordance with the horizontal plane 3 DOF model used throughout
        the project.
        """
       
        if nu_ref is None:
            nu_ref = np.zeros(6)
        if acc_ref is None:
            acc_ref = np.zeros(6)
 
        psi = eta[5]
 
        # errors, computed in NED 
        e_ned = eta_ref[:2] - eta[:2]
        e_psi = wrap_angle_pi(eta_ref[5] - psi)   # atan2-style wrap, always in (-pi, pi]
 
        # rotate the NED position error into BODY frame 
        # This is the step that's easy to forget: thrusters act in BODY,
        # so a NED error must go through R(psi)^T before it becomes a force.
        e_body_xy = ned_to_body_xy(e_ned, psi)
 
        # integrate error 
        self.int_ned = self.int_ned + e_ned * dt
        self.int_psi = self.int_psi + e_psi * dt
        int_body_xy = ned_to_body_xy(self.int_ned, psi)
 
        # velocity term: reference velocity (rotated to BODY) minus
        # measured BODY velocity, rather than differentiating the error 
        nu_ref_body_xy = nu_ref_body_xy = ned_to_body_xy(nu_ref[:2], psi)
        e_vel_body_xy = nu_ref_body_xy - nu[:2]
        e_vel_psi = nu_ref[5] - nu[5]
 
        if not self.use_feedforward:
            # Pure feedback: ignore reference velocity, damp on measured
            # velocity only. Still a valid PID controller, but tracking of
            # a moving reference (during reference model transitions) will
            # lag more without the feedforward term.
            e_vel_body_xy = -nu[:2]
            e_vel_psi = -nu[5]
 
        # PID terms (BODY frame) 
        P_xy = self.Kp[:2] * e_body_xy
        I_xy = self.Ki[:2] * int_body_xy
        D_xy = self.Kd[:2] * e_vel_body_xy
 
        P_psi = self.Kp[2] * e_psi
        I_psi = self.Ki[2] * self.int_psi
        D_psi = self.Kd[2] * e_vel_psi
 
        tau_d = np.zeros(6)
        tau_d[0], tau_d[1] = P_xy + I_xy + D_xy
        tau_d[5] = P_psi + I_psi + D_psi
 
        self._tau_unsat = tau_d.copy()
 
        self.last_pid_body = {
            "P": np.array([P_xy[0], P_xy[1], 0, 0, 0, P_psi]),
            "I": np.array([I_xy[0], I_xy[1], 0, 0, 0, I_psi]),
            "D": np.array([D_xy[0], D_xy[1], 0, 0, 0, D_psi]),
        }
 
        return tau_d
 
    def apply_external_aw(self, tau_applied: np.ndarray, psi: float, dt: float) -> None:
        """
        Back-calculation anti-windup (project text, Section 6.3):
 
            x_dot_i = e + Kaw * (tau_sat - tau_unsat)
 
        This is called by the simulator with the wrench that was actually 
        applied after thrust allocation and any actuator limits. 
        If the applied wrench differs from what compute requested, it means 
        the allocator or the actuators could not deliver the full demand, 
        and the integrator is pulled back proportionally to Kaw instead of being allowed to
        keep growing. 

        Since tau is expressed in the body frame while the
        stored integrator state int_ned is expressed in NED, the
        correction term is rotated from body to NED using body_to_ned_xy
        before it is added to the integrator, so that the units and frame
        of the correction match the units and frame of the integrator it
        is applied to.
        """
        tau_applied = np.asarray(tau_applied, dtype=float).reshape(6)
        diff_xy_ned = tau_applied[:2] -  self._tau_unsat[:2]  
        # tau is in BODY; integrator states int_ned are in NED, so rotate
        # the correction from BODY back to NED before applying it.
        diff_ned = body_to_ned_xy(diff_xy_ned, psi)
        self.int_ned += self.Kaw[:2] * diff_ned * dt / self.Ki[:2]
        self.int_psi += self.Kaw[2] * (tau_applied[5] - self._tau_unsat[5]) * dt / self.Ki[2]