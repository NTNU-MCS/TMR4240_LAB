"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
from simulation.utils import Rz, wrap_angle_pi
import numpy as np

# Mass matrix
M3 = np.array([
    [6.007e5,      0.0,       0.0],
    [0.0,      7.067e5,  -4.733e5],
    [0.0,     -5.712e5,   5.456e7],
])
# Damping matrix
D3 = np.array([
    [1117.6,     0.0,     0.0],
    [0.0,    2.229e4,     0.0],
    [0.0,        0.0,  1.95e6],
])

@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )

import warnings
def _vec3(x, name: str) -> np.ndarray:
    """Accept a scalar or a length-3 sequence and return a (3,) float array."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        a = np.full(3, float(a))
    a = a.reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"{name} must be a scalar or have 3 elements, got shape {a.shape}")
    return a.copy()





class NonlinearPassiveObserver(Observer):
    """Nonlinear passive wave-filtering observer (3-DOF: surge, sway, yaw).

    Estimates horizontal-plane states (Fossen & Strand, 1999):
        - xi_hat  : (6,) wave-frequency (WF) motion states in NED
        - eta_hat : (3,) low-frequency (LF) pose [N, E, psi] in NED
        - b_hat   : (3,) slowly varying environmental bias in NED
        - nu_hat  : (3,) low-frequency (LF) velocity [u, v, r] in BODY

    Wave peak frequency can be given as `omega_p` [rad/s] or peak period `tp` [s].
    If `omega_c` is None it is `omega_c_ratio * omega_p`. Unmodeled DOFs
    (z, phi, theta) are low-pass filtered (or passed through), not zeroed.

    Tuning order:
      1. Set DEFAULT_TP.
      2. Adjust omega_c_ratio, lam, zeta_n (wave filtering vs. lag).
      3. Tune k4.
      4. Tune k3 last.
    """

    name = "nonlinear_passive"
    DEFAULT_TP = 8.0  # [s] used when select_observer is called without args

    # Raw parameters, everything else is derived in _build()
    _RAW = (
        "omega_p", "lam", "zeta_n", "omega_c_ratio", "omega_c_user",
        "tb", "k3", "k4", "rest_cutoff_ratio",
    )

    def __init__(
        self,
        omega_p=None,                       # Wave peak frequency [rad/s]
        tp=None,                            # Wave peak period [s] (alternative to omega_p)
        lam=0.1,                            # Relative damping of wave spectrum [-]
        zeta_n=1.0,                         # Notch damping ratio [-] (zeta_n > lam)
        omega_c=None,                       # Cut-off [rad/s]; None -> omega_c_ratio * omega_p
        omega_c_ratio=1.2,                  # Used when omega_c is None
        tb=1000.0,                          # Bias time constant [s]
        k3=(7.0e3, 8.0e3, 6.0e5),           # Bias innovation gains
        k4=(1.2e5, 1.4e5, 1.0e7),           # Velocity innovation gains
        m3=None,                            # Optional 3x3 inertia matrix (defaults to M3)
        d3=None,                            # Optional 3x3 damping matrix (defaults to D3)
        symmetrize_mass=True,               # Use M = 0.5*(M3 + M3.T)
        lowpass_rest=True,                  # Low-pass [z, phi, theta] (False = passthrough)
        rest_cutoff_ratio=0.25,             # [z, phi, theta] cut-off as fraction of omega_p
        strict=True,                        # Raise on unknown kwargs (False = warn)
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            msg = f"Unknown keyword arguments for {self.name}: {tuple(kwargs)}"
            if strict:
                raise TypeError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        if omega_p is not None and tp is not None:
            raise ValueError("Specify either omega_p or tp, not both.")

        # Store raw parameters only
        if tp is not None:
            tp_vec = _vec3(tp, "tp")
            if np.any(tp_vec <= 0.0):
                raise ValueError("tp must be strictly positive.")
            self.omega_p = 2.0 * np.pi / tp_vec
        elif omega_p is not None:
            self.omega_p = _vec3(omega_p, "omega_p")
        else:
            self.omega_p = _vec3(2.0 * np.pi / self.DEFAULT_TP, "omega_p")

        self.lam = _vec3(lam, "lam")
        self.zeta_n = _vec3(zeta_n, "zeta_n")
        self.omega_c_ratio = _vec3(omega_c_ratio, "omega_c_ratio")
        self.omega_c_user = None if omega_c is None else _vec3(omega_c, "omega_c")
        self.tb = _vec3(tb, "tb")
        self.k3 = _vec3(k3, "k3")
        self.k4 = _vec3(k4, "k4")
        self.rest_cutoff_ratio = float(rest_cutoff_ratio)
        self.symmetrize_mass = bool(symmetrize_mass)
        self.lowpass_rest = bool(lowpass_rest)
        self._m3_raw = M3 if m3 is None else np.asarray(m3, dtype=float)
        self._d3_raw = D3 if d3 is None else np.asarray(d3, dtype=float)

        self._build()   # derive every matrix in one place
        self.reset()

    def _build(self):
        # Validate raw parameters and derive all matrices
        omega_p, lam, zeta_n = self.omega_p, self.lam, self.zeta_n
        tb, k3, k4 = self.tb, self.k3, self.k4
        omega_c = (
            self.omega_c_user.copy()
            if self.omega_c_user is not None
            else self.omega_c_ratio * omega_p
        )

        # Validation
        if np.any(omega_p <= 0.0):
            raise ValueError("omega_p must be strictly positive.")
        if np.any(lam <= 0.0):
            raise ValueError("lam must be strictly positive.")
        if not np.all(zeta_n > lam):
            raise ValueError(f"Need zeta_n > lam (got {zeta_n} vs {lam}).")
        if not np.all(omega_c > omega_p):
            raise ValueError(f"Need omega_c > omega_p (got {omega_c} vs {omega_p}).")
        if np.any(tb <= 0.0):
            raise ValueError("tb must be strictly positive.")
        if np.any(k3 <= 0.0) or np.any(k4 <= 0.0):
            raise ValueError("k3 and k4 must be strictly positive.")
        if not self.rest_cutoff_ratio > 0.0:
            raise ValueError("rest_cutoff_ratio must be strictly positive.")

        # LF vessel model
        M_base = self._m3_raw
        M = 0.5 * (M_base + M_base.T) if self.symmetrize_mass else M_base.copy()
        if M.shape != (3, 3) or np.any(np.linalg.eigvalsh(0.5 * (M + M.T)) <= 0.0):
            raise ValueError("Mass matrix must be a 3x3 positive definite matrix.")
        M_inv = np.linalg.inv(M)

        # Soft check: 1/Tb < K3/K4 < omega_p (Fossen & Strand 1999 SPR condition)
        ratio = k3 / k4
        if not np.all((1.0 / tb < ratio) & (ratio < omega_p)):
            warnings.warn(
                "Gains do not satisfy 1/Tb < K3/K4 < omega_p; check the Bode "
                "phase of h_zi(jw) for SPR passivity.",
                UserWarning, stacklevel=3,
            )

        # Wave model and loop-shaping gains
        Omega, Lambda = np.diag(omega_p), np.diag(lam)
        Aw = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-Omega @ Omega, -2.0 * Lambda @ Omega],
        ])
        Cw = np.hstack([np.zeros((3, 3)), np.eye(3)])
        K1_top = -2.0 * (zeta_n - lam) * (omega_c / omega_p)
        K1_bot = 2.0 * omega_p * (zeta_n - lam)
        K1 = np.vstack([np.diag(K1_top), np.diag(K1_bot)])

        # Commit matrices and fast diagonal vectors for Symplectic Euler
        self.omega_c = omega_c
        self.M, self.M_inv, self.D = M, M_inv, self._d3_raw.copy()
        self.Aw, self.Cw, self.K1 = Aw, Cw, K1
        self._k1_top, self._k1_bot = K1_top, K1_bot
        self.K2 = np.diag(omega_c)
        self.K3, self.K4 = np.diag(k3), np.diag(k4)
        self.Tb_inv = np.diag(1.0 / tb)
        self.omega_rest = float(self.rest_cutoff_ratio * np.mean(omega_p))

        # Stability limit of the closed-loop wave filter under Symplectic Euler
        self._dt_limit = float(np.min(2.0 / np.maximum(omega_c, omega_p * zeta_n)))
        self._dt_warned = False

    def _apply(self, **changes):
        #Set raw parameters and rebuild; roll back if validation fails
        old = {k: getattr(self, k) for k in self._RAW}
        try:
            for k, v in changes.items():
                setattr(self, k, v)
            self._build()
        except Exception:
            for k, v in old.items():
                setattr(self, k, v)
            raise

    def set_wave_peak(self, omega_p=None, tp=None):
        #Change the wave frequency. Estimator states are kept.
        if omega_p is not None and tp is not None:
            raise ValueError("Specify either omega_p or tp, not both.")
        if tp is not None:
            tp_vec = _vec3(tp, "tp")
            if np.any(tp_vec <= 0.0):
                raise ValueError("tp must be strictly positive.")
            new = 2.0 * np.pi / tp_vec
        elif omega_p is None:
            new = _vec3(2.0 * np.pi / self.DEFAULT_TP, "omega_p")
        else:
            new = _vec3(omega_p, "omega_p")
        self._apply(omega_p=new)

    def retune(self, **params):
        # Change gains/filters after construction, e.g. obs.retune(k4=(...), lam=0.2).
        # Allowed: lam, zeta_n, omega_c, omega_c_ratio, tb, k3, k4, rest_cutoff_ratio.
        
        allowed = {
            "lam", "zeta_n", "omega_c", "omega_c_ratio", "tb", "k3", "k4",
            "rest_cutoff_ratio",
        }
        unknown = set(params) - allowed
        if unknown:
            raise TypeError(f"retune(): unknown parameter(s) {sorted(unknown)}")
        changes = {}
        for k, v in params.items():
            if k == "omega_c":
                changes["omega_c_user"] = None if v is None else _vec3(v, k)
            elif k == "rest_cutoff_ratio":
                changes[k] = float(v)
            else:
                changes[k] = _vec3(v, k)
        self._apply(**changes)

    def reset(self, eta0: np.ndarray | None = None):
        # Initialize LF pose to eta0 (3-DOF or 6-DOF) and all other states to zero.
        self.xi_hat = np.zeros(6, dtype=float)   # WF motion states (NED)
        self.b_hat = np.zeros(3, dtype=float)    # Bias (NED)
        self.nu_hat = np.zeros(3, dtype=float)   # LF velocity (BODY)
        self._dt_warned = False

        if eta0 is not None:
            arr = np.asarray(eta0, dtype=float).ravel()
            if arr.size == 6:
                self.eta_hat = arr[[0, 1, 5]].copy()
                self.rest_hat = arr[[2, 3, 4]].copy()
                self.rest_hat[1] = wrap_angle_pi(self.rest_hat[1])
                self.rest_hat[2] = wrap_angle_pi(self.rest_hat[2])
            elif arr.size == 3:
                self.eta_hat = arr.copy()
                self.rest_hat = None
            else:
                raise ValueError(f"eta0 must have size 3 or 6, got {arr.size}.")
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])
            self._eta_initialized = np.isfinite(self.eta_hat)
            self.eta_hat = np.where(self._eta_initialized, self.eta_hat, 0.0)
        else:
            self.eta_hat = np.zeros(3, dtype=float)
            self.rest_hat = None
            self._eta_initialized = np.zeros(3, dtype=bool)

    @property
    def x_hat(self) -> np.ndarray:
        # Unified 15-state view [xi, eta, b, nu], always consistent with the states.
        return np.concatenate([self.xi_hat, self.eta_hat, self.b_hat, self.nu_hat])

    def _update_rest(self, dt: float, y_rest: np.ndarray):
        # First-order low-pass for unmodeled DOFs [z, phi, theta] with per-channel NaN safety.
        valid = np.isfinite(y_rest)
        if not np.any(valid):
            if self.rest_hat is None:
                self.rest_hat = np.zeros(3, dtype=float)
            return

        if self.rest_hat is None or not self.lowpass_rest:
            if self.rest_hat is None:
                self.rest_hat = np.zeros(3, dtype=float)
            self.rest_hat = np.where(valid, y_rest, self.rest_hat)
            self.rest_hat[1] = wrap_angle_pi(self.rest_hat[1])
            self.rest_hat[2] = wrap_angle_pi(self.rest_hat[2])
            return

        if dt <= 0.0:
            return  # hold the last value instead of jumping to the raw measurement

        alpha = 1.0 - np.exp(-self.omega_rest * dt)
        diff = np.where(valid, y_rest - self.rest_hat, 0.0)
        if valid[1]:
            diff[1] = wrap_angle_pi(diff[1])
        if valid[2]:
            diff[2] = wrap_angle_pi(diff[2])

        self.rest_hat = self.rest_hat + alpha * diff
        self.rest_hat[1] = wrap_angle_pi(self.rest_hat[1])
        self.rest_hat[2] = wrap_angle_pi(self.rest_hat[2])

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        # Advance the NPO by one timestep dt using Symplectic Euler integration.
        eta_meas6 = np.asarray(eta_measured, dtype=float).reshape(6)
        y = eta_meas6[[0, 1, 5]]
        y_rest = eta_meas6[[2, 3, 4]]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[[0, 1, 5]]
        tau = np.where(np.isfinite(tau), tau, 0.0)

        if dt > self._dt_limit and not self._dt_warned:
            warnings.warn(
                f"dt={dt:g}s exceeds the stability limit ~{self._dt_limit:.3g}s "
                "of the wave filter; the observer may become unstable.",
                UserWarning, stacklevel=2,
            )
            self._dt_warned = True

        # Per-channel warm-start on first finite measurement if reset() had no eta0
        valid_y = np.isfinite(y)
        uninit_and_valid = (~self._eta_initialized) & valid_y
        if np.any(uninit_and_valid):
            self.eta_hat = np.where(uninit_and_valid, y, self.eta_hat)
            if uninit_and_valid[2]:
                self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])
            self._eta_initialized |= uninit_and_valid

        # Element-wise innovation (gracefully handles partial sensor dropouts)
        y_hat = self.eta_hat + self.Cw @ self.xi_hat
        y_tilde = np.where(valid_y, y - y_hat, 0.0)
        if valid_y[2]:
            y_tilde[2] = wrap_angle_pi(y[2] - y_hat[2])

        R = Rz(self.eta_hat[2])  # filtered LF heading avoids wave cross-coupling

        # Symplectic (Semi-Implicit) Euler integration
        if dt > 0.0:
            # 1. Wave oscillator: update velocity-like state xi_2, then position-like xi_1
            xi1, xi2 = self.xi_hat[:3], self.xi_hat[3:]
            xi2_new = xi2 + dt * (
                -(self.omega_p ** 2) * xi1
                - 2.0 * self.lam * self.omega_p * xi2
                + self._k1_bot * y_tilde
            )
            xi1_new = xi1 + dt * (xi2_new + self._k1_top * y_tilde)
            self.xi_hat[:3], self.xi_hat[3:] = xi1_new, xi2_new

            # 2. Bias & LF velocity first, then LF pose using updated nu_hat
            b_dot = -self.Tb_inv @ self.b_hat + self.K3 @ y_tilde
            self.b_hat = self.b_hat + dt * b_dot

            nu_dot = self.M_inv @ (
                -self.D @ self.nu_hat
                + R.T @ self.b_hat
                + tau
                + R.T @ (self.K4 @ y_tilde)
            )
            self.nu_hat = self.nu_hat + dt * nu_dot

            eta_dot = R @ self.nu_hat + self.K2 @ y_tilde
            self.eta_hat = self.eta_hat + dt * eta_dot
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

        # Update heave/roll/pitch
        self._update_rest(dt, y_rest)

        # Pack into 6-DOF outputs
        eta = np.zeros(6, dtype=float)
        eta[[0, 1, 5]] = self.eta_hat
        eta[[2, 3, 4]] = self.rest_hat

        nu = np.zeros(6, dtype=float)
        nu[[0, 1, 5]] = self.nu_hat

        bias = np.zeros(6, dtype=float)
        bias[[0, 1, 5]] = self.b_hat

        return ObserverEstimate(eta=eta, nu=nu, bias=bias)


def dRz(psi: float) -> np.ndarray:
    """ Derivative of Rz(psi) with respect to psi. """
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[ -s, -c, 0.0],
                     [  c, -s, 0.0],
                     [0.0, 0.0, 0.0]], dtype=float)
class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(
            self,
            # Tunable parameters (can be changed to 1x3 tuples)
            omega_p = 2.0 * np.pi / 8.0, # Bytt 8.0 med peak perioden fra bølgemodel/klass typ: "sim.waves.tp" hvis mulig
            lam = 0.2, # Øk for å filtrere et bredere bånd
            kw = np.array([1.5, 1.0, 0.06]), # Økning betyr større bølgebevegelse i korresponderende DOF
            tb = 600.0, # Tidskonstant for bias
            kb = np.array([1.0, 1.0, 1.0]), # Tune siste 3 diag-elementer i Q isteden

            *args,
            **kwargs
            ):
        
        # WF model
        Omega = omega_p * np.eye(3)
        Lambda = lam * np.eye(3)
        Kw = np.diag(kw)

        # WF state-space matrices
        self.Aw = np.block([[np.zeros((3, 3)), np.eye(3)], [- Omega @ Omega, - 2 * Lambda @ Omega]])
        self.Cw = np.hstack([np.zeros((3, 3)), np.eye(3)])
        self.Ew = np.vstack([np.zeros((3, 3)), Kw])

        # Bias model
        Tb = tb * np.eye(3)
        self.Tb_inv = np.linalg.inv(Tb)
        self.Eb = np.diag(kb)

        # LF model
        self.M_inv = np.linalg.inv(M3)
        self.D = D3.copy()

        # EKF model
        self.B = np.vstack([np.zeros((6, 3)), np.zeros((3, 3)), np.zeros((3, 3)), self.M_inv])

        self.E = np.zeros((15, 6))
        self.E[0:6, 0:3] = self.Ew
        self.E[9:12, 3:6] = self.Eb

        self.H = np.hstack([self.Cw, np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))])

        self.Q = np.diag([1.0]*3 + [1e8, 1e7, 1e9])
        self.R = np.diag(np.array([0.02, 0.02, 0.0034])**2)
        self.P0 = np.diag([1.0]*6 + [0.1**2]*3 + [1e5**2, 1e5**2, 1e7**2] + [0.1**2]*3)

        self.reset()    

    def reset(self, eta0=None):
        self.x_hat = np.zeros(15)
        if eta0 is not None:
         self.x_hat[6:9] = np.asarray(eta0, float).reshape(6)[[0, 1, 5]]
        self.P_hat = self.P0.copy()
        self.x_bar = self.x_hat.copy()
        self.P_bar = self.P0.copy()

    def _f(self, x: np.ndarray) -> np.ndarray:
        psi = x[8]
        eps = x[0:6]
        eta = x[6:9]
        b = x[9:12]
        nu = x[12:15]
        R = Rz(psi)
        f = np.zeros(15)
        f[0:6] = self.Aw @ eps
        f[6:9] = R @ nu
        f[9:12] = -self.Tb_inv @ b
        f[12:15] = -self.M_inv @ self.D @ nu + self.M_inv @ R.T @ b
        return f

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        psi = x[8]
        R = Rz(psi)
        dR = dRz(psi)
        b, nu = x[9:12], x[12:15]
        J = np.zeros((15, 15))
        J[0:6, 0:6] = self.Aw
        J[6:9, 12:15] = R
        J[6:9, 8] = dR @ nu
        J[9:12, 9:12] = -self.Tb_inv
        J[12:15, 12:15] = -self.M_inv @ self.D
        J[12:15, 9:12] = self.M_inv @ R.T
        J[12:15, 8] = self.M_inv @ (dR.T @ b)
        return J

    def update_kf_gain(self):
        K = self.P_bar @ self.H.T @ np.linalg.inv(self.H @ self.P_bar @ self.H.T + self.R)
        return K

    def update_covariance(self):
        I = np.eye(15)
        P_hat = (I - self.K @ self.H) @ self.P_bar @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T
        P_hat = 0.5 * (P_hat + P_hat.T)  # Ensure symmetry
        return P_hat

    def update_state_estimate(self, eta_measured):
        innov = eta_measured - self.H @ self.x_bar
        innov[2] = wrap_angle_pi(innov[2])  # Wrap heading innovation
        x_hat = self.x_bar + self.K @ innov
        return x_hat

    def predict_state(self, dt: float, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        x_bar = x + dt * (self._f(x) + self.B @ tau)
        return x_bar

    def predict_covariance(self, dt: float, P: np.ndarray, x: np.ndarray) -> np.ndarray:
        phi = np.eye(15) + dt * self._jacobian(x)
        gamma = dt * self.E
        P_bar = phi @ P @ phi.T + gamma @ self.Q @ gamma.T
        return P_bar

    def step(self, t, dt, eta_measured, tau_est):
        eta_measured_3DOF = np.asarray(eta_measured, dtype=float).reshape(6)[[0, 1, 5]]
        tau_est_3DOF = np.asarray(tau_est, dtype=float).reshape(6)[[0, 1, 5]]

        # Predict
        self.P_bar = self.predict_covariance(dt, self.P_hat, self.x_hat)
        self.x_bar = self.predict_state(dt, self.x_hat, tau_est_3DOF)
        self.x_bar[8] = wrap_angle_pi(self.x_bar[8])

        # Correct
        self.K = self.update_kf_gain()
        self.P_hat = self.update_covariance()
        self.x_hat = self.update_state_estimate(eta_measured_3DOF)
        self.x_hat[8] = wrap_angle_pi(self.x_hat[8])

        eta = np.zeros(6)
        eta[[0, 1, 5]] = self.x_hat[6:9]

        nu = np.zeros(6)
        nu[[0, 1, 5]] = self.x_hat[12:15]

        bias = np.zeros(6)
        bias[[0, 1, 5]] = self.x_hat[9:12]
        return ObserverEstimate(eta=eta, nu=nu, bias=bias)

def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

