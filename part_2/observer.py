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

class NonlinearPassiveObserver(Observer):
    """Nonlinear passive wave-filtering observer (3-DOF: surge, sway, yaw)."""

    name = "nonlinear_passive"

    def __init__(
        self,
        omega_p=2.0 * np.pi / 8.0, # Peak frequency
        lam=0.1,
        zeta_n=1.0,
        omega_c=1.2 * (2.0 * np.pi / 8.0),
        tb=1000.0,
        k3=np.array([7.0e3, 8.0e3, 6.0e5]),
        k4=np.array([1.2e5, 1.4e5, 1.0e7]),
    ):
        super().__init__()

        #Parameter checks
        omega_p, lam, zeta_n, omega_c = map(np.asarray, (omega_p, lam, zeta_n, omega_c))

        # Notch damping must exceed wave damping (otherwise K1 has the wrong sign)
        if not np.all(zeta_n > lam):
            raise ValueError(f"Need zeta_n > lam (got {zeta_n} vs {lam}).")

        # Cut-off must be above the wave peak frequency
        if not np.all(omega_c > omega_p):
            raise ValueError(f"Need omega_c > omega_p (got {omega_c} vs {omega_p}).")

        # SPR frequency hierarchy: 1/Tb < K3/K4 < omega_p
        gain_ratio = k3 / k4
        if not np.all((1.0 / tb < gain_ratio) & (gain_ratio < omega_p)):
            raise ValueError(
                f"Need 1/tb < k3/k4 < omega_p. Got k3/k4 = {gain_ratio}, "
                f"1/tb = {1.0 / tb:.4g}, omega_p = {float(np.max(omega_p)):.4g}."
            )

        # WF model
        Omega = omega_p * np.eye(3)
        Lambda = lam * np.eye(3)
        self.Aw = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-Omega @ Omega, -2.0 * Lambda @ Omega],
        ])
        self.Cw = np.hstack([np.zeros((3, 3)), np.eye(3)])

        # Bias and LF vessel model
        self.Tb_inv = np.linalg.inv(tb * np.eye(3))
        M = 0.5 * (M3 + M3.T)
        #Checking the passivity proof
        if np.any(np.linalg.eigvalsh(M) <= 0.0):
            raise ValueError("Symmetrized M is not positive definite.")
        if np.any(np.linalg.eigvalsh(0.5 * (D3 + D3.T)) <= 0.0):
            raise ValueError("D is not positive definite.")
        
        self.M_inv = np.linalg.inv(M)
        self.D = D3.copy()

        # Observer gains (Fossen & Strand, 1999)
        k1_top = -2.0 * (zeta_n - lam) * (omega_c / omega_p) * np.eye(3)
        k1_bot = 2.0 * omega_p * (zeta_n - lam) * np.eye(3)
        self.K1 = np.vstack([k1_top, k1_bot])
        self.K2 = omega_c * np.eye(3)
        self.K3 = np.diag(k3)
        self.K4 = np.diag(k4)

        self.reset()

    def reset(self, eta0=None):
        self.xi_hat = np.zeros(6)
        self.eta_hat = np.zeros(3)
        self.b_hat = np.zeros(3)
        self.nu_hat = np.zeros(3)

        if eta0 is not None:
            self.eta_hat = np.asarray(eta0, dtype=float).reshape(6)[[0, 1, 5]].copy()
            self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

    def step(self, t, dt, eta_measured, tau_est):
        eta_meas6 = np.asarray(eta_measured, dtype=float).reshape(6)
        y = eta_meas6[[0, 1, 5]]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[[0, 1, 5]]

        # Innovation
        y_hat = self.eta_hat + self.Cw @ self.xi_hat
        innov = y - y_hat
        innov[2] = wrap_angle_pi(innov[2])

        R = Rz(self.eta_hat[2])

        # State derivatives that don´t depend on the updated bias
        xi_dot = self.Aw @ self.xi_hat + self.K1 @ innov
        b_dot = -self.Tb_inv @ self.b_hat + self.K3 @ innov

        #Wave oscillator: symplectic (semi-implicit) Euler
        self.xi_hat[3:6] = self.xi_hat[3:6] + dt * xi_dot[3:6]
        self.xi_hat[0:3] = self.xi_hat[0:3] + dt * (self.xi_hat[3:6] + self.K1[0:3] @ innov)

        # Bias first, then velocity (uses the updated bias)
        self.b_hat = self.b_hat + dt * b_dot
        nu_dot = self.M_inv @ (
            -self.D @ self.nu_hat + R.T @ self.b_hat + tau + R.T @ (self.K4 @ innov))        
        self.nu_hat = self.nu_hat + dt * nu_dot

        # Pose, using the updated velocity
        eta_dot = R @ self.nu_hat + self.K2 @ innov
        self.eta_hat = self.eta_hat + dt * eta_dot
        self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

        # Output: observer handles N, E, psi; z, phi, theta pass through from measurement
        eta = eta_meas6.copy()
        eta[[0, 1, 5]] = self.eta_hat

        nu = np.zeros(6)
        nu[[0, 1, 5]] = self.nu_hat

        bias = np.zeros(6)
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

