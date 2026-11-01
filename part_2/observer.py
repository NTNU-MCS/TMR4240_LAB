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
import numpy as np

from simulation.utils import Rz, wrap_angle_pi


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

    def step(self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Nonlinear Passive Observer for R/V Gunnerus (3-DOF DP).

    Estimates low-frequency (LF) position/heading eta_hat, velocity nu_hat,
    disturbance bias b_hat, and wave-frequency (WF) motion xi_hat.
    The only measurement input is GPS + compass: y = [N, E, psi].
    """

    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        #  M og D from Appendix B
        self.M = np.array(
            [
                [6.007e5, 0.0, 0.0],
                [0.0, 7.067e5, -4.733e5],
                [0.0, -5.712e5, 5.456e7],
            ]
        )
        self.D = np.array(
            [
                [1.118e3, 0.0, 0.0],
                [0.0, 2.229e4, 0.0],
                [0.0, 0.0, 1.950e6],
            ]
        )
        self.M_inv = np.linalg.inv(self.M)

        # Wave frequency model
        # ω₀ = 2π/Tp
        omega0 = 2.0 * np.pi / 8.0  # rad/s
        lam = 0.1  # lamda

        # 2×2 blokk for ett DOF:  [0, 1; -ω₀², -2λω₀]
        def blk(w0):
            return np.array([[0.0, 1.0], [-(w0**2), -2 * lam * w0]])

        # Block diagonal matrix 6×6: surge (N), sway (E), yaw (ψ)

        Z = np.zeros((2, 2))
        self.Aw = np.block(
            [
                [blk(omega0), Z, Z],
                [Z, blk(omega0), Z],
                [Z, Z, blk(omega0)],
            ]
        )  # (6, 6)

        # xi_hat = [x1_N, x2_N, x1_E, x2_E, x1_ψ, x2_ψ]
        self.Cw = np.zeros((3, 6))
        self.Cw[0, 0] = 1.0  # N-component
        self.Cw[1, 2] = 1.0  # E-Component
        self.Cw[2, 4] = 1.0  # ψ-component

        Tb = 1000.0
        self.Tb_inv = (1.0 / Tb) * np.eye(3)

        # states
        self.xi_hat = np.zeros(6)
        self.eta_hat = np.zeros(6)
        self.nu_hat = np.zeros(6)
        self.b_hat = np.zeros(3)

        p = 0.1
        Mii = np.diag(self.M)
        self.K1 = np.zeros((6, 3))  # K1: wavestate correction
        for i in range(3):
            self.K1[2 * i, i] = 0.5
            self.K1[2 * i + 1, i] = 0.2
        self.K2 = np.diag([3 * p, 3 * p, 3 * p * 0.5])  # position/heading correction gain
        self.K3 = self.M @ np.diag(
            [3 * p**2] * 3
        )  # positive diagonal (passivity condition), passivity condition satisfied: K3 = M · K3_bar
        self.K4 = np.diag(2.0 * Mii * p**3) # bias correction gain

        self.check_stability()

    def check_stability(self):
        """Eigenvalues of the linearised error dynamics (psi = 0). Must all be negative."""
        S = np.hstack([np.eye(3), self.Cw, np.zeros((3, 6))])
        A = np.zeros((15, 15))
        A[0:3, 9:12] = np.eye(3)
        A[3:9, 3:9] = self.Aw
        A[9:12, 9:12] = -self.M_inv @ self.D
        A[9:12, 12:15] = self.M_inv
        A[12:15, 12:15] = -self.Tb_inv
        A[0:3] -= self.K2 @ S
        A[3:9] -= self.K1 @ S
        A[9:12] -= self.M_inv @ self.K3 @ S
        A[12:15] -= self.K4 @ S
        worst = np.linalg.eigvals(A).real.max()
        print(f"NPO slowest pole: {worst:+.4f}  ({'stable' if worst < 0 else 'UNSTABLE'})")
        if worst >= 0:
            raise ValueError("NPO gains give an unstable observer")

    def reset(self, eta0=None):
        """Called by the simulator with the true initial pose before each run."""
        self.xi_hat = np.zeros(6)
        self.eta_hat = np.zeros(6)
        self.nu_hat = np.zeros(6)
        self.b_hat = np.zeros(3)
        if eta0 is not None:
            self.eta_hat = np.asarray(eta0, dtype=float).reshape(6).copy()

    def step(self, t, dt, eta_measured, tau_est):
        IDX = [0, 1, 5]
        eta3 = self.eta_hat[IDX].copy()  # [N̂, Ê, ψ̂]
        nu3 = self.nu_hat[IDX].copy()  # [û, v̂, r̂]
        y = np.asarray(eta_measured, dtype=float).reshape(6)[IDX]
        tau3 = np.asarray(tau_est, dtype=float).reshape(6)[IDX]

        y_hat = eta3 + self.Cw @ self.xi_hat
        y_tilde = y - y_hat
        y_tilde[2] = wrap_angle_pi(y_tilde[2])

        # Rotaion matrix
        psi = eta3[2]
        J = Rz(psi)  # skipsramme → NED
        Jt = J.T  # NED → skipsramme

        xi_dot = self.Aw @ self.xi_hat + self.K1 @ y_tilde

        eta_dot = J @ nu3 + self.K2 @ y_tilde

        nu_dot = self.M_inv @ (-self.D @ nu3 + Jt @ self.b_hat + tau3 + self.K3 @ y_tilde)

        b_dot = -self.Tb_inv @ self.b_hat + self.K4 @ y_tilde

        # Forward Euler
        self.xi_hat += dt * xi_dot
        self.eta_hat[IDX] = eta3 + dt * eta_dot
        self.nu_hat[IDX] = nu3 + dt * nu_dot
        self.b_hat += dt * b_dot

        # bias
        bias6 = np.zeros(6)
        bias6[0] = self.b_hat[0]  # bN
        bias6[1] = self.b_hat[1]  # bE
        bias6[5] = self.b_hat[2]  # bψ

        return ObserverEstimate(
            eta=self.eta_hat.copy(),
            nu=self.nu_hat.copy(),
            bias=bias6,
        )


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):

        # ======== Tunable parameters ========
        # q_w: wave noise intensity [surge, sway, yaw]. 1 = expected wave intensity.
        # q_b: bias noise intensity [N^2/s, N^2/s, (Nm)^2/s]. Start: 2*sigma_b^2/T_b.
        # Continuous-time intensities; converted to per-sample Q = Qc/dt inside step().
        self.q_w = (1.0, 1.0, 1.0)
        self.q_b = (1.0e5, 1.0e5, 4.0e5)
        self.Qc = np.diag(np.concatenate([self.q_w, self.q_b]))  # 6x6 diagonal [Q_w, Q_b]

        # ======== Fixed parameters ========
        self.idx = [0, 1, 5]

        # Vessel model
        self.M = np.array([[6.007e5, 0.0, 0.0], [0.0, 7.067e5, -4.733e5], [0.0, -5.712e5, 5.456e7]])
        self.D = np.array([[1117.6, 0.0, 0.0], [0.0, 2.229e4, 0.0], [0.0, 0.0, 1.95e6]])
        self.M_inv = np.linalg.inv(self.M)

        # Wave model
        omega = 0.628 * np.ones(3)  # dominant wave frequency [rad/s]
        zeta = 0.1 * np.ones(3)  # relative damping
        sigma_wf = np.array([0.375, 0.375, 0.087])  # std of WF motion [m, m, rad]
        Kw = 2.0 * np.sqrt(zeta * omega) * sigma_wf  # K_w,i = 2 sqrt(zeta_i omega_i) sigma_wf,i

        # WF model
        Z3, I3 = np.zeros((3, 3)), np.eye(3)
        # np.block glues the four 3x3 blocks into one 6x6 matrix
        self.Aw = np.block([[Z3, I3], [-np.diag(omega**2), -2.0 * np.diag(zeta * omega)]])
        Cw = np.hstack([Z3, I3])  # 3x6, picks the "output" half of xi_w

        # Bias model
        Tb = 1000.0
        self.Tb_inv = np.eye(3) / Tb

        # Measurement matrix
        self.H = np.hstack([Cw, I3, np.zeros((3, 6))])

        # Noise Input matrix E
        self.E = np.zeros((15, 6))
        self.E[3:6, 0:3] = np.diag(Kw)  # wave noise enters the xi_w "velocity" half
        self.E[9:12, 3:6] = np.eye(3)  # bias noise (E_b = I)

        # Covariances
        sigma_b = np.array([40e3, 40e3, 80e3])  # bias std [N, N, Nm]
        # Continuous-time intensities Qc; converted to per-sample Q = Qc/dt inside step()
        self.R = np.diag([0.2**2, 0.2**2, np.deg2rad(0.20) ** 2])

        sigma_nu = np.array([0.3, 0.3, 0.3])  # initial velocity std
        self.P0 = np.diag(
            np.concatenate(
                [
                    sigma_wf**2 / omega**2,  # integrated wave states
                    sigma_wf**2,  # wave output states
                    sigma_wf**2,  # eta
                    sigma_b**2,  # bias
                    sigma_nu**2,  # nu
                ]
            )
        )

        self.reset()

    def reset(self, eta0=None):
        self.x_bar = np.zeros(15)
        self.P_bar = self.P0.copy()
        self.need_init = eta0 is None  # if no true pose is given, use the first measurement
        if eta0 is not None:
            self.x_bar[6:9] = np.asarray(eta0, dtype=float)[self.idx]

    @staticmethod
    def _dJ(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])

    def _f(self, x, u):
        """Continuous dynamics f(x) + B u."""
        xi, psi, b, nu = x[0:6], x[8], x[9:12], x[12:15]
        J = Rz(psi)
        xdot = np.zeros(15)
        xdot[0:6] = self.Aw @ xi
        xdot[6:9] = J @ nu
        xdot[9:12] = -self.Tb_inv @ b
        xdot[12:15] = self.M_inv @ (-self.D @ nu + J.T @ b + u)
        return xdot

    def _jacobian(self, x):
        """F = d(f + B u)/dx evaluated at x (the matrix G in the report)."""
        psi, b, nu = x[8], x[9:12], x[12:15]
        F = np.zeros((15, 15))
        F[0:6, 0:6] = self.Aw
        F[6:9, 8] = self._dJ(psi) @ nu  # d(J nu)/d psi (only psi column)
        F[6:9, 12:15] = Rz(psi)
        F[9:12, 9:12] = -self.Tb_inv
        F[12:15, 8] = self.M_inv @ (self._dJ(psi).T @ b)  # d(J^T b)/d psi
        F[12:15, 9:12] = self.M_inv @ Rz(psi).T
        F[12:15, 12:15] = -self.M_inv @ self.D
        return F

    def step(self, t, dt, eta_measured, tau_est):
        idx = self.idx
        eta_measured = np.asarray(eta_measured, dtype=float)
        y = eta_measured[idx]  # [N, E, psi]; list indexing picks 3 entries
        u = np.asarray(tau_est, dtype=float)[idx]  # [tau_x, tau_y, tau_psi]

        if self.need_init:  # no reset(eta0) given: start at first measurement
            self.x_bar[6:9] = y
            self.need_init = False

        x, P = self.x_bar, self.P_bar

        # Corrector
        innov = y - self.H @ x
        innov[2] = wrap_angle_pi(innov[2])  # wrap heading innovation
        S = self.H @ P @ self.H.T + self.R
        # K = P H^T S^-1 computed as (S^-1 H P)^T, since S and P are symmetric (no explicit inverse)
        K = np.linalg.solve(S, self.H @ P).T
        x_hat = x + K @ innov
        x_hat[8] = wrap_angle_pi(x_hat[8])  # wrap heading state
        I_KH = np.eye(15) - K @ self.H
        P_hat = I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form
        P_hat = 0.5 * (P_hat + P_hat.T)  # keep P exactly symmetric

        # Output (6-DOF)
        eta_out = eta_measured.copy()  # z, phi, theta passed through from measurement
        eta_out[idx] = x_hat[6:9]  # LF pose (wave part excluded)
        nu_out = np.zeros(6)
        nu_out[idx] = x_hat[12:15]
        bias_out = np.zeros(6)
        bias_out[idx] = x_hat[9:12]

        # Predictor
        if dt > 0.0:
            F = self._jacobian(x_hat)
            Phi = np.eye(15) + dt * F
            Gamma = dt * self.E
            Q = self.Qc / dt  # per-sample covariance
            self.P_bar = Phi @ P_hat @ Phi.T + Gamma @ Q @ Gamma.T
            self.P_bar = 0.5 * (self.P_bar + self.P_bar.T)
            self.x_bar = x_hat + dt * self._f(x_hat, u)
            self.x_bar[8] = wrap_angle_pi(self.x_bar[8])
        else:
            self.x_bar, self.P_bar = x_hat, P_hat

        return ObserverEstimate(eta=eta_out, nu=nu_out, bias=bias_out)


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
