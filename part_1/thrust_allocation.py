"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig

from scipy.optimize import minimize


#First itteration of thrust allocation, using pseudo-inverse to find the z-vector, and then scaling it down if any of the thrusters exceed their maximum thrust. This is a simple and effective method, but it does not take into account the dynamics of the thrusters or the power consumption. It also does not guarantee that the solution is optimal in any sense.
#Time invariant thrust allocation for a set of thrusters.
#given different name to avoid confusion with the improved ThrustAllocator, when simulations are called
#This code is not used, but included here to show the evolution of the thrust allocation algorithm, and to provide a baseline for comparison with the improved ThrustAllocator.

class ThrustAllocatorBaseline: 
    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        self.Be = self._build_Be()

    #Finding the Be matrix

    def _build_Be(self) -> np.ndarray:

        #Constant 3 x n_z effectiveness matrix.

        n_z = sum(2 if th.kind == "azimuth" else 1 for th in self.thrusters)
        Be = np.zeros((3, n_z))
        col = 0
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                Be[:, col] = [np.cos(thr.alpha0), np.sin(thr.alpha0), thr.x *np.sin(thr.alpha0) - thr.y *np.cos(thr.alpha0)]
                col += 1
            elif thr.kind == "azimuth":
                Be[:, col] = [1.0, 0.0, -thr.y]
                Be[:, col + 1] = [0.0, 1.0, thr.x]
                col += 2
        return Be

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        #Only interested in surge, sway and yaw moment (3-DOF)
        tau = tau_d[[0, 1, 5]]
        z = np.linalg.pinv(self.Be) @ tau

        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)

        #Magnitude of each thruster command, and scale down if any exceed max thrust
        #as recommended in discussion forum

        col = 0
        magnitudes = []
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                magnitudes.append(abs(z[col]))
                col += 1
            elif thr.kind == "azimuth":
                magnitudes.append(np.hypot(z[col], z[col + 1]))
                col += 2

        r = max(mag / thr.u_max for mag, thr in zip(magnitudes, self.thrusters))
        if r > 1.0:
            z = z / r

        col = 0
        for i, thr in enumerate(self.thrusters):
            if thr.kind == "tunnel":
                u_cmd[i] = z[col]
                alpha_cmd[i] = thr.alpha0
                col += 1
            elif thr.kind == "azimuth":
                Fx, Fy = z[col], z[col + 1]
                u_cmd[i] = np.hypot(Fx, Fy)
                alpha_cmd[i] = np.arctan2(Fy, Fx)
                col += 2

        return u_cmd, alpha_cmd


class ThrustAllocator:
    """Thrust allocation via direct SLSQP solve of the true convex disc
    constraint. Only invoked when the pseudo-inverse baseline violates a
    thruster limit; otherwise falls through to the plain pseudo-inverse
    solution. Uses an augmented decision vector [z, lambda] to preserve the
    requested wrench direction when the full demand is infeasible."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        self.Be = self._build_Be()

        self.n_calls = 0  #counter
        self.n_triggered = 0  #counter
        self.n_optimized = 0  #counter
        self.n_fallback = 0  #counter
        self.last_used_optimizer = False  #counter

    def _build_Be(self) -> np.ndarray:

        #Constant 3 x n_z effectiveness matrix, identical to the one in ThrustAllocator

        n_z = sum(2 if th.kind == "azimuth" else 1 for th in self.thrusters)
        Be = np.zeros((3, n_z))
        col = 0
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                Be[:, col] = [np.cos(thr.alpha0), np.sin(thr.alpha0), thr.x * np.sin(thr.alpha0) - thr.y * np.cos(thr.alpha0)]
                col += 1
            elif thr.kind == "azimuth":
                Be[:, col] = [1.0, 0.0, -thr.y]
                Be[:, col + 1] = [0.0, 1.0, thr.x]
                col += 2
        return Be

    def _magnitudes(self, z: np.ndarray) -> list: #Magnutude is needed multiple times, so it is extracted to a function to avoid code duplication
        magnitudes = []
        col = 0
        for thr in self.thrusters:
            if thr.kind == "tunnel":
                magnitudes.append(abs(z[col]))
                col += 1
            elif thr.kind == "azimuth":
                magnitudes.append(np.hypot(z[col], z[col + 1]))
                col += 2
        return magnitudes

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)
        tau = tau_d[[0, 1, 5]]
        z = np.linalg.pinv(self.Be) @ tau

        magnitudes = self._magnitudes(z)
        violates_limit = any(mag > thr.u_max for mag, thr in zip(magnitudes, self.thrusters))

        self.n_calls += 1  #counter
        self.last_used_optimizer = violates_limit  #counter

        if violates_limit:

            self.n_triggered += 1  #counter

            n_z = self.Be.shape[1]

            # Needed the z-vector to be augmented with a lambda variable, so we can maximize it while minimizing the objective function.

            SCALE = 1000.0  # Working in kN instead of N to avoid numerical issues with the optimization, recommended by Enio
            tau_s = tau / SCALE

            def objective(z_bar):
                lam = z_bar[-1]  # Retrieves the lambda variable, can not use 1/lambda because of possible divison by 0
                return -lam  # Minimize -lambda to maximize lambda

            def equality(z_bar):
                lam = z_bar[-1]
                return self.Be @ z_bar[:-1] - lam * tau_s

            def make_tunnel_constraint(col, u_max):
                def constraint(z_bar):
                    return u_max**2 - z_bar[col]**2
                return constraint

            def make_azimuth_constraint(col_x, col_y, u_max):
                def constraint(z_bar):
                    return u_max**2 - (z_bar[col_x]**2 + z_bar[col_y]**2)
                return constraint

            constraints = [{"type": "eq", "fun": equality}]
            col = 0
            for thr in self.thrusters:
                if thr.kind == "tunnel":
                    constraints.append({"type": "ineq", "fun": make_tunnel_constraint(col, thr.u_max / SCALE)})
                    col += 1
                elif thr.kind == "azimuth":
                    constraints.append({"type": "ineq", "fun": make_azimuth_constraint(col, col + 1, thr.u_max / SCALE)})
                    col += 2

            bounds = [(None, None)] * n_z + [(0, 1)]  # z-components are unbounded, lambda variable is bounded between 0 and 1

            # a good initial guess for the optimization is to scale down the z-vector by the maximum ratio of magnitude to u_max, and set lambda to 1/r
            r = max(mag / thr.u_max for mag, thr in zip(magnitudes, self.thrusters))
            lam0 = 1.0 / r
            z_bar0 = np.concatenate([(z / r) / SCALE, [lam0]])

            res = minimize(objective, z_bar0, method="SLSQP", bounds=bounds, constraints=constraints)
            z_bar = res.x

            #Check if the optimization was successful and if the solution satisfies the constraints. If not, fall back to the original z-vector scaled down by r.
            z_candidate = z_bar[:-1] * SCALE
            magnitudes_candidate = self._magnitudes(z_candidate)

            residual = equality(z_bar)
            residual_ok = np.allclose(residual, 0, atol=1e-3)
            
            TOL = 1e-3  # kN-scale margin
            limits_ok = all(mag <= thr.u_max + TOL * 1000 for mag, thr in zip(magnitudes_candidate, self.thrusters))

            if residual_ok and limits_ok:
                self.n_optimized += 1  #counter
                z = z_candidate
            else:
                self.n_fallback += 1  #counter
                #Check if tolerance is needed
                if not limits_ok:
                    violations = [
                        (thr.name if hasattr(thr, "name") else i, mag - thr.u_max)
                        for i, (mag, thr) in enumerate(zip(magnitudes_candidate, self.thrusters))
                        if mag > thr.u_max
                    ]
                    print(f"t={t:.1f}: limit violation(s) = {[(idx, f'{v/1000:.4f} kN over') for idx, v in violations]}")
                if not residual_ok:
                    print(f"t={t:.1f}: equality residual = {np.max(np.abs(residual)):.6f} (kN scale)")
                z = z / r

        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)

        col = 0
        for i, thr in enumerate(self.thrusters):
            if thr.kind == "tunnel":
                u_cmd[i] = z[col]
                alpha_cmd[i] = thr.alpha0
                col += 1
            elif thr.kind == "azimuth":
                Fx, Fy = z[col], z[col + 1]
                u_cmd[i] = np.hypot(Fx, Fy)
                alpha_cmd[i] = np.atan2(Fy, Fx)
                col += 2

        return u_cmd, alpha_cmd