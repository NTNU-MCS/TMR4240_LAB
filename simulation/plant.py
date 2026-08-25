"""Step-based plant API for the Gunnerus 3-DOF simulation model.

The plant owns the actuator state and vessel state.  With thruster dynamics
enabled, commands are rate-limited and saturated before they are applied to
the vessel; with ideal actuators (Part 1) they are applied exactly.  Wind and
wave models remain outside the plant: they provide generalized BODY loads,
which makes it possible to exchange environmental models without changing the
vessel or actuator code.

Array convention
----------------
Every generalized vector at this interface is 6-DOF, ordered
``[surge, sway, heave, roll, pitch, yaw]``:

* loads (``wind``, ``waves``, ``external``): BODY ``[Fx, Fy, Fz, Mx, My, Mz]``
* current: NED velocity ``[V_N, V_E, V_D, 0, 0, 0]`` [m/s]
* states: ``eta = [N, E, z, phi, theta, psi]``, ``nu = [u, v, w, p, q, r]``

The underlying model is 3-DOF: only the surge, sway, and yaw components
(indices ``[0, 1, 5]``) are integrated; the remaining components are zero.
The reduction happens in exactly one place (``utils.to_3dof``).
"""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable, Sequence

import numpy as np

from models.gunnerus_3dof import Gunnerus3DOF
from models.thruster_dynamics import ThrusterConfig, ThrusterSet
from .utils import to_3dof, to_6dof

ArrayLike = Sequence[float] | np.ndarray
LoadInput = ArrayLike | Callable[..., ArrayLike] | None


@dataclass(frozen=True)
class PlantStep:
    """Inputs, loads, and state resulting from one plant integration step.

    All generalized vectors are 6-DOF (see module docstring).
    """

    t: float
    eta_before: np.ndarray
    nu_before: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    thrust: np.ndarray
    azimuth: np.ndarray
    tau_thrusters: np.ndarray
    tau_wind: np.ndarray
    tau_waves: np.ndarray
    tau_external: np.ndarray
    tau_total: np.ndarray
    current_ned: np.ndarray


class GunnerusPlant3DOF:
    """Combine Gunnerus vessel dynamics, thruster dynamics, and environmental loads.

    ``wind``, ``waves``, and ``external`` passed to :meth:`step` are BODY-frame
    generalized 6-DOF loads ``[Fx, Fy, Fz, Mx, My, Mz]``.  Each may be a
    constant vector or a callable with the signature ``load(t, eta, nu)`` or
    ``load(t, eta)`` receiving the 6-DOF state vectors.

    ``current`` is a NED generalized velocity ``[V_N, V_E, V_D, 0, 0, 0]``.
    It is passed to the Gunnerus model as speed and direction, so its
    relative-velocity and added-mass terms are handled by the vessel equations.
    """

    def __init__(
        self,
        thrusters: list[ThrusterConfig],
        *,
        dt: float = 0.05,
        method: str = "Euler",
        thruster_dynamics: bool = True,
    ) -> None:
        """``thruster_dynamics=False`` gives ideal actuators: commands are
        applied exactly as requested — no rate limits, no saturation, no
        mechanical limits (Project Part 1 uses ideal thrusters).

        ``method`` should stay ``"Euler"``: the underlying mcsimpy model
        integrates the kinematics with the stored velocity state, which is
        exact for Euler but silently wrong for multi-stage methods (RK4)."""
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = float(dt)
        self.vessel = Gunnerus3DOF(dt=self.dt, method=method)
        self.thrusters = ThrusterSet(thrusters, dynamics=thruster_dynamics)

    def reset(
        self,
        eta: ArrayLike | None = None,
        nu: ArrayLike | None = None,
    ) -> None:
        """Reset both vessel and actuator states (6-DOF vectors, zeros allowed)."""
        eta6 = np.zeros(6) if eta is None else np.asarray(eta, dtype=float).reshape(6)
        nu6 = np.zeros(6) if nu is None else np.asarray(nu, dtype=float).reshape(6)
        self.vessel.set_eta(to_3dof(eta6))
        self.vessel.set_nu(to_3dof(nu6))
        self.thrusters.reset()

    def step(
        self,
        t: float,
        thrust_command: ArrayLike,
        azimuth_command: ArrayLike,
        *,
        current: ArrayLike | None = None,
        wind: LoadInput = None,
        waves: LoadInput = None,
        external: LoadInput = None,
    ) -> PlantStep:
        """Advance one time step and return the applied loads and new state."""
        eta_before = to_6dof(self.vessel.get_eta())
        nu_before = to_6dof(self.vessel.get_nu())
        thrust, azimuth, tau_thr3 = self.thrusters.step(
            thrust_command, azimuth_command, self.dt
        )
        tau_thrusters = to_6dof(tau_thr3)
        tau_wind = body_wrench(wind, t, eta_before, nu_before)
        tau_waves = body_wrench(waves, t, eta_before, nu_before)
        tau_external = body_wrench(external, t, eta_before, nu_before)
        tau_total = tau_thrusters + tau_wind + tau_waves + tau_external

        current_ned = (
            np.zeros(6) if current is None
            else np.asarray(current, dtype=float).reshape(6)
        )
        Uc = float(np.hypot(current_ned[0], current_ned[1]))
        beta_c = float(np.arctan2(current_ned[1], current_ned[0]))
        self.vessel.integrate(Uc=Uc, beta_c=beta_c, tau=to_3dof(tau_total))

        return PlantStep(
            t=float(t),
            eta_before=eta_before,
            nu_before=nu_before,
            eta=to_6dof(self.vessel.get_eta()),
            nu=to_6dof(self.vessel.get_nu()),
            thrust=thrust.copy(),
            azimuth=azimuth.copy(),
            tau_thrusters=tau_thrusters,
            tau_wind=tau_wind,
            tau_waves=tau_waves,
            tau_external=tau_external,
            tau_total=tau_total,
            current_ned=current_ned,
        )


    def run(
        self,
        T: float,
        thrust_command: ArrayLike | Callable[[float], ArrayLike],
        azimuth_command: ArrayLike | Callable[[float], ArrayLike],
        *,
        current: ArrayLike | Callable[[float], ArrayLike] | None = None,
        wind: LoadInput = None,
        waves: LoadInput = None,
        external: LoadInput = None,
        reset: bool = True,
        eta0: ArrayLike | None = None,
        nu0: ArrayLike | None = None,
    ) -> dict[str, np.ndarray]:
        """Step the plant for ``T`` seconds and return stacked time histories.

        Commands (and ``current``) may be constant vectors or callables of
        time ``f(t)``; loads follow the same rules as :meth:`step`.  Returns a
        dict of arrays keyed like the :class:`PlantStep` fields (``t``,
        ``eta``, ``nu``, ``thrust``, ``azimuth``, ``tau_*``, ``current_ned``),
        each with one row per time step.  Row ``k`` of ``eta``/``nu`` is the
        state at ``t[k]`` (before that step's integration), so the state
        histories line up with the commands and loads; the state after the
        final step remains in the plant.  ``reset=True`` (default) restarts
        from ``eta0``/``nu0`` (6-DOF vectors, zeros if omitted) — e.g.
        ``eta0=[0, 0, 0, 0, 0, psi0]`` starts at heading ``psi0``.  With
        ``reset=False`` the run continues from the current state (note that
        the time axis still restarts at ``t = 0``).
        """
        if reset:
            self.reset(eta0, nu0)
        elif eta0 is not None or nu0 is not None:
            raise ValueError("eta0/nu0 are initial conditions and require reset=True")
        n = int(round(T / self.dt))
        t = np.arange(n) * self.dt

        def _at(x, tk):
            return x(tk) if callable(x) else x

        steps = [
            self.step(
                t=tk,
                thrust_command=_at(thrust_command, tk),
                azimuth_command=_at(azimuth_command, tk),
                current=_at(current, tk),
                wind=wind, waves=waves, external=external,
            )
            for tk in t
        ]
        out: dict[str, np.ndarray] = {"t": t}
        # eta/nu are logged pre-step so that row k is the state at t[k],
        # aligned with the commands and loads (and with the closed-loop
        # simulator logs); the post-step state would belong to t[k] + dt.
        out["eta"] = np.array([s.eta_before for s in steps])
        out["nu"] = np.array([s.nu_before for s in steps])
        for key in ("thrust", "azimuth", "tau_thrusters",
                    "tau_wind", "tau_waves", "tau_external", "tau_total",
                    "current_ned"):
            out[key] = np.array([getattr(s, key) for s in steps])
        return out


def body_wrench(
    load: LoadInput,
    t: float,
    eta: np.ndarray,
    nu: np.ndarray,
) -> np.ndarray:
    """Evaluate a load input and validate it as a generalized 6-DOF BODY wrench."""
    if load is None:
        return np.zeros(6)
    value = _evaluate_load(load, t, eta, nu) if callable(load) else load
    wrench = np.asarray(value, dtype=float).reshape(-1)
    if wrench.shape != (6,):
        raise ValueError(
            "loads are generalized 6-DOF BODY vectors [Fx, Fy, Fz, Mx, My, Mz]; "
            f"got shape {wrench.shape}"
        )
    return wrench.copy()


def _evaluate_load(
    load: Callable[..., ArrayLike], t: float, eta: np.ndarray, nu: np.ndarray
) -> ArrayLike:
    """Call either a state-aware load or a standard ``WaveLoad(t, eta)``."""
    eta = eta.copy()
    nu = nu.copy()
    try:
        parameters = signature(load)
    except (TypeError, ValueError):
        return load(t, eta, nu)
    try:
        parameters.bind(t, eta, nu)
    except TypeError:
        try:
            parameters.bind(t, eta)
        except TypeError as exc:
            raise TypeError(
                "environmental load callable must accept (t, eta, nu) or (t, eta)"
            ) from exc
        return load(t, eta)
    return load(t, eta, nu)
