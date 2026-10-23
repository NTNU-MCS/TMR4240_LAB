"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from simulation.utils import wrap_angle_pi

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.

    Gust implementation
    -------------------
    ``gust=True`` synthesizes a stationary realization of the one-sided
    Davenport longitudinal spectrum

        S_u(f) = 4 k U^2 x^2 / (f (1 + x^2)^(4/3)),
        x = 1200 f / U,  k = 0.003,

    as a sum of random-phase harmonics. ``gust_params`` may override
    ``n_harmonics`` (256), ``f_min`` (1/600 Hz), ``f_max`` (2 Hz), and
    ``davenport_k`` (0.003). Each harmonic amplitude is
    ``sqrt(2 S_u(f_i) df_i)``, using logarithmic frequency-bin widths.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed
        self._slow = 0.0
        self._gust_value = 0.0
        self._direction_offset = 0.0
        self._rng = np.random.default_rng(seed)
        self._alpha_deg, self._coefficients = load_wind_coefficients()
        self._gust_freq = np.empty(0)
        self._gust_amplitude = np.empty(0)
        self._gust_phase = np.empty(0)
        if self.gust:
            n_harmonics = int(self.gust_params.get("n_harmonics", 256))
            f_min = float(self.gust_params.get("f_min", 1.0 / 600.0))
            f_max = float(self.gust_params.get("f_max", 2.0))
            davenport_k = float(self.gust_params.get("davenport_k", 0.003))
            if n_harmonics < 1 or f_min <= 0.0 or f_max <= f_min or davenport_k <= 0.0:
                raise ValueError("invalid Davenport gust spectrum parameters")
            edges = np.geomspace(f_min, f_max, n_harmonics + 1)
            self._gust_freq = np.sqrt(edges[:-1] * edges[1:])
            bin_widths = np.diff(edges)
            reference_speed = max(abs(self.mean_speed), 1e-6)
            reduced_frequency = 1200.0 * self._gust_freq / reference_speed
            spectrum = (
                4.0 * davenport_k * reference_speed**2 * reduced_frequency**2
                / (self._gust_freq * (1.0 + reduced_frequency**2)**(4.0 / 3.0))
            )
            self._gust_amplitude = np.sqrt(2.0 * spectrum * bin_widths)
            self._gust_phase = self._rng.uniform(0.0, 2.0 * np.pi, n_harmonics)

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.semantics not in {"from", "towards"}:
            raise ValueError("semantics must be 'from' or 'towards'")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        if self.sigma_slow > 0.0:
            if self.tau_slow <= 0.0:
                raise ValueError("tau_slow must be > 0 when sigma_slow is nonzero")
            decay = np.exp(-dt / self.tau_slow)
            self._slow = decay * self._slow + self.sigma_slow * np.sqrt(1.0 - decay**2) * self._rng.normal()
        u_mean = max(0.0, self.mean_speed + self._slow)

        if self.gust:
            if dt > 0.0 and self._gust_freq[-1] >= 0.5 / dt:
                raise ValueError("gust f_max must be below the simulation Nyquist frequency")
            u_gust = float(np.dot(
                self._gust_amplitude,
                np.cos(2.0 * np.pi * self._gust_freq * t + self._gust_phase),
            ))
        else:
            u_gust = 0.0

        if self.sigma_dir > 0.0:
            if self.tau_dir <= 0.0:
                raise ValueError("tau_dir must be > 0 when sigma_dir is nonzero")
            decay = np.exp(-dt / self.tau_dir)
            self._direction_offset = decay * self._direction_offset + self.sigma_dir * np.sqrt(1.0 - decay**2) * self._rng.normal()
            self._direction_offset = float(np.clip(self._direction_offset, -self.dir_limit, self.dir_limit))
        else:
            self._direction_offset = 0.0

        ambient_speed = u_mean + u_gust
        beta_mean = self.beta + (np.pi if self.semantics == "from" else 0.0)
        beta_ned = wrap_angle_pi(beta_mean + self._direction_offset)
        wind_ned = ambient_speed * np.array([np.cos(beta_ned), np.sin(beta_ned)])
        psi = float(np.asarray(eta, dtype=float).reshape(6)[5])
        c, s = np.cos(psi), np.sin(psi)
        wind_body = np.array([c * wind_ned[0] + s * wind_ned[1],
                              -s * wind_ned[0] + c * wind_ned[1]])
        vessel_body = np.asarray(nu, dtype=float).reshape(6)[:2]
        relative_body = wind_body - vessel_body
        relative_speed = float(np.linalg.norm(relative_body))
        alpha_body = float(np.arctan2(relative_body[1], relative_body[0]) % (2.0 * np.pi))
        alpha_deg = np.degrees(alpha_body)
        coefficients = np.array([
            np.interp(alpha_deg, self._alpha_deg, self._coefficients[:, i])
            for i in range(self._coefficients.shape[1])
        ])
        tau_w6 = relative_speed**2 * coefficients
        return tau_w6, {
            "U": ambient_speed,
            "beta_ned": beta_ned,
            "alpha_body": alpha_body,
            "U_mean": u_mean,
            "U_gust": u_gust,
        }
