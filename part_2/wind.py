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
from simulation.utils import ned_to_body_xy

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
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        # TODO: Store and use the parameters above in step().
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

        #  load wind coefficient table
        self.alpha_deg, self.C6 = load_wind_coefficients()
        self.rng =np.random.default_rng(seed) # seed=123
        self.U_slow = self.rng.normal(0,  self.sigma_slow)
        self.U_gust  =0.0
        self.beta_slow =  0.0

        # Harris wind spectrum parameters for part 2 gusts
        self.L =float(self.gust_params.get("L" , 1800.0 ))
        self.U10 =self.mean_speed
        self.kappa_g =float(self.gust_params.get("kappa_g",  0.0026))
        self.N_gust =int(self.gust_params.get("N", 500) )
        self.f_gust =  np.linspace(0.001, 1.0 , self.N_gust)
        self.df_gust =  self.f_gust[1] -  self.f_gust[0]
        self.phi_gust =self.rng.uniform(
            0.0, 2.0 * np.pi, self.N_gust
        )

    # Interpolate wind coefficients to find the values for the current wind angle
    def wind_coeffs(self, alpha_rw_deg: float) -> np.ndarray :
        a =np.mod(alpha_rw_deg, 360.0)
        return np.array( [
            np.interp(a,  self.alpha_deg, self.C6[: , i], period =360.0)
            for i in range(6)
        ])
    
    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:

        #Slowly varying wind speed
        U_mean =self.mean_speed + self.U_slow
        #Harris, sum of freq. components with random phases to generate wind gusts
        if self.gust:
            f_tilde =  self.L *  self.f_gust/ self.U10
            S_gust = (
                4 * self.kappa_g * self.L * self.U10
                / (2 + f_tilde**2)**(5 /  6)
            )
            self.U_gust = np.sum(
                np.sqrt(2 * S_gust * self.df_gust)
                * np.cos(2 * np.pi * self.f_gust * t + self.phi_gust)
            )
        else:
            self.U_gust = 0.0

        U_gust = self.U_gust        


        
        U_wind =max(U_mean + U_gust,0.0) #max so no negative wind veolcity
        
        #updating slowly varying wind speed with first ordeer Gauss-Markov
        a =np.exp(-dt /  self.tau_slow )
        #b = self.sigma_slow**2 * (1 - a**2) #unødvendig å ha
        self.U_slow =(
            a * self.U_slow
            + self.sigma_slow * np.sqrt(1 - a**2) *self.rng.normal()
        )

        #  adding a slowly varying wind direction, np.clip to not exceed the dir_limit
        a_dir =np.exp(-dt /  self.tau_dir)
        b_dir =  self.sigma_dir * np.sqrt(1 -  a_dir**2)

        self.beta_slow =  (
            a_dir *  self.beta_slow
            + b_dir * self.rng.normal()
        )

        self.beta_slow =np.clip(
            self.beta_slow,
            -self.dir_limit,
            self.dir_limit
        )

        beta_mean =self.beta + self.beta_slow

        #Converting the wind direction  from "from " to "toward"
        if self.semantics =="from":
            beta =(beta_mean  + np.pi) % (2 *  np.pi)
        else:
            beta =  beta_mean

            
        # Relative wind velocity in NED 
        V_wind_ned =np.array([
            U_wind *  np.cos(beta),
            U_wind * np.sin(beta )
        ])
        #Relative wind in BODY, subtracting vessel velocity
        V_rw_b =ned_to_body_xy(V_wind_ned, eta[5])  - np.array([
            nu[0], nu[1]
        ])

        U_rw = np.sqrt(V_rw_b[0]**2 + V_rw_b[1]**2 )
        alpha_rw =np.arctan2(V_rw_b[1],  V_rw_b[0])
        
        # Wind coeffs from relativ angle, calculating the 6 load components
        wind_coeffs =self.wind_coeffs(np.degrees(alpha_rw ))
        tau_w6 =U_rw**2 *  wind_coeffs

        info ={
            "U": U_wind,
            "beta_ned":  beta,
            "alpha_body": alpha_rw,
            "U_mean": U_mean,
            "U_gust": U_gust,
        }

        return tau_w6, info
