"""
Performance metrics for the Part 2 simulations (notebooks/part_2_simulations.ipynb).

All functions take the ``Part2Logs`` returned by ``DPSimulatorPart2.run`` and
an evaluation window [t_from, t_to]. Paired comparisons (Simulation 2 vs 3,
raw vs observer, design sea vs heavy sea) must use windows of the same length,
so every function takes the window explicitly instead of picking one.

Conventions:
  * pose errors are measured against the reference-model output ``logs.sp``
    (tracking) or the commanded set-point ``logs.cmd`` (settling, station
    keeping); heading errors are wrapped to (-pi, pi];
  * "LF truth" is the 30 s centred moving average of the true state, the same
    definition as ``simulation/capability.py`` and ``plot_observer_error``;
    the first and last half window are excluded, where that average is biased;
  * the allocation residual is r_tau = tau_c - tau_thr (Section 8.2).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from simulation.capability import low_frequency_motion, max_deviation

IDX = [0, 1, 5]   # surge/sway/yaw (or N/E/psi) in the 6-DOF arrays


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def window(logs, t_from: float = 0.0, t_to: Optional[float] = None) -> np.ndarray:
    """Boolean mask for t_from <= t <= t_to (t_to = end of run by default)."""
    t_to = logs.t[-1] if t_to is None else t_to
    return (logs.t >= t_from - 1e-9) & (logs.t <= t_to + 1e-9)


def _interior(logs, sel: np.ndarray, lf_window: float) -> np.ndarray:
    """Drop the first/last half moving-average window from a selection."""
    half = 0.5 * lf_window
    return sel & (logs.t >= logs.t[0] + half) & (logs.t <= logs.t[-1] - half)


def _rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def _moving_average(x: np.ndarray, dt: float, lf_window: float) -> np.ndarray:
    k = max(1, int(round(lf_window / dt)))
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _lf(x: np.ndarray, dt: float, lf_window: float) -> np.ndarray:
    """Column-wise 30 s moving average of an (n, m) array."""
    return np.column_stack([_moving_average(x[:, j], dt, lf_window)
                            for j in range(x.shape[1])])


# ---------------------------------------------------------------------------
# Tracking and station keeping
# ---------------------------------------------------------------------------

def pose_error(logs, target: str = "sp") -> np.ndarray:
    """(n, 3) [N, E, psi] error eta - target, target 'sp' (reference) or 'cmd'."""
    ref = getattr(logs, target)
    e = logs.eta[:, IDX] - ref[:, IDX]
    e[:, 2] = wrap(e[:, 2])
    return e


def tracking_metrics(logs, sel: np.ndarray, target: str = "sp") -> Dict[str, float]:
    """Max/RMS horizontal position error [m] and heading error [deg]."""
    e = pose_error(logs, target)[sel]
    pos = np.hypot(e[:, 0], e[:, 1])
    return {
        "max_pos_m": float(np.max(pos)),
        "rms_pos_m": float(_rms(pos)),
        "max_psi_deg": float(np.degrees(np.max(np.abs(e[:, 2])))),
        "rms_psi_deg": float(np.degrees(_rms(e[:, 2]))),
    }


def settling_time(logs, t_start: float, t_end: float, pos_tol: float = 1.0,
                  psi_tol_deg: float = 1.0) -> float:
    """Time after t_start until the pose stays within the tolerances of the
    commanded set-point up to t_end. Uses the LF motion, so wave-frequency
    oscillation does not count. NaN if it never settles in the window.

    The centred 30 s average already sees the next leg in the last 15 s
    before a set-point switch, so the check ends half a window before t_end."""
    lf_window = 30.0
    lf = low_frequency_motion(logs, lf_window)
    sel = window(logs, t_start, t_end - 0.5 * lf_window)
    t = logs.t[sel]
    pos = np.hypot(lf[sel, 0] - logs.cmd[sel, 0], lf[sel, 1] - logs.cmd[sel, 1])
    psi = np.abs(wrap(lf[sel, 2] - logs.cmd[sel, 5]))
    outside = (pos > pos_tol) | (psi > np.radians(psi_tol_deg))
    if outside[-1]:
        return float("nan")
    last_out = np.flatnonzero(outside)
    return 0.0 if last_out.size == 0 else float(t[last_out[-1] + 1] - t_start)


def leg_table(logs, hold: float, n_legs: int, pos_tol: float = 1.0,
              psi_tol_deg: float = 1.0) -> List[Dict[str, float]]:
    """Per four-corner leg: tracking error w.r.t. the reference trajectory,
    settling time and final error w.r.t. the commanded corner."""
    rows = []
    dt = float(logs.t[1] - logs.t[0])
    for k in range(n_legs):
        # the command switches at t = (k+1)*hold, so stop one sample before
        t0, t1 = k * hold, (k + 1) * hold - 0.5 * dt
        sel = window(logs, t0, t1)
        row = {"leg": k + 1, **tracking_metrics(logs, sel, "sp"),
               "settle_s": settling_time(logs, t0, t1, pos_tol, psi_tol_deg)}
        i_end = np.flatnonzero(sel)[-1]
        e = pose_error(logs, "cmd")[i_end]
        row["final_pos_m"] = float(np.hypot(e[0], e[1]))
        row["final_psi_deg"] = float(np.degrees(e[2]))
        rows.append(row)
    return rows


def station_keeping(logs, t_from: float, t_to: float) -> Dict[str, float]:
    """Max total and LF deviation from the set-point, and RMS position error.
    Same deviation definitions as Simulation 5 (simulation/capability.py)."""
    sel = window(logs, t_from, t_to)
    sub = _subset(logs, sel)
    pos_tot, psi_tot = max_deviation(sub, sub.t[0], deviation="total")
    pos_lf, psi_lf = max_deviation(sub, sub.t[0], deviation="lf")
    e = pose_error(logs, "cmd")[sel]
    return {
        "max_pos_total_m": pos_tot, "max_psi_total_deg": float(np.degrees(psi_tot)),
        "max_pos_lf_m": pos_lf, "max_psi_lf_deg": float(np.degrees(psi_lf)),
        "rms_pos_m": float(_rms(np.hypot(e[:, 0], e[:, 1]))),
        "rms_psi_deg": float(np.degrees(_rms(e[:, 2]))),
    }


class _subset:
    """Light view of the logs restricted to a window (for max_deviation)."""

    def __init__(self, logs, sel):
        self.t, self.eta, self.sp = logs.t[sel], logs.eta[sel], logs.cmd[sel]


# ---------------------------------------------------------------------------
# Actuators and allocation
# ---------------------------------------------------------------------------

def actuator_effort(logs, thrusters, sel: np.ndarray) -> Dict[str, float]:
    """Thrust use and actuator activity in the window.

    util_mean_pct : mean thrust utilisation (Simulation 5 definition)
    thrust_kN     : mean sum_i |u_i|
    thrust_rate_kN_s : mean sum_i |du_i/dt|   (how hard the thrusters work)
    azimuth_travel_deg : total azimuth rotation, both azimuths together
    """
    u, a = logs.u[sel], logs.alpha[sel]
    dt = float(logs.t[1] - logs.t[0])
    u_max = np.array([th.u_max for th in thrusters])
    az = [i for i, th in enumerate(thrusters) if th.kind.lower() != "tunnel"]
    return {
        "util_mean_pct": float(100.0 * np.mean(np.sum(np.abs(u), axis=1)) / np.sum(u_max)),
        "thrust_kN": float(np.mean(np.sum(np.abs(u), axis=1)) / 1e3),
        "thrust_rate_kN_s": float(np.mean(np.sum(np.abs(np.diff(u, axis=0)), axis=1)) / dt / 1e3),
        "azimuth_travel_deg": float(np.degrees(np.sum(np.abs(wrap(np.diff(a[:, az], axis=0)))))),
    }


def residual(logs) -> np.ndarray:
    """(n, 3) allocation residual r_tau = tau_c - tau_thr [N, N, Nm]."""
    return logs.tau_d[:, IDX] - logs.tau_thr[:, IDX]


def residual_metrics(logs, sel: np.ndarray, f_tol: float = 1e3,
                     m_tol: float = 10e3) -> Dict[str, float]:
    """Max/RMS residual per DOF and the share of time it exceeds a tolerance."""
    r = residual(logs)[sel]
    big = (np.abs(r[:, 0]) > f_tol) | (np.abs(r[:, 1]) > f_tol) | (np.abs(r[:, 2]) > m_tol)
    return {
        "max_rX_kN": float(np.max(np.abs(r[:, 0])) / 1e3),
        "max_rY_kN": float(np.max(np.abs(r[:, 1])) / 1e3),
        "max_rN_kNm": float(np.max(np.abs(r[:, 2])) / 1e3),
        "rms_rX_kN": float(_rms(r[:, 0]) / 1e3),
        "rms_rY_kN": float(_rms(r[:, 1]) / 1e3),
        "rms_rN_kNm": float(_rms(r[:, 2]) / 1e3),
        "time_above_tol_pct": float(100.0 * np.mean(big)),
    }


def active_limits(logs, thrusters, frac: float = 0.98) -> Dict[str, Dict[str, np.ndarray]]:
    """Boolean time series per thruster of which actuator limit is active.

    saturated : |u| >= frac * u_max
    ramping   : |du/dt| >= frac * u_rate      (thrust ramp-rate limit)
    slewing   : |dalpha/dt| >= frac * rot_speed (azimuth rate limit)
    """
    dt = float(logs.t[1] - logs.t[0])
    out = {}
    for i, th in enumerate(thrusters):
        du = np.r_[0.0, np.abs(np.diff(logs.u[:, i]))] / dt
        lim = {"saturated": np.abs(logs.u[:, i]) >= frac * th.u_max,
               "ramping": du >= frac * th.u_rate}
        if th.rot_speed > 0:
            da = np.r_[0.0, np.abs(wrap(np.diff(logs.alpha[:, i])))] / dt
            lim["slewing"] = da >= frac * th.rot_speed
        out[th.name] = lim
    return out


def intervals(t: np.ndarray, mask: np.ndarray, min_duration: float = 1.0) -> List[Tuple[float, float]]:
    """Contiguous intervals where mask is True, at least min_duration long."""
    m = np.r_[False, mask, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1) - 1
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)
            if t[e] - t[s] >= min_duration]


def limit_summary(logs, thrusters, sel: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Share of time [%] each limit is active, per thruster."""
    sel = np.ones_like(logs.t, dtype=bool) if sel is None else sel
    return {name: {k: float(100.0 * np.mean(v[sel])) for k, v in lim.items()}
            for name, lim in active_limits(logs, thrusters).items()}


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

def estimate_lag(x_est: np.ndarray, x_ref: np.ndarray, t: np.ndarray,
                 max_lag: float = 30.0) -> float:
    """Delay [s] of x_est behind x_ref: the time shift that minimises the RMS
    difference after a linear trend is removed from both (positive = the
    estimate lags). Resolution is one sample. A large constant estimation
    offset can be partly traded against a shift, so a small negative value
    means "no measurable lag", not a lead."""
    A = np.c_[t, np.ones_like(t)]
    a = x_est - A @ np.linalg.lstsq(A, x_est, rcond=None)[0]
    b = x_ref - A @ np.linalg.lstsq(A, x_ref, rcond=None)[0]
    dt = float(t[1] - t[0])
    best_rms, best_k = np.inf, 0
    for k in range(-int(round(max_lag / dt)), int(round(max_lag / dt)) + 1):
        e = a[k:] - b[:len(b) - k] if k >= 0 else a[:k] - b[-k:]
        rms = float(_rms(e))
        if rms < best_rms:
            best_rms, best_k = rms, k
    return best_k * dt


def observer_metrics(logs, sel: np.ndarray, lf_window: float = 30.0) -> Dict[str, float]:
    """Estimation quality in the window.

    rms_lf_*   : RMS of estimate - LF truth (the quantity the controller should see)
    rms_tot_*  : RMS of estimate - true state
    wf_ratio_* : std of the wave-frequency part (signal minus its 30 s average)
                 of the estimate over that of the measurement; < 1 = attenuation
    rms_vel_*  : RMS of velocity estimate - LF true velocity
    rough_*    : RMS of the estimate's rate of change, minus that of the LF truth
                 rate (0 = as smooth as the LF motion)
    lag_*_s    : delay of the estimate behind the LF truth (estimate_lag)

    wf_ratio is only meaningful with waves; without waves there is no
    wave-frequency motion to attenuate and the ratio compares noise floors.
    """
    dt = float(logs.t[1] - logs.t[0])
    sel = _interior(logs, sel, lf_window)
    eta, est, meas = (np.unwrap(x[:, IDX], axis=0) for x in (logs.eta, logs.eta_est, logs.eta_measured))
    eta_lf = _lf(eta, dt, lf_window)
    nu_lf = _lf(logs.nu[:, IDX], dt, lf_window)
    e_lf = est - eta_lf
    e_tot = est - eta
    e_lf[:, 2], e_tot[:, 2] = wrap(e_lf[:, 2]), wrap(e_tot[:, 2])
    hf_est = est - _lf(est, dt, lf_window)
    hf_meas = meas - _lf(meas, dt, lf_window)
    rate_est = np.gradient(est, dt, axis=0)
    rate_lf = np.gradient(eta_lf, dt, axis=0)
    e_nu = logs.nu_est[:, IDX] - nu_lf

    out = {}
    for j, (k, scale) in enumerate((("N", 1.0), ("E", 1.0), ("psi", np.degrees(1.0)))):
        out[f"rms_lf_{k}"] = float(scale * _rms(e_lf[sel, j]))
        out[f"rms_tot_{k}"] = float(scale * _rms(e_tot[sel, j]))
        s_meas = np.std(hf_meas[sel, j])
        out[f"wf_ratio_{k}"] = float(np.std(hf_est[sel, j]) / s_meas) if s_meas > 1e-12 else float("nan")
        out[f"rough_{k}"] = float(scale * (_rms(rate_est[sel, j]) - _rms(rate_lf[sel, j])))
        out[f"lag_{k}_s"] = estimate_lag(est[sel, j], eta_lf[sel, j], logs.t[sel])
    for j, (k, scale) in enumerate((("u", 1.0), ("v", 1.0), ("r", np.degrees(1.0)))):
        out[f"rms_vel_{k}"] = float(scale * _rms(e_nu[sel, j]))
    return out


def convergence_time(logs_offset, logs_ref, frac: float = 0.05) -> Dict[str, float]:
    """Time [s] until the effect of an initial estimation error has decayed.

    logs_offset is a run with the estimate started at an offset, logs_ref the
    identical run started on the true pose (same environment, seed and
    wrench, so the true motion is identical). Their estimate difference is
    then only the effect of the initial error, without the wave and model
    errors both runs share. Per DOF: the time after which
    |difference| < frac * |initial difference| for the rest of the run
    (NaN if it never gets there)."""
    if not np.allclose(logs_offset.eta, logs_ref.eta):
        raise ValueError("the two runs must have identical true motion")
    d = logs_offset.eta_est[:, IDX] - logs_ref.eta_est[:, IDX]
    d[:, 2] = wrap(d[:, 2])
    out = {}
    for j, k in enumerate(("N", "E", "psi")):
        tol = frac * abs(d[0, j])
        if tol == 0.0:
            out[f"t_conv_{k}_s"] = float("nan")
            continue
        outside = np.abs(d[:, j]) > tol
        if outside[-1]:
            out[f"t_conv_{k}_s"] = float("nan")
        else:
            idx = np.flatnonzero(outside)
            out[f"t_conv_{k}_s"] = 0.0 if idx.size == 0 else float(logs_offset.t[idx[-1] + 1])
    return out
