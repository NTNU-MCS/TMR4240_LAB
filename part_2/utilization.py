"""Thrust utilisation for Project Part 2, Simulation 5 — student work.

Definitions (from the project text, Part 2, Simulation 5):

* **Thrust utilisation** at time t is the sum of the magnitudes of the
  individual thruster forces, as a percentage of the sum of the maximum
  nominal thrusts of all active thrusters,

      U_T(t) = 100 * sum_i |u_i(t)| / sum_i u_max,i .

* **Average thrust utilisation** is the time average of U_T(t) over the
  evaluation window — by default the second half of the run, so the
  transient is excluded.

``simulation/capability.py`` (provided) runs the directional sweep, computes
the deviation measures and draws the polar plots; it calls the two functions
below for every direction.  Part 2, Simulation 5 raises
``NotImplementedError`` until you implement them.  The public check
(``python check.py --part 2``) verifies them on a synthetic log.

Inputs: ``logs`` carries ``logs.t`` (n,) time [s] and ``logs.u``
(n, n_thrusters) ACTUAL thrusts [N] (one signed column per thruster);
``thrusters`` is the list of ``ThrusterConfig`` with ``u_max`` [N].
"""
from __future__ import annotations

import numpy as np


def thrust_utilization(logs, thrusters) -> np.ndarray:
    """Instantaneous thrust utilisation U_T(t) in percent, from ACTUAL thrusts.

    The denominator is the installed nominal capacity of the supplied thrusters.
    """
    thrusts = np.asarray(logs.u, dtype=float)
    if thrusts.ndim != 2 or thrusts.shape[1] != len(thrusters):
      raise ValueError("logs.u must have one thrust column per thruster")
    capacity = float(sum(thr.u_max for thr in thrusters))
    if capacity <= 0.0:
      raise ValueError("total nominal thrust capacity must be positive")
    return 100.0 * np.sum(np.abs(thrusts), axis=1) / capacity


def average_utilization(logs, thrusters, t_from: float | None = None) -> float:
    """Average thrust utilisation [%] over ``t >= t_from`` (default: the second
    half of the run, so the transient is excluded).

    By default the evaluation window starts halfway through the logged run.
    """
    times = np.asarray(logs.t, dtype=float).reshape(-1)
    values = thrust_utilization(logs, thrusters)
    if len(times) != len(values) or len(times) == 0:
      raise ValueError("logs.t and logs.u must contain the same nonzero number of samples")
    if np.any(np.diff(times) < 0.0):
      raise ValueError("logs.t must be nondecreasing")
    start = times[0] + 0.5 * (times[-1] - times[0]) if t_from is None else float(t_from)
    selected = times >= start
    if not np.any(selected):
      raise ValueError("t_from is later than the final log sample")
    window_times = times[selected]
    window_values = values[selected]
    if len(window_times) == 1 or window_times[-1] == window_times[0]:
      return float(window_values[-1])
    return float(np.trapezoid(window_values, window_times) / (window_times[-1] - window_times[0]))
