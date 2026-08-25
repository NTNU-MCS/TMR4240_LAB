"""Automated checks for the Part 1 subsystems and mandatory simulations.

Every check returns a :class:`CheckResult` with a status:

* ``PASS``            — the criteria below the check name were all met.
* ``FAIL``            — the subsystem ran but at least one criterion failed.
* ``NOT IMPLEMENTED`` — the template placeholder is still in place.
* ``ERROR``           — the code raised an exception or does not follow the
                        template interface (the message says what to fix).

The same checks are exposed three ways, all backed by this module:

* ``python check.py``                — terminal PASS/FAIL report,
* ``pytest``                         — the suites under ``tests/``,
* ``notebooks/part_1_demo.ipynb`` — the mandatory simulations plus checks.

The fast checks mirror the sign tests recommended in the project text
("Tips": simple sign tests, heading wrapping, allocation rank); the
simulation checks run Simulations 1-4 from "Mandatory Tests and Report" and
verify station-keeping accuracy against the expected-performance figures
given there.  Passing them shows the closed loop works; the report is still
assessed on analysis and justification, not on these checks alone.

Environmental models are constructed through the constructor contract
documented in ``part_1/current.py`` and ``part_1/wind.py`` — keep those
signatures working.  Angles are radians, NED, 0 = North, pi/2 = East.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.thruster_dynamics import ThrusterSet
from part_1.config import SimConfig, default_thrusters_gunnerus3
from .utils import wrap_angle_pi

DT = 0.05

# Tolerances for the mandatory simulations.  "Expected performance" in the
# project text: maximum position deviation 1-5 m for a well-tuned solution.
MAX_DEVIATION = 5.0        # [m]   largest allowed excursion from the setpoint
STEADY_POS = 1.0           # [m]   mean position error over the steady window
STEADY_PSI = np.deg2rad(3) # [rad] mean heading error over the steady window
SETTLE_POS = 0.5           # [m]   Simulation 3/4 settling accuracy
STEADY_WINDOW = 100.0      # [s]   averaging window at the end of a run/leg


@dataclass
class CheckResult:
    """Outcome of one check: status, per-criterion lines, and any run logs."""

    name: str
    status: str                 # PASS / FAIL / NOT IMPLEMENTED / ERROR
    details: list[str] = field(default_factory=list)
    logs: dict = field(default_factory=dict)   # label -> Logs, for plotting

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class _Criteria:
    """Collect (description, ok, measured) rows and reduce them to a status."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, description: str, ok: bool, measured: str = "") -> None:
        self.rows.append((description, bool(ok), measured))

    def result(self, name: str, logs: dict | None = None) -> CheckResult:
        details = [
            f"{'ok  ' if ok else 'FAIL'}  {desc}" + (f"  [{measured}]" if measured else "")
            for desc, ok, measured in self.rows
        ]
        status = "PASS" if all(ok for _, ok, _ in self.rows) else "FAIL"
        return CheckResult(name, status, details, logs or {})


def _state6(psi: float = 0.0) -> np.ndarray:
    eta = np.zeros(6)
    eta[5] = psi
    return eta


def _wrench6(Fx: float = 0.0, Fy: float = 0.0, Mz: float = 0.0) -> np.ndarray:
    tau = np.zeros(6)
    tau[0], tau[1], tau[5] = Fx, Fy, Mz
    return tau


# ---------------------------------------------------------------------------
# Fast subsystem checks (fractions of a second each)
# ---------------------------------------------------------------------------

def check_current() -> CheckResult:
    """Current model: 'from' convention and the Simulation 2 rotation profile."""
    from part_1.current import Current

    name = "current: direction conventions"
    try:
        c = Current(0.5, np.pi / 2, semantics="from")           # from east
        v = np.asarray(c.step(0.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    except TypeError as exc:
        return CheckResult(name, "ERROR", [
            f"constructor/step does not follow the template contract: {exc}",
            "expected Current(speed, beta, semantics=...) — see part_1/current.py",
        ])
    if np.allclose(v, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["Current.step() still returns the zero placeholder"])

    crit = _Criteria()
    crit.add("from east -> flows west: [V_N, V_E] = [0, -0.5]",
             np.allclose(v[:2], [0.0, -0.5], atol=1e-6), f"got {np.round(v[:2], 3)}")
    crit.add("components 2-5 of the generalized vector are zero",
             np.allclose(v[2:], 0.0), f"got {np.round(v[2:], 3)}")

    cp = Current(0.5, 0.0, semantics="from", beta_end=np.pi / 2, duration=300.0)
    v0 = np.asarray(cp.step(0.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    vm = np.asarray(cp.step(150.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    v1 = np.asarray(cp.step(300.0, DT, _state6(), np.zeros(6)), float).reshape(6)
    crit.add("rotation start: from north -> [-0.5, 0]",
             np.allclose(v0[:2], [-0.5, 0.0], atol=1e-3), f"got {np.round(v0[:2], 3)}")
    mid = 0.5 * np.array([np.cos(5 * np.pi / 4), np.sin(5 * np.pi / 4)])
    vm_norm = float(np.hypot(vm[0], vm[1]))
    crit.add("rotation midpoint: from northeast (towards southwest), speed 0.5",
             abs(vm_norm - 0.5) < 0.02 and np.allclose(vm[:2], mid, atol=0.05),
             f"got {np.round(vm[:2], 3)}, |v| = {vm_norm:.3f}")
    crit.add("rotation end: from east -> [0, -0.5]",
             np.allclose(v1[:2], [0.0, -0.5], atol=1e-3), f"got {np.round(v1[:2], 3)}")
    return crit.result(name)


def check_wind() -> CheckResult:
    """Wind model: load signs, relative wind, and the slowly-varying component."""
    from part_1.wind import Wind

    name = "wind: load signs and relative wind"
    try:
        w = Wind(10.0, 0.0, semantics="from", seed=0)           # head wind at psi=0
        tau, _ = w.step(0.0, DT, _state6(), np.zeros(6))
        tau = np.asarray(tau, float).reshape(6)
    except TypeError as exc:
        return CheckResult(name, "ERROR", [
            f"constructor/step does not follow the template contract: {exc}",
            "expected Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)"
            " — see part_1/wind.py",
        ])
    if np.allclose(tau, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["Wind.step() still returns the zero placeholder"])

    crit = _Criteria()
    crit.add("wind from north at psi=0: surge force Fx < 0",
             tau[0] < 0.0, f"Fx = {tau[0]/1e3:.2f} kN")
    crit.add("wind from north at psi=0: |Fy| near zero",
             abs(tau[1]) < 0.2 * abs(tau[0]),
             f"Fy = {tau[1]/1e3:.2f} kN vs Fx = {tau[0]/1e3:.2f} kN")

    we = Wind(10.0, np.pi / 2, semantics="from", seed=0)        # from east at psi=0
    tau_e, _ = we.step(0.0, DT, _state6(), np.zeros(6))
    tau_e = np.asarray(tau_e, float).reshape(6)
    crit.add("wind from east at psi=0: sway force Fy < 0 and dominant",
             tau_e[1] < 0.0 and abs(tau_e[1]) > abs(tau_e[0]),
             f"Fx = {tau_e[0]/1e3:.2f} kN, Fy = {tau_e[1]/1e3:.2f} kN")

    # Magnitudes, not just signs: tau_wind = U_rw^2 * C(alpha_rw) with the
    # provided table, so the expected values are computable exactly. A
    # sign-preserving scaling bug (e.g. an extra angle or density factor)
    # passes every sign test above but fails here.
    from part_1.wind import load_wind_coefficients
    alpha_tab, C6_tab = load_wind_coefficients()
    Fx_exp = 10.0 ** 2 * np.interp(180.0, alpha_tab, C6_tab[:, 0])  # head wind
    Fy_exp = 10.0 ** 2 * np.interp(270.0, alpha_tab, C6_tab[:, 1])  # beam, from east
    crit.add("head wind magnitude: Fx = U^2*Cx(180 deg) within 5%",
             abs(tau[0] - Fx_exp) <= 0.05 * abs(Fx_exp),
             f"Fx = {tau[0]/1e3:.2f} kN vs expected {Fx_exp/1e3:.2f} kN")
    crit.add("beam wind magnitude: Fy = U^2*Cy(270 deg) within 5%",
             abs(tau_e[1] - Fy_exp) <= 0.05 * abs(Fy_exp),
             f"Fy = {tau_e[1]/1e3:.2f} kN vs expected {Fy_exp/1e3:.2f} kN")

    nu_fwd = np.zeros(6)
    nu_fwd[0] = 2.0                                             # sailing into the wind
    w2 = Wind(10.0, 0.0, semantics="from", seed=0)
    tau_fwd, _ = w2.step(0.0, DT, _state6(), nu_fwd)
    tau_fwd = np.asarray(tau_fwd, float).reshape(6)
    crit.add("relative wind: sailing into the wind increases |Fx|",
             tau_fwd[0] < tau[0],
             f"Fx = {tau_fwd[0]/1e3:.2f} kN moving vs {tau[0]/1e3:.2f} kN at rest")

    wv = Wind(10.0, np.pi, semantics="from", sigma_slow=1.0, seed=0)
    Fx_hist = []
    for k in range(int(600.0 / DT)):
        tau_k, _ = wv.step(k * DT, DT, _state6(), np.zeros(6))
        Fx_hist.append(np.asarray(tau_k, float).reshape(6)[0])
    Fx_hist = np.asarray(Fx_hist)
    crit.add("sigma_slow=1.0 gives a slowly-varying load (Part 1 requirement)",
             np.std(Fx_hist) > 0.01 * np.mean(np.abs(Fx_hist)),
             f"std(Fx)/|mean| = {np.std(Fx_hist)/max(np.mean(np.abs(Fx_hist)), 1e-9):.3f}")
    return crit.result(name)


def check_allocation() -> CheckResult:
    """Thrust allocation: isolated wrenches reproduced, limits respected."""
    from part_1.thrust_allocation import ThrustAllocator

    name = "allocation: isolated wrenches"
    cfgs = default_thrusters_gunnerus3()
    cases = [
        ("pure surge 20 kN", _wrench6(Fx=20e3)),
        ("pure sway 20 kN", _wrench6(Fy=20e3)),
        ("pure yaw 200 kNm", _wrench6(Mz=200e3)),
        ("combined 10 kN / 10 kN / 100 kNm", _wrench6(10e3, 10e3, 100e3)),
    ]
    crit = _Criteria()
    for i, (label, tau_d) in enumerate(cases):
        ts = ThrusterSet(cfgs, dynamics=False)                  # ideal actuators
        try:
            u_cmd, a_cmd = ThrustAllocator(cfgs).allocate(
                0.0, DT, tau_d, u_now=ts.get_thrusts(), alpha_now=ts.get_angles())
        except Exception as exc:                                # noqa: BLE001
            return CheckResult(name, "ERROR", [f"allocate() raised: {exc!r}"])
        u_cmd = np.asarray(u_cmd, float)
        if i == 0 and np.allclose(u_cmd, 0.0):
            return CheckResult(name, "NOT IMPLEMENTED",
                               ["allocate() still returns the zero placeholder"])
        _, _, tau_ach = ts.step(u_cmd, a_cmd, DT)
        tau_req = tau_d[[0, 1, 5]]
        err = np.linalg.norm(tau_ach - tau_req) / max(np.linalg.norm(tau_req), 1.0)
        crit.add(f"{label}: achieved wrench within 2%", err < 0.02,
                 f"relative error {100*err:.2f}%")
        within = np.all(np.abs(u_cmd) <= np.array([c.u_max for c in cfgs]) + 1.0)
        crit.add(f"{label}: |u_cmd| within thruster limits", within,
                 f"u_cmd = {np.round(u_cmd/1e3, 1)} kN")
    return crit.result(name)


def check_controller() -> CheckResult:
    """Controller: error signs under rotation and heading wrapping."""
    from part_1.controller import DPController

    name = "controller: sign tests and heading wrap"

    def wrench(psi, dN=0.0, dE=0.0, psi_d=None):
        ctl = DPController()                                    # fresh integrators
        if hasattr(ctl, "reset"):
            ctl.reset()
        eta = _state6(psi)
        ref = eta.copy()
        ref[0] += dN
        ref[1] += dE
        if psi_d is not None:
            ref[5] = psi_d
        tau = ctl.compute(0.0, DT, eta, np.zeros(6), ref, np.zeros(6), np.zeros(6))
        return np.asarray(tau, float).reshape(6)

    try:
        tau_n = wrench(0.0, dN=10.0)
    except Exception as exc:                                    # noqa: BLE001
        return CheckResult(name, "ERROR", [f"compute() raised: {exc!r}"])
    if np.allclose(tau_n, 0.0):
        return CheckResult(name, "NOT IMPLEMENTED",
                           ["compute() still returns the zero placeholder"])

    crit = _Criteria()
    crit.add("psi=0, error +10 m North: Fx > 0 and dominant",
             tau_n[0] > 0.0 and abs(tau_n[0]) > abs(tau_n[1]),
             f"tau = [{tau_n[0]/1e3:.1f}, {tau_n[1]/1e3:.1f}] kN")
    tau_r = wrench(np.pi / 2, dN=10.0)
    crit.add("psi=90 deg, error +10 m North: Fy < 0 and dominant (NED->BODY rotation)",
             tau_r[1] < 0.0 and abs(tau_r[1]) > abs(tau_r[0]),
             f"tau = [{tau_r[0]/1e3:.1f}, {tau_r[1]/1e3:.1f}] kN")
    tau_w1 = wrench(np.deg2rad(170.0), psi_d=np.deg2rad(-170.0))
    crit.add("psi=+170 deg -> psi_d=-170 deg: Mz > 0 (shortest way, +20 deg)",
             tau_w1[5] > 0.0, f"Mz = {tau_w1[5]/1e3:.1f} kNm")
    tau_w2 = wrench(np.deg2rad(-170.0), psi_d=np.deg2rad(170.0))
    crit.add("psi=-170 deg -> psi_d=+170 deg: Mz < 0 (shortest way, -20 deg)",
             tau_w2[5] < 0.0, f"Mz = {tau_w2[5]/1e3:.1f} kNm")
    return crit.result(name)


def check_reference() -> CheckResult:
    """Reference model: smooth step response and heading wrap near +-pi."""
    from part_1.reference import ReferenceModel

    name = "reference model: smooth step and heading wrap"
    rm = ReferenceModel(dt=DT)
    rm.reset(np.zeros(6))
    cmd = np.zeros(6)
    cmd[0] = 10.0
    eta_ref, _, _ = rm.step(0.0, DT, cmd)
    if abs(np.asarray(eta_ref, float).reshape(6)[0] - 10.0) < 0.1:
        return CheckResult(name, "NOT IMPLEMENTED", [
            "step() is still the pass-through placeholder (eta_ref jumps to the"
            " setpoint immediately)"])

    n = int(600.0 / DT)
    N_ref = np.empty(n)
    N_ref[0] = np.asarray(eta_ref, float).reshape(6)[0]
    for k in range(1, n):
        eta_ref, _, _ = rm.step(k * DT, DT, cmd)
        N_ref[k] = np.asarray(eta_ref, float).reshape(6)[0]

    crit = _Criteria()
    crit.add("smooth start: reference below 1 m one second after a 10 m step",
             N_ref[int(1.0 / DT)] < 1.0, f"N_ref(1 s) = {N_ref[int(1.0/DT)]:.2f} m")
    crit.add("overshoot below 10% of the step", np.max(N_ref) < 11.0,
             f"max N_ref = {np.max(N_ref):.2f} m")
    crit.add("converges to the setpoint within 600 s", abs(N_ref[-1] - 10.0) < 0.2,
             f"final N_ref = {N_ref[-1]:.2f} m")

    rm2 = ReferenceModel(dt=DT)
    rm2.reset(_state6(3.0))
    cmd_psi = _state6(-3.0)                                     # shortest way crosses pi
    psi_hist = np.empty(n)
    for k in range(n):
        eta_ref, _, _ = rm2.step(k * DT, DT, cmd_psi)
        psi_hist[k] = np.asarray(eta_ref, float).reshape(6)[5]
    travel = np.sum(np.abs([wrap_angle_pi(d) for d in np.diff(psi_hist)]))
    crit.add("heading 3.0 -> -3.0 rad goes the short way across pi (no 2*pi turn)",
             travel < 1.0, f"total travel {travel:.2f} rad (short way is 0.28)")
    crit.add("heading converges to the wrapped setpoint",
             abs(wrap_angle_pi(psi_hist[-1] - (-3.0))) < 0.05,
             f"final psi_ref = {psi_hist[-1]:.3f} rad")
    return crit.result(name)


# ---------------------------------------------------------------------------
# Mandatory simulations (each runs the full closed loop; seconds per check)
# ---------------------------------------------------------------------------

def _missing_prerequisites(need_current: bool = False, need_wind: bool = False) -> list[str]:
    """Names of required subsystems whose placeholder is still in place."""
    missing = []
    if check_controller().status == "NOT IMPLEMENTED":
        missing.append("controller")
    if check_allocation().status == "NOT IMPLEMENTED":
        missing.append("thrust allocation")
    if need_current and check_current().status == "NOT IMPLEMENTED":
        missing.append("current model")
    if need_wind and check_wind().status == "NOT IMPLEMENTED":
        missing.append("wind model")
    return missing


def _run(cfg: SimConfig, eta_cmd, current=None, wind=None):
    from part_1.controller import DPController
    from .simulation_part_1 import DPSimulator3DOF

    sim = DPSimulator3DOF(cfg, DPController(), default_thrusters_gunnerus3())
    sim.reset_state()
    return sim.run(np.asarray(eta_cmd, dtype=float), current=current, wind=wind)


def _errors(logs, target6, t_from: float, t_to: float | None = None):
    """(mean position error, mean heading error, max position error) on a window.

    The max position error is taken over the whole run, the means over
    [t_from, t_to] — the steady-state window.
    """
    sel = (logs.t >= t_from) if t_to is None else (logs.t >= t_from) & (logs.t <= t_to)
    pos_err = np.hypot(logs.eta[:, 0] - target6[0], logs.eta[:, 1] - target6[1])
    psi_err = np.abs([wrap_angle_pi(p - target6[5]) for p in logs.eta[:, 5]])
    return float(np.mean(pos_err[sel])), float(np.mean(np.asarray(psi_err)[sel])), \
        float(np.max(pos_err))


def _station_keeping_criteria(crit: _Criteria, logs, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    T = logs.t[-1]
    mean_pos, mean_psi, max_pos = _errors(logs, np.zeros(6), T - STEADY_WINDOW)
    crit.add(f"{prefix}all states finite (no NaN)",
             np.all(np.isfinite(logs.eta)) and np.all(np.isfinite(logs.nu)))
    crit.add(f"{prefix}max position deviation < {MAX_DEVIATION:.0f} m",
             max_pos < MAX_DEVIATION, f"max {max_pos:.2f} m")
    crit.add(f"{prefix}steady-state position error < {STEADY_POS:.1f} m"
             f" (last {STEADY_WINDOW:.0f} s)", mean_pos < STEADY_POS,
             f"mean {mean_pos:.2f} m")
    crit.add(f"{prefix}steady-state heading error < {np.rad2deg(STEADY_PSI):.0f} deg",
             mean_psi < STEADY_PSI, f"mean {np.rad2deg(mean_psi):.2f} deg")


def check_sim1_current() -> CheckResult:
    """Simulation 1a: station keeping, current 0.5 m/s from east, no wind."""
    from part_1.current import Current

    name = "Simulation 1a: station keeping in current (0.5 m/s from east)"
    missing = _missing_prerequisites(need_current=True)
    if missing:
        return CheckResult(name, "NOT IMPLEMENTED",
                           [f"requires: {', '.join(missing)}"])
    logs = _run(SimConfig(T=800.0), np.zeros(6),
                current=Current(0.5, np.pi / 2, semantics="from"))
    crit = _Criteria()
    _station_keeping_criteria(crit, logs)
    return crit.result(name, {"current 0.5 m/s from east": logs})


def check_sim1_wind() -> CheckResult:
    """Simulation 1b: station keeping, wind 15 m/s mean from east, no current."""
    from part_1.wind import Wind

    name = "Simulation 1b: station keeping in wind (15 m/s from east)"
    missing = _missing_prerequisites(need_wind=True)
    if missing:
        return CheckResult(name, "NOT IMPLEMENTED",
                           [f"requires: {', '.join(missing)}"])
    logs = _run(SimConfig(T=800.0), np.zeros(6),
                wind=Wind(15.0, np.pi / 2, semantics="from", sigma_slow=1.0, seed=0))
    crit = _Criteria()
    _station_keeping_criteria(crit, logs)
    return crit.result(name, {"wind 15 m/s from east": logs})


def check_sim2_rotating_current() -> CheckResult:
    """Simulation 2: current rotating from north to east over 300 s, no wind."""
    from part_1.current import Current

    name = "Simulation 2: rotating current (north -> east over 300 s)"
    missing = _missing_prerequisites(need_current=True)
    if missing:
        return CheckResult(name, "NOT IMPLEMENTED",
                           [f"requires: {', '.join(missing)}"])
    logs = _run(SimConfig(T=900.0), np.zeros(6),
                current=Current(0.5, 0.0, semantics="from",
                                beta_end=np.pi / 2, duration=300.0))
    crit = _Criteria()
    _station_keeping_criteria(crit, logs)
    return crit.result(name, {"rotating current": logs})


def check_sim3_setpoint_change() -> CheckResult:
    """Simulation 3: step to [10, 10, 3*pi/2], with and without reference model."""
    name = "Simulation 3: setpoint change with/without reference model"
    missing = _missing_prerequisites()
    if missing:
        return CheckResult(name, "NOT IMPLEMENTED",
                           [f"requires: {', '.join(missing)}"])
    target = np.zeros(6)
    target[0], target[1], target[5] = 10.0, 10.0, 3.0 * np.pi / 2.0

    logs = {}
    crit = _Criteria()
    for label, use_ref in [("with reference model", True),
                           ("without reference model", False)]:
        lg = _run(SimConfig(T=600.0, use_reference=use_ref), target)
        logs[label] = lg
        T = lg.t[-1]
        mean_pos, mean_psi, _ = _errors(lg, target, T - 50.0)
        crit.add(f"{label}: settles at the setpoint (pos err < {SETTLE_POS} m)",
                 mean_pos < SETTLE_POS, f"mean {mean_pos:.2f} m over the last 50 s")
        crit.add(f"{label}: heading settles (err < 2 deg, wrapped)",
                 mean_psi < np.deg2rad(2.0), f"mean {np.rad2deg(mean_psi):.2f} deg")
    # Overshoot along the approach direction of each channel (start -> target),
    # so a target in the negative direction is measured the same way.
    lg_ref = logs["with reference model"]
    overshoot = max(
        float(np.max(np.sign(target[i] - lg_ref.eta[0, i])
                     * (lg_ref.eta[:, i] - target[i])))
        for i in (0, 1)
    )
    crit.add("with reference model: position overshoot < 2 m",
             overshoot < 2.0, f"overshoot {overshoot:.2f} m")
    return crit.result(name, logs)


def check_sim4_four_corner() -> CheckResult:
    """Simulation 4: four-corner test, timed legs, settle at every corner."""
    name = "Simulation 4: four-corner test"
    missing = _missing_prerequisites()
    if missing:
        return CheckResult(name, "NOT IMPLEMENTED",
                           [f"requires: {', '.join(missing)}"])
    corners = [
        ("eta1 = [50, 0, 0]", [50.0, 0.0, 0.0]),
        ("eta2 = [50, -50, 0]", [50.0, -50.0, 0.0]),
        ("eta3 = [50, -50, -pi/4]", [50.0, -50.0, -np.pi / 4]),
        ("eta4 = [0, -50, -pi/4]", [0.0, -50.0, -np.pi / 4]),
        ("eta5 = [0, 0, 0]", [0.0, 0.0, 0.0]),
    ]
    hold = 300.0                                    # [s] per leg (timed switching)
    cfg = SimConfig(T=hold * len(corners))
    n_steps = int(round(cfg.T / cfg.dt)) + 1
    per_leg = int(round(hold / cfg.dt))
    eta_cmd = np.zeros((n_steps, 6))
    for i, (_, c) in enumerate(corners):
        rows = slice(i * per_leg, n_steps if i == len(corners) - 1 else (i + 1) * per_leg)
        eta_cmd[rows, 0], eta_cmd[rows, 1], eta_cmd[rows, 5] = c

    logs = _run(cfg, eta_cmd)
    crit = _Criteria()
    crit.add("all states finite (no NaN)",
             np.all(np.isfinite(logs.eta)) and np.all(np.isfinite(logs.nu)))
    for i, (label, c) in enumerate(corners):
        t_end = (i + 1) * hold
        target = np.zeros(6)
        target[0], target[1], target[5] = c
        mean_pos, mean_psi, _ = _errors(logs, target, t_end - 50.0, t_end)
        crit.add(f"{label}: settled before the switch (pos err < {STEADY_POS} m)",
                 mean_pos < STEADY_POS, f"mean {mean_pos:.2f} m over the last 50 s")
        crit.add(f"{label}: heading settled (err < 2 deg)",
                 mean_psi < np.deg2rad(2.0), f"mean {np.rad2deg(mean_psi):.2f} deg")
    return crit.result(name, {"four-corner test": logs})


# ---------------------------------------------------------------------------
# Registry and reporting
# ---------------------------------------------------------------------------

FAST_CHECKS = {
    "current": check_current,
    "wind": check_wind,
    "allocation": check_allocation,
    "controller": check_controller,
    "reference": check_reference,
}

SIMULATION_CHECKS = {
    "sim1_current": check_sim1_current,
    "sim1_wind": check_sim1_wind,
    "sim2_rotating_current": check_sim2_rotating_current,
    "sim3_setpoint_change": check_sim3_setpoint_change,
    "sim4_four_corner": check_sim4_four_corner,
}

ALL_CHECKS = {**FAST_CHECKS, **SIMULATION_CHECKS}


def run_check(key: str) -> CheckResult:
    """Run one check by registry key, converting crashes into ERROR results."""
    func = ALL_CHECKS[key]
    try:
        return func()
    except Exception as exc:                                    # noqa: BLE001
        name = (func.__doc__ or key).splitlines()[0].rstrip(".")
        return CheckResult(name, "ERROR", [f"check crashed: {exc!r}"])


def run_all(fast_only: bool = False) -> list[CheckResult]:
    keys = FAST_CHECKS if fast_only else ALL_CHECKS
    return [run_check(k) for k in keys]


def print_result(r: CheckResult) -> None:
    print(f"[{r.status:^15}] {r.name}")
    for line in r.details:
        print(f"                  {line}")


def print_report(results: list[CheckResult]) -> bool:
    """Print all results plus a summary line; return True if everything passed."""
    for r in results:
        print_result(r)
        print()
    n_pass = sum(r.passed for r in results)
    print(f"{n_pass}/{len(results)} checks passed "
          f"({', '.join(sorted({r.status for r in results if not r.passed})) or 'all good'})")
    return n_pass == len(results)
