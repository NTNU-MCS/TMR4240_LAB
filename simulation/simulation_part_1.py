# simulation_part_1.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System 
#
# Author: Saber Sakhrieh
# Date: 10 June 2026
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
from dataclasses import dataclass
import numpy as np

from part_1.config import SimConfig
from .plant import LoadInput, body_wrench
from .utils import Rz, to_3dof, to_6dof
from models.gunnerus_3dof import Gunnerus3DOF  # mcsimpy wrapper
from models.thruster_dynamics import ThrusterSet, ThrusterConfig
from part_1.controller import DPController
from part_1.reference import ReferenceModel
from part_1.thrust_allocation import ThrustAllocator
from part_1.current import Current
from part_1.wind import Wind


@dataclass
class Logs:
    """All time histories returned by a simulation run.

    Every generalized vector is logged 6-DOF, ordered
    [surge, sway, heave, roll, pitch, yaw]; the 3-DOF model populates
    indices [0, 1, 5] and the remaining components are zero.
    """
    t: np.ndarray
    eta: np.ndarray
    nu: np.ndarray
    sp: np.ndarray
    cmd: np.ndarray
    tau_d: np.ndarray
    tau_thr: np.ndarray
    tau_total: np.ndarray
    u: np.ndarray
    alpha: np.ndarray
    cur_body: np.ndarray
    Uc: np.ndarray
    beta_c: np.ndarray
    U_w: np.ndarray
    alpha_w: np.ndarray
    beta_w: np.ndarray
    tau_w6: np.ndarray
    tau_wave: np.ndarray
    thruster_names: list[str]

    I_ned: np.ndarray     # integrator state for N/E (m·s)
    I_psi: np.ndarray     # integrator state for yaw (rad·s)
    tau_P: np.ndarray     # (N,6) BODY contributions from P
    tau_I: np.ndarray     # (N,6) BODY contributions from I
    tau_D: np.ndarray     # (N,6) BODY contributions from D
    nu_ref: np.ndarray    # (N,6) BODY-mapped reference velocities (NaN when ref off)
    acc_ref: np.ndarray   # (N,6) NED reference accelerations (NaN when ref off)


class DPSimulator3DOF:
    """
    Pipeline:
      controller -> allocation -> thruster dynamics -> vessel (mcsimpy)
      + optional reference model, ambient current & wind loads.

    Construct once with (cfg, controller, thrusters, reference, pkl_path).
    Run many scenarios by calling run(eta_cmd, current=..., wind=..., ...).
    """
    def __init__(self,
                 cfg: SimConfig,
                 controller: DPController,
                 thrusters: list[ThrusterConfig],
                 *,
                 reference: ReferenceModel | None = None,
                 pkl_path: str | None = None):
        self.cfg = cfg
        self.controller = controller
        self.thruster_cfgs = thrusters

        # Vessel (wrapper loads the packaged mcsimpy PKL unless pkl_path overrides it)
        self.vessel = Gunnerus3DOF(dt=cfg.dt, method=cfg.method, config_file=pkl_path)
        self.vessel.set_eta(np.zeros(3))
        self.vessel.set_nu(np.zeros(3))

        # Actuation chain
        self.allocator = ThrustAllocator(thrusters)
        self.thrusters = ThrusterSet(thrusters, dynamics=cfg.thruster_dynamics)
        self.thruster_names = [th.name for th in thrusters]

        # Reference model (enabled by cfg.use_reference). The instance — and
        # therefore all its parameters — is YOURS: configure it in run_case_part1.py
        # and pass it in via `reference=`. If none is given, the model from
        # part_1/reference.py is built with its own defaults. Nothing is tuned
        # inside the engine.
        self.ref = None
        if cfg.use_reference:
            self.ref = reference if reference is not None else ReferenceModel(dt=cfg.dt)

    def reset_state(self, eta0: np.ndarray | None = None, nu0: np.ndarray | None = None):
        """Reset vessel, actuator, reference, and controller state before a run
        (6-DOF vectors, zeros if omitted)."""
        self.vessel.set_eta(np.zeros(3) if eta0 is None else to_3dof(eta0))
        self.vessel.set_nu(np.zeros(3)  if nu0  is None else to_3dof(nu0))
        self.thrusters.reset()
        if self.ref is not None:
            self.ref.reset(to_6dof(self.vessel.get_eta()))
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def run(self,
            eta_cmd: np.ndarray,
            *,
            current: Current | None = None,
            wind: Wind | None = None,
            waves: LoadInput = None) -> Logs:
        """
        eta_cmd: (6,) constant setpoint or (N,6) time-series
                 [N, E, z, phi, theta, psi] (psi in rad). The 3-DOF model
                 uses only [N, E, psi] = indices [0, 1, 5].
        current: instance from part_1/current.py (default: off).
        wind   : instance from part_1/wind.py (default: off).
        waves  : generalized BODY load (6,) [Fx, Fy, Fz, Mx, My, Mz], or a
                 callable ``waves(t, eta, nu)`` or ``waves(t, eta)`` returning
                 that vector. Callables receive the 6-DOF state vectors
                 (psi = eta[5]), matching the plant API.
        """
        cfg = self.cfg
        # default: inactive environment (zero current, zero wind)
        current = current if current is not None else Current()
        wind = wind if wind is not None else Wind()

        # exact time axis: every logged t is a true integration instant
        # (linspace would silently diverge from k*dt when T is not an
        # integer multiple of dt)
        n_steps = int(round(cfg.T / cfg.dt)) + 1
        t = np.arange(n_steps) * cfg.dt

        eta_hist = np.zeros((n_steps, 6))
        nu_hist  = np.zeros((n_steps, 6))
        cmd_hist = np.zeros((n_steps, 6))
        sp_hist  = np.zeros((n_steps, 6))

        tau_d_hist     = np.zeros((n_steps, 6))
        tau_thr_hist   = np.zeros((n_steps, 6))
        tau_total_hist = np.zeros((n_steps, 6))

        n_thr = len(self.thruster_names)
        u_hist = np.zeros((n_steps, n_thr))
        a_hist = np.zeros((n_steps, n_thr))

        cur_body_hist = np.zeros((n_steps, 2))
        Uc_hist       = np.zeros(n_steps)
        beta_c_hist   = np.zeros(n_steps)

        U_w_hist      = np.zeros(n_steps)
        alpha_w_hist  = np.zeros(n_steps)
        beta_w_hist   = np.zeros(n_steps)
        tau_w6_hist   = np.zeros((n_steps, 6))
        tau_wave_hist = np.zeros((n_steps, 6))

        I_ned_hist = np.zeros((n_steps, 2))
        I_psi_hist = np.zeros(n_steps)
        tau_P_hist = np.zeros((n_steps, 6))
        tau_I_hist = np.zeros((n_steps, 6))
        tau_D_hist = np.zeros((n_steps, 6))
        nu_ref_hist = np.full((n_steps, 6), np.nan)   # NaN when ref is off
        acc_ref_hist = np.full((n_steps, 6), np.nan)  # NaN when ref is off


        # Helper for time-varying commands (6-DOF setpoints). A series shorter
        # than the run holds its last row instead of raising IndexError (the
        # engine takes round(T/dt)+1 steps, which can differ by one from a
        # series built with ceil or linspace).
        def _cmd_at(k):
            ec = np.asarray(eta_cmd, dtype=float)
            if ec.ndim == 1:
                return ec.reshape(6)
            return ec[min(k, ec.shape[0] - 1), :].reshape(6)

        # init reference
        if self.ref is not None:
            self.ref.reset(to_6dof(self.vessel.get_eta()))

        for k in range(n_steps):
            # Full 6-DOF state (psi = eta[5]); the 3-DOF vessel model
            # populates indices [0, 1, 5].
            eta = to_6dof(self.vessel.get_eta())
            nu  = to_6dof(self.vessel.get_nu())
            eta_hist[k] = eta
            nu_hist[k]  = nu

            cmd_k = _cmd_at(k)
            cmd_hist[k] = cmd_k

            if self.ref is not None:
                sp_k, nu_ref_ned, acc_ref_ned = self.ref.step(t[k], cfg.dt, cmd_k)
                acc_ref_hist[k, :] = acc_ref_ned
                # map N/E rates to BODY for comparison
                R = Rz(eta[5])                             # NED = R @ BODY
                uv_ref_body = R.T @ np.array([nu_ref_ned[0], nu_ref_ned[1], 0.0])
                nu_ref_hist[k, :] = 0.0                    # ref on: zeros, not NaN
                nu_ref_hist[k, 0] = uv_ref_body[0]         # u_ref
                nu_ref_hist[k, 1] = uv_ref_body[1]         # v_ref
                nu_ref_hist[k, 5] = nu_ref_ned[5]          # r_ref (same in BODY)
            else:
                sp_k = cmd_k
                nu_ref_ned = np.zeros(6)
                acc_ref_ned = np.zeros(6)
            sp_hist[k] = sp_k

            # controller → desired BODY wrench (full state + full reference)
            tau_d = np.asarray(
                self.controller.compute(t[k], cfg.dt, eta, nu,
                                        sp_k, nu_ref_ned, acc_ref_ned),
                dtype=float).reshape(6)
            tau_d_hist[k] = tau_d
            # log PID component breakdown if the controller exposes one (optional)
            pid = getattr(self.controller, "last_pid_body", None)
            if pid is not None:
                tau_P_hist[k, :] = pid["P"]
                tau_I_hist[k, :] = pid["I"]
                tau_D_hist[k, :] = pid["D"]

            # Allocation + thruster dynamics (or bypass)
            if cfg.bypass_actuators:
                tau_thr = tau_d
                a_act = self.thrusters.get_angles()
                u_act = np.zeros_like(a_act)
            else:
                u_cmd, a_cmd = self.allocator.allocate(
                    t[k], cfg.dt, tau_d,
                    u_now=self.thrusters.get_thrusts(),
                    alpha_now=self.thrusters.get_angles())
                u_act, a_act, tau_thr3 = self.thrusters.step(u_cmd, a_cmd, cfg.dt)
                tau_thr = to_6dof(tau_thr3)

            u_hist[k], a_hist[k] = u_act, a_act
            tau_thr_hist[k] = tau_thr

            # Wind loads from the full 6-DOF state
            tau_w6, winfo = wind.step(t[k], cfg.dt, eta, nu)
            tau_w6 = np.asarray(tau_w6, dtype=float).reshape(6)
            winfo = winfo or {}
            U_w_hist[k]     = winfo.get("U", 0.0)
            alpha_w_hist[k] = winfo.get("alpha_body", 0.0)
            beta_w_hist[k]  = winfo.get("beta_ned", 0.0)
            tau_w6_hist[k]  = tau_w6

            # Wave loads are supplied directly by the selected wave model.
            # This keeps the controller loop independent of a particular sea-state API.
            tau_wave = body_wrench(waves, t[k], eta, nu)
            tau_wave_hist[k] = tau_wave

            # Total wrench for the plant
            tau_total = tau_thr + tau_w6 + tau_wave
            tau_total_hist[k] = tau_total

            # External anti-windup based on applied thruster wrench (optional hook)
            if hasattr(self.controller, "apply_external_aw"):
                self.controller.apply_external_aw(tau_thr, psi=eta[5], dt=cfg.dt)

            # Log integrator states if the controller exposes them (optional)
            I_ned_hist[k, :] = getattr(self.controller, "int_ned", np.zeros(2))
            I_psi_hist[k]    = getattr(self.controller, "int_psi", 0.0)

            # Ambient current: generalized NED velocity vector (6,)
            nu_c_ned = np.asarray(
                current.step(t[k], cfg.dt, eta, nu), dtype=float).reshape(6)
            Uc = float(np.hypot(nu_c_ned[0], nu_c_ned[1]))
            beta_c = float(np.arctan2(nu_c_ned[1], nu_c_ned[0]))
            cur_body_hist[k] = (Rz(eta[5]).T @ nu_c_ned[:3])[:2]
            Uc_hist[k] = Uc; beta_c_hist[k] = beta_c

            # Integrate plant (reduce to the 3-DOF vessel model at this
            # boundary only)
            self.vessel.integrate(Uc=Uc, beta_c=beta_c, tau=to_3dof(tau_total))

        return Logs(
            t=t, eta=eta_hist, nu=nu_hist,
            sp=sp_hist, cmd=cmd_hist, tau_d=tau_d_hist,
            tau_thr=tau_thr_hist, tau_total=tau_total_hist,
            u=u_hist, alpha=a_hist,
            cur_body=cur_body_hist, Uc=Uc_hist, beta_c=beta_c_hist,
            U_w=U_w_hist, alpha_w=alpha_w_hist, beta_w=beta_w_hist, tau_w6=tau_w6_hist,
            tau_wave=tau_wave_hist,
            thruster_names=self.thruster_names,
            I_ned=I_ned_hist, I_psi=I_psi_hist,
            tau_P=tau_P_hist, tau_I=tau_I_hist, tau_D=tau_D_hist,
            nu_ref=nu_ref_hist, acc_ref=acc_ref_hist
        )
