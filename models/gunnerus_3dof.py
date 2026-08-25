# simulator/models/gunnerus_3dof.py
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

# mcsimpy model class
from mcsimpy.simulator.gunnerus import GunnerusManeuvering3DoF as _G


from importlib.resources import files


class Gunnerus3DOF:
    """Thin wrapper around mcsimpy’s 3-DOF Gunnerus with a stable interface.

    Only ``method="Euler"`` is supported: mcsimpy's ``x_dot`` computes the
    kinematics from the stored ``self._nu`` rather than the integration
    variable, so multi-stage integrators (RK4) evaluate stale velocities and
    give silently wrong results. The constructor therefore rejects any other
    method instead of letting it corrupt the simulation.
    """
    def __init__(self, dt: float = 0.05, method: str = "Euler", config_file: str | None = None):
        if method.strip().lower() != "euler":
            raise ValueError(
                f"Gunnerus3DOF supports method='Euler' only, got {method!r}: "
                "mcsimpy evaluates multi-stage integrators (e.g. RK4) on stale "
                "velocities, giving silently wrong results.")
        # a caller-supplied config_file overrides the packaged mcsimpy PKL
        cf = config_file if config_file is not None \
            else files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        self._v = _G(dt=dt, method="Euler", config_file=cf)

    # passthrough API
    def get_eta(self): return self._v.get_eta()
    def get_nu(self):  return self._v.get_nu()
    def set_eta(self, e): self._v.set_eta(e)
    def set_nu(self, n):  self._v.set_nu(n)
    def integrate(self, *, Uc: float, beta_c: float, tau):
        return self._v.integrate(Uc=Uc, beta_c=beta_c, tau=tau)
