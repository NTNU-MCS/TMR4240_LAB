"""Simulation infrastructure and plant API."""

from .utils import SURGE_SWAY_YAW, to_3dof, to_6dof

__all__ = [
    "GunnerusPlant3DOF", "PlantStep", "body_wrench",
    "SURGE_SWAY_YAW", "to_3dof", "to_6dof",
]


def __getattr__(name: str):
    """Load plant classes lazily to avoid the models <-> simulation import cycle."""
    if name in {"GunnerusPlant3DOF", "PlantStep", "body_wrench"}:
        from .plant import GunnerusPlant3DOF, PlantStep, body_wrench

        return {
            "GunnerusPlant3DOF": GunnerusPlant3DOF,
            "PlantStep": PlantStep,
            "body_wrench": body_wrench,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
