"""Mandatory Simulations 1-4 as closed-loop checks.

Each test wraps one simulation check from ``simulation/checks.py``.  A
simulation whose required subsystems are still template placeholders is
reported as skipped, not failed.
"""
import pytest

from simulation.checks import SIMULATION_CHECKS, run_check


@pytest.mark.parametrize("key", list(SIMULATION_CHECKS), ids=list(SIMULATION_CHECKS))
def test_simulation(key):
    result = run_check(key)
    detail = "\n".join([result.name, *result.details])
    if result.status == "NOT IMPLEMENTED":
        pytest.skip(detail)
    assert result.passed, detail
