"""Fast subsystem checks: sign conventions, heading wrap, allocation.

Each test wraps one check from ``simulation/checks.py``.  A subsystem whose
template placeholder is still in place is reported as skipped, not failed.
"""
import pytest

from simulation.checks import FAST_CHECKS, run_check


@pytest.mark.parametrize("key", list(FAST_CHECKS), ids=list(FAST_CHECKS))
def test_subsystem(key):
    result = run_check(key)
    detail = "\n".join([result.name, *result.details])
    if result.status == "NOT IMPLEMENTED":
        pytest.skip(detail)
    assert result.passed, detail
