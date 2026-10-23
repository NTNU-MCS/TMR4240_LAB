import numpy as np
from part_1.config import default_thrusters_gunnerus3
from part_1.thrust_allocation import ThrustAllocator, ThrustAllocatorBaseline


def test_build_Be_matches_hand_calculation():
    ta = ThrustAllocatorBaseline(default_thrusters_gunnerus3())

    expected = np.array([
        [0,1,0,1,0],
        [1,0,1,0,1],
        [12.0,-3.0,-13.0,3.0,-13.0]
    ])

    assert ta.Be.shape == expected.shape
    assert np.allclose(ta.Be, expected)

def test_allocation_slsqp_isolated_wrenches():
    """Mirrors simulation/checks.py::check_allocation, for the SLSQP-based ThrustAllocator."""
    from models.thruster_dynamics import ThrusterSet

    cfgs = default_thrusters_gunnerus3()
    cases = [
        ("pure surge 20 kN", np.array([20e3, 0, 0, 0, 0, 0])),
        ("pure sway 20 kN", np.array([0, 20e3, 0, 0, 0, 0])),
        ("pure yaw 200 kNm", np.array([0, 0, 0, 0, 0, 200e3])),
        ("combined 10 kN / 10 kN / 100 kNm", np.array([10e3, 10e3, 0, 0, 0, 100e3])),
    ]

    allocator = ThrustAllocator(cfgs)
    for label, tau_d in cases:
        ts = ThrusterSet(cfgs, dynamics=False)
        u_cmd, a_cmd = allocator.allocate(
            0.0, 0.05, tau_d, u_now=ts.get_thrusts(), alpha_now=ts.get_angles())

        _, _, tau_ach = ts.step(u_cmd, a_cmd, 0.05)
        tau_req = tau_d[[0, 1, 5]]
        err = np.linalg.norm(tau_ach - tau_req) / max(np.linalg.norm(tau_req), 1.0)
        assert err < 0.02, f"{label}: relative error {100*err:.2f}%"

        within = np.all(np.abs(u_cmd) <= np.array([c.u_max for c in cfgs]) + 1.0)
        assert within, f"{label}: u_cmd = {np.round(u_cmd/1e3, 1)} kN"