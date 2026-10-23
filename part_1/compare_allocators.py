"""
Compare the baseline (pseudo-inverse + uniform de-rating) and the improved
(SLSQP, true disc-constraint) thrust allocators.

Run manually to (re)generate the comparison figures/tables for the report:

    python part_1/compare_allocators.py

This is NOT a pytest test file
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from part_1.thrust_allocation import ThrustAllocator, ThrustAllocatorBaseline
from part_1.config import default_thrusters_gunnerus3


def achieved_wrench(allocator, u_cmd, alpha_cmd):
    """Reconstruct the 3-DOF wrench [Fx, Fy, Mz] actually delivered by u_cmd/alpha_cmd."""
    z = []
    for thr, u, a in zip(allocator.thrusters, u_cmd, alpha_cmd):
        if thr.kind == "tunnel":
            z.append(u)
        else:  # azimuth
            z.append(u * np.cos(a))
            z.append(u * np.sin(a))
    return allocator.Be @ np.array(z)


def plot_capacity_envelope(baseline, improved, out_path, M=5e5, n_angles=72):
    """Sweep direction in the Fx-Fy plane and plot the achievable force
    magnitude before saturation, for both allocators, on a polar plot.

    M is chosen far above the summed u_max of all thrusters so that every
    direction is guaranteed to saturate -- this makes the plotted radius
    the true capacity boundary, not an arbitrary point below it.
    """
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cap_baseline = np.zeros_like(angles)
    cap_improved = np.zeros_like(angles)

    for i, theta in enumerate(angles):
        tau_d = np.zeros(6)
        tau_d[0] = M * np.cos(theta)
        tau_d[1] = M * np.sin(theta)

        u_b, a_b = baseline.allocate(0.0, 0.1, tau_d)
        u_i, a_i = improved.allocate(0.0, 0.1, tau_d)

        cap_baseline[i] = np.hypot(*achieved_wrench(baseline, u_b, a_b)[:2])
        cap_improved[i] = np.hypot(*achieved_wrench(improved, u_i, a_i)[:2])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    ax.plot(angles, cap_baseline, label="Baseline (pseudo-inverse + uniform de-rating)")
    ax.plot(angles, cap_improved, label="Improved (SLSQP, true disc constraint)")
    ax.set_title("Achievable force capacity by direction [N]")
    ax.legend(loc="lower right", bbox_to_anchor=(1.05, -0.05), fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved capacity envelope plot to {out_path}")


def compare_on_test_cases(baseline, improved, test_cases):
    """Print a table of achieved-vs-requested error and solve time for each
    allocator on a set of representative wrench demands."""
    header = f"{'case':>10} | {'allocator':>9} | {'error':>8} | {'solve_time':>10} | {'SLSQP used':>10}"
    print(header)
    print("-" * len(header))

    for name, tau_d in test_cases.items():
        for label, allocator in [("baseline", baseline), ("improved", improved)]:
            t0 = time.perf_counter()
            u, a = allocator.allocate(0.0, 0.1, tau_d)
            dt_solve = time.perf_counter() - t0

            achieved = achieved_wrench(allocator, u, a)
            requested = tau_d[[0, 1, 5]]
            err = np.linalg.norm(achieved - requested) / max(np.linalg.norm(requested), 1e-9)

            # baseline has no last_used_optimizer attribute -> shows as "-"
            used = getattr(allocator, "last_used_optimizer", None)
            used_str = "-" if used is None else str(used)

            print(f"{name:>10} | {label:>9} | {err:7.2%} | {dt_solve * 1e3:8.2f} ms | {used_str:>10}")


def main():
    thrusters = default_thrusters_gunnerus3()
    baseline = ThrustAllocatorBaseline(thrusters)
    improved = ThrustAllocator(thrusters)

    plot_capacity_envelope(baseline, improved, out_path="figures/capacity_polar.png")

    # Scaled up from the original values so that "surge"/"sway"/"yaw"/"combined"
    # actually saturate at least one thruster and trigger SLSQP in `improved`.
    # If "SLSQP used" shows False for a case, that demand is still within the
    # pseudo-inverse baseline's capacity
    test_cases = {
        "surge":    np.array([200000, 0, 0, 0, 0, 0]),
        "sway":     np.array([0, 200000, 0, 0, 0, 0]),
        "yaw":      np.array([0, 0, 0, 0, 0, 800000]),
        "combined": np.array([150000, 100000, 0, 0, 0, 600000]),
    }
    compare_on_test_cases(baseline, improved, test_cases)

    print()
    print(f"improved: n_calls={improved.n_calls}, n_triggered={improved.n_triggered}, "
          f"n_optimized={improved.n_optimized}, n_fallback={improved.n_fallback}")


if __name__ == "__main__":
    main()