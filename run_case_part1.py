"""
run_case template

Students can edit this file to define:
    - simulation time and time step,
    - controller settings,
    - reference model settings,
    - setpoint or trajectory,
    - current and wind models,
    - plots to show.

"""
import matplotlib.pyplot as plt
import numpy as np

from part_1.config import SimConfig, default_thrusters_gunnerus3
from simulation.simulation_part_1 import DPSimulator3DOF
from simulation.plotter import plot_dashboard, plot_time_histories
from part_1.controller import DPController
from part_1.reference import ReferenceModel
from part_1.current import Current
from part_1.wind import Wind




def main():
    # 1) Simulation clock and options
    cfg = SimConfig(dt=0.05, T=300.0, method="Euler", use_reference=True)

    # 2) Controller, reference model, and thruster layout
    # Pass your own design parameters (gains, limits, ...) to your controller.
    controller = DPController()

    # Reference model. Tune it via the RefAxisConfig defaults in
    # part_1/config.py — NOT by passing values here: the automated checks
    # build ReferenceModel(dt) from those defaults, so parameters overridden
    # only in this file never reach the checks.
    reference = ReferenceModel(dt=cfg.dt)

    thrusters = default_thrusters_gunnerus3()

    # 3) Create simulator
    sim = DPSimulator3DOF(cfg, controller, thrusters,
                          reference=reference, pkl_path=None)

    # 4) Define setpoint or trajectory (6-DOF: [N, E, z, phi, theta, psi]).
    # The 3-DOF model uses N = eta_cmd[0], E = eta_cmd[1], psi = eta_cmd[5];
    # leave the other components zero.
    # Constant setpoint example:
    eta_cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Students may replace eta_cmd with a time series of shape (N_steps, 6).

    # 5) Define environment models (default: calm water)
    current = Current()
    wind = Wind()

    # Simulation 1a from the project description — station keeping at the
    # origin in a 0.5 m/s current from east, no wind. Once your subsystems
    # are implemented, uncomment these two lines (and set T=800.0 above):
    # current = Current(0.5, np.pi / 2, semantics="from")
    # wind = Wind()

    # 6) Run simulation
    sim.reset_state()
    logs = sim.run(eta_cmd, current=current, wind=wind)

    # 7) Plot results
    # See simulation/plotter.py for more: plot_xy, plot_thrusters,
    # plot_wrench, plot_current, plot_wind.
    plot_dashboard(logs)
    plot_time_histories(logs)
    plt.show()

    # Confirmation
    print("Simulation finished.")

if __name__ == "__main__":
    main()
