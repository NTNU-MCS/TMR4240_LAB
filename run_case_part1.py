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
from simulation.plotter import plot_wind





def main():
    hold = 700.0  # [s] per hjørne — låst av spesifikasjonen, ikke et valg

    # corners = [
    #     [50.0,   0.0,  0.0],
    #     [50.0, -50.0,  0.0],
    #     [50.0, -50.0, -np.pi / 4],
    #     [ 0.0, -50.0, -np.pi / 4],
    #     [ 0.0,   0.0,  0.0],
    # ]
    # 1) Simulation clock and options
    # 1a and 1b : T=800, 2 T=800, 3 T=600 4 T=hold*len(corners)
    cfg = SimConfig(dt=0.05, T=800.0, method="Euler", use_reference=True)

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

    # n_steps = round(cfg.T / cfg.dt) + 1
    # per_leg = int(round(hold / cfg.dt))
    # eta_cmd = np.zeros((n_steps, 6))

    eta_cmd = np.zeros(6)
    # Students may replace eta_cmd with a time series of shape (N_steps, 6).

    # for i, c in enumerate(corners):
    #     start = i * per_leg
    #     end = n_steps if i == len(corners) - 1 else (i + 1) * per_leg
    #     eta_cmd[start:end, 0] = c[0]   # N
    #     eta_cmd[start:end, 1] = c[1]   # E
    #     eta_cmd[start:end, 5] = c[2]   # psi

    # 5) Define environment models (default: calm water)
    #current = Current()
    #wind = Wind()


    # Simulation 1a from the project description — station keeping at the
    # origin in a 0.5 m/s current from east, no wind. Once your subsystems
    # are implemented, uncomment these two lines (and set T=800.0 above):

    current = Current()
    wind = Wind(15.0, np.pi / 2, semantics="from",
                sigma_slow=1.0, seed=42)
    #wind = Wind(mean_speed=15, beta= np.pi/2, semantics="from", sigma_slow= 1)

    # 6) Run simulation
    sim.reset_state()
    logs = sim.run(eta_cmd, current=current, wind=wind)

    print("Fraction of samples at/near saturation:")
    limits_kN = {"Tunnel_Bow": 32.0, "Azimuth_1": 80.0, "Azimuth_2": 80.0}
    for i, name in enumerate(logs.thruster_names):
        u_kN = np.abs(logs.u[:, i]) / 1000.0
        frac_sat = np.mean(u_kN >= 0.99 * limits_kN[name])
        print(f"  {name}: saturated {frac_sat*100:.1f}% of the run "
            f"(peak {u_kN.max():.1f} kN of {limits_kN[name]} kN)")

    print("\nPeak |controller wrench| (before allocation):")
    print(f"  Fx: {np.max(np.abs(logs.tau_d[:,0]))/1000:.1f} kN")
    print(f"  Fy: {np.max(np.abs(logs.tau_d[:,1]))/1000:.1f} kN")
    print(f"  Mz: {np.max(np.abs(logs.tau_d[:,5]))/1000:.1f} kNm")


    # 7) Plot results
    # See simulation/plotter.py for more: plot_xy, plot_thrusters,
    # plot_wrench, plot_current, plot_wind.
    plot_dashboard(logs)
    plot_time_histories(logs)
    plot_wind(logs)
    plt.show()

    # Confirmation
    print("Simulation finished.")

if __name__ == "__main__":
    main()
