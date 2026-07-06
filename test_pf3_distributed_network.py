"""
Test PF3 Distributed Network with turbine guide vane step.

This script tests the PF3 distributed network with a step change in turbine
guide vane opening:
1. Start at steady state (y_T = 0.47059)
2. At t = 2s, step y_T down to 0.4
3. Observe transient response for 6 more seconds

Initial operating point (from SIMSEN):
- y_T = 0.47059 (turbine guide vane)
- N_T = 369.3346 rpm (turbine speed)
- N_P = -313.2579 rpm (pump speed, both pumps)
- H_T ≈ 5.0 m (turbine head)
- Q_T ≈ 0.196 m³/s (turbine flow)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.characteristics import Characteristics
from src.pf3_network import build_pf3_full


def create_turbine_and_pump():
    """Load real turbine and pump characteristic curves."""
    # Turbine: Francis turbine from SIMSEN data
    # Parameters from FTURB1.DAT
    turbine = Characteristics(
        d_ref=0.3477,
        h_n=5.0,
        q_n=0.1959,
        t_n=224.057,
        n_n=369.3346,
        char_file=Path("data/pf3/missing_files/STORE_4_quadrant_characteristic.txt"),
    )

    # Pump: Centrifugal pump from SIMSEN data
    # Parameters from PUMP1.DAT
    pump = Characteristics(
        d_ref=0.535,
        h_n=50.0,
        q_n=1.30,
        t_n=1280.3,
        n_n=790.0,
        char_file=Path("data/pf3/missing_files/PF3_FP_D_535_ext.txt"),
    )

    return turbine, pump


def main():
    print("=" * 70)
    print("PF3 Distributed Network Test")
    print("=" * 70)

    # Operating conditions
    y_T_initial = 0.47059  # Turbine guide vane (initial)
    y_T_step = 0.4  # Turbine guide vane after step
    t_step = 2.0  # Time of step change [s]
    N_T = 369.3346  # Turbine speed [rpm]
    N_P = -313.2579  # Pump speed [rpm]

    # Expected steady-state values (from SIMSEN DAT files)
    H_T_expected = 5.0  # [m] (from FTURB1.DAT)
    Q_T_expected = 0.1970  # [m³/s] (from FTURB1.DAT: 1.96982552146E-0001)
    Q_P1_expected = 0.1019  # [m³/s] (from PUMP1.DAT: 1.01869747658E-0001)
    Q_P2_expected = 0.0951  # [m³/s] (from PUMP2.DAT: 9.51021858327E-0002)

    # Simulation settings
    t_start = 0.0
    t_end = 8.0  # 2s steady + 6s after step
    dt_output = 0.01  # Output timestep for plotting

    # Guide vane step function
    def y_T_func(t):
        return y_T_initial if t < t_step else y_T_step

    # Create turbine and pump models
    print("\n1. Loading characteristic curves...")
    turbine, pump = create_turbine_and_pump()

    # Test turbine model at operating point
    H_turb_test = turbine.compute_H(y_T_initial, N_T, Q_T_expected)
    print(
        f"   Turbine H at (y={y_T_initial}, N={N_T}, Q={Q_T_expected:.4f}): {H_turb_test:.3f} m"
    )

    # Test pump model at operating point
    # Note: SIMSEN pump uses negative Q convention, so negate for compute_H
    H_pump1_test = pump.compute_H(1.0, N_P, -Q_P1_expected)
    H_pump2_test = pump.compute_H(1.0, N_P, -Q_P2_expected)
    print(f"   Pump1 H at (y=1, N={N_P}, Q={Q_P1_expected:.4f}): {H_pump1_test:.3f} m")
    print(f"   Pump2 H at (y=1, N={N_P}, Q={Q_P2_expected:.4f}): {H_pump2_test:.3f} m")

    # Verify characteristic curves are loaded
    print(f"   Turbine data points: {len(turbine.y_data)}")
    print(f"   Pump data points: {len(pump.y_data)}")

    # Create PF3 network
    print("\n2. Building PF3 distributed network...")
    data_dir = Path("data/pf3/passive_elements")

    try:
        network = build_pf3_full(
            interp_method="linear",
            data_dir=data_dir,
        )
        network.print_summary()
    except Exception as e:
        print(f"ERROR building network: {e}")
        import traceback

        traceback.print_exc()
        return

    # Get initial state from SIMSEN DAT files
    print("\n3. Setting initial conditions from SIMSEN DAT files...")
    y0 = network.get_initial_state()
    print(f"   Initial state size: {len(y0)}")
    print(f"   Initial state (first 10): {y0[:10]}")

    # Debug: Check initial pump flow values in state vector
    # Find pump branch indices
    branch_order = list(network._Q_indices.keys())
    for name in ["LP1_Q0", "LP2_Q0", "L11_Q0"]:
        if name in network._Q_indices:
            idx = network._Q_indices[name]
            print(f"   {name} at index {idx}: y0[{idx}] = {y0[idx]:.4f}")

    # Run simulation with step in turbine guide vane
    print("\n4. Running simulation...")
    print(f"   t = [{t_start}, {t_end}] s")
    print(f"   y_T = {y_T_initial} for t < {t_step}s, then {y_T_step}")
    print(f"   N_T = {N_T} rpm (constant)")
    print(f"   N_P = {N_P} rpm (constant)")

    t_eval = np.arange(t_start, t_end + dt_output, dt_output)

    try:
        result = network.simulate(
            t_span=(t_start, t_end),
            y0=y0,
            t_eval=t_eval,
            external_inputs_func=lambda t: {
                "y_T": y_T_func(t),
                "N_T": N_T,
                "N_P": N_P,
                "y_P1": 1.0,
                "y_P2": 1.0,
            },
            method="BDF",  # Implicit solver for stiff hydraulic system
        )
        print(f"   Simulation completed: {len(result.t)} time points")
    except Exception as e:
        print(f"ERROR in simulation: {e}")
        import traceback

        traceback.print_exc()
        return

    # Extract results
    print("\n5. Extracting results...")
    results = network.extract_results(result)

    t = results["t"]
    H_stank = results.get("H_STANK", np.zeros_like(t))

    # Get turbine path flows (from CONE which has turbine)
    Q_turb_in = results.get("Q_CONE_in", results.get("Q_L4_in", np.zeros_like(t)))
    Q_turb_out = results.get("Q_CONE_out", results.get("Q_L12_out", np.zeros_like(t)))

    # Get pump flows - use branch names directly (Q_{pipe}_Q0 for inlet)
    # Debug: print available pump-related keys
    pump_keys = [
        k for k in results.keys() if "LP" in k or "L5" in k or "L6" in k or "L11" in k
    ]
    print(f"   Pump-related keys: {pump_keys}")

    Q_P1 = results.get("Q_LP1_in", results.get("Q_LP1_Q0", np.zeros_like(t)))
    Q_P2 = results.get("Q_LP2_in", results.get("Q_LP2_Q0", np.zeros_like(t)))

    # Debug: check initial values from results
    print(f"   Q_P1[0] = {Q_P1[0]:.4f} (expected: {Q_P1_expected:.4f})")
    print(f"   Q_P2[0] = {Q_P2[0]:.4f} (expected: {Q_P2_expected:.4f})")

    print(f"   Available result keys: {list(results.keys())[:20]}...")

    # Compute turbine head from characteristic at each timestep
    # (The network doesn't directly output H_T, but we can compute it)
    H_T_computed = np.array(
        [turbine.compute_H(y_T_func(t[i]), N_T, Q_turb_in[i]) for i in range(len(t))]
    )

    # Analyze which states changed most from initial to final (causing excitation)
    print("\n6. State deviation analysis (initial vs final):")
    y_final = result.y[:, -1]
    state_changes = np.abs(y_final - y0)

    # Get state names from network
    state_names = []
    for branch_name in network._branch_order:
        state_names.append(f"Q_{branch_name}")
    for node_name in network._cap_nodes:
        state_names.append(f"H_{node_name}")

    # Find top 15 most changed states
    sorted_indices = np.argsort(state_changes)[::-1]
    print("   Top 15 states with largest |initial - final| deviation:")
    for i, idx in enumerate(sorted_indices[:15]):
        name = state_names[idx] if idx < len(state_names) else f"state_{idx}"
        print(
            f"   {i+1:2d}. {name:25s}: init={y0[idx]:+.4f}, final={y_final[idx]:+.4f}, Δ={state_changes[idx]:+.4f}"
        )

    # Print final values
    print("\n7. Final values:")
    print(f"   H_stank:  {H_stank[-1]:.4f} m")
    print(f"   Q_turb:   {Q_turb_in[-1]:.6f} m³/s (expected: {Q_T_expected:.4f})")
    print(f"   Q_P1:     {Q_P1[-1]:.6f} m³/s (expected: {Q_P1_expected:.4f})")
    print(f"   Q_P2:     {Q_P2[-1]:.6f} m³/s (expected: {Q_P2_expected:.4f})")
    print(f"   H_T (computed): {H_T_computed[-1]:.4f} m (expected: {H_T_expected})")

    # Check continuity: Q_T = Q_P1 + Q_P2
    Q_sum = Q_P1[-1] + Q_P2[-1]
    print(f"\n   Continuity check: Q_P1 + Q_P2 = {Q_sum:.6f} m³/s")
    print(f"   Error: {abs(Q_turb_in[-1] - Q_sum):.6f} m³/s")

    # Plot results
    print("\n7. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"PF3 Distributed Network - Turbine Guide Vane Step\n"
        f"y_T: {y_T_initial} → {y_T_step} at t={t_step}s, N_T={N_T} rpm, N_P={N_P} rpm"
    )

    # Flow plot
    ax = axes[0, 0]
    ax.plot(t, Q_turb_in, "b-", label="Q_turb (in)", linewidth=1.5)
    ax.plot(t, Q_P1, "r--", label="Q_P1", linewidth=1)
    ax.plot(t, Q_P2, "g--", label="Q_P2", linewidth=1)
    ax.plot(t, Q_P1 + Q_P2, "k:", label="Q_P1 + Q_P2", linewidth=1)
    ax.axhline(
        Q_T_expected,
        color="b",
        linestyle=":",
        alpha=0.5,
        label=f"Q_T initial ({Q_T_expected})",
    )
    ax.axvline(
        t_step, color="gray", linestyle="--", alpha=0.7, label=f"Step at t={t_step}s"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flow [m³/s]")
    ax.set_title("Flows")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Head plot
    ax = axes[0, 1]
    ax.plot(t, H_T_computed, "b-", label="H_T (from characteristic)", linewidth=1.5)
    ax.plot(t, H_stank, "g-", label="H_stank", linewidth=1)
    ax.axhline(
        H_T_expected,
        color="b",
        linestyle=":",
        alpha=0.5,
        label=f"H_T initial ({H_T_expected})",
    )
    ax.axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Head [m]")
    ax.set_title("Heads")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Guide vane plot
    ax = axes[1, 0]
    y_T_trace = np.array([y_T_func(ti) for ti in t])
    ax.plot(t, y_T_trace, "b-", linewidth=1.5)
    ax.axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Guide vane opening [-]")
    ax.set_title("Turbine Guide Vane (y_T)")
    ax.set_ylim([0.35, 0.5])
    ax.grid(True, alpha=0.3)

    # Head response
    ax = axes[1, 1]
    ax.plot(t, H_T_computed, "b-", linewidth=1.5, label="H_T")
    ax.axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(
        H_T_expected,
        color="b",
        linestyle=":",
        alpha=0.5,
        label=f"H_T initial ({H_T_expected})",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Turbine Head [m]")
    ax.set_title("Turbine Head Response")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "pf3_turbine_step_response.png"
    plt.savefig(output_file, dpi=150)
    print(f"   Plot saved to: {output_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
