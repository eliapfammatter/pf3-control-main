"""
FMI Master Script for PF3 - Controller Comparison

Available pump speed control strategies:
1. FPOINTS: Open-loop trajectory following (src.fpoints_controller)
2. PI Control: Gain-scheduled PI with IMC tuning (src.gain_scheduled_pi_controller)
3. MPC-IncidenceNetwork (5-state): Linear MPC with pluggable state estimation (src.mpc_incidence_network)
4. NMPC-CasADi (3-state): Nonlinear MPC with CasADi model (src.nmpc_casadi)

Uses the NEW FMU (partial) where turbine is controlled internally via FPOINTS.
Only pump speed (PUMP1-N, PUMP2-N) is controlled via FMI.

Controllers use a common interface from src.controllers.Controller, enabling
plug-and-play comparison of different control strategies.

Usage:
    python run_fmi_control.py
"""

import sys
from pathlib import Path
from tokenize import Number

from numpy.typing import ArrayLike

from src.ltv_mpc_dompc import LTVMPCDoMPC
from src.pf3_static_model_b2 import PF3StaticModelB2

# Add project root to path (allows running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import pickle
import time as time_module
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

from src.characteristics import Characteristics
from src.controllers import Controller, ControllerState
from src.fpoints_controller import FPointsController
from src.gain_scheduled_pi_controller import PIController
from src.helpers import FPoints
from src.nmpc_casadi import MPCParams, NMPCCasadi
from src.pf3_lumped_casadi import PF3LumpedCasadi
from src.state_estimators import OracleEstimator

CACHE_DIR = Path("output/cache")


def get_cache_key(controller: Controller, stop_time: float, step_size: float) -> str:
    """Generate a cache key based on controller type and parameters."""
    safe_name = (
        controller.name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )

    # For MPC controllers, include all parameters in cache key
    if isinstance(controller, (NMPCCasadi, LTVMPCDoMPC)):
        p = controller.params
        param_str = (
            f"_dt{p.dt}_Np{p.N_p}_QHT{p.Q_H_T}_Qterm{p.Q_terminal}"
            f"_Rdu{p.R_du}_umin{p.u_min}_umax{p.u_max}_Pmax{p.P_max}"
        )
        return f"{safe_name}{param_str}_t{stop_time}_dt{step_size}"

    return f"{safe_name}_t{stop_time}_dt{step_size}"


def get_cache_path(controller: Controller, stop_time: float, step_size: float) -> Path:
    """Get cache file path for a controller."""
    cache_key = get_cache_key(controller, stop_time, step_size)
    return CACHE_DIR / f"{cache_key}.pkl"


def load_cached_results(
    controller: Controller, stop_time: float, step_size: float
) -> Optional[Dict]:
    """Load cached simulation results if available."""
    cache_path = get_cache_path(controller, stop_time, step_size)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                print(f"[CACHE] Loading: {cache_path}")
                return pickle.load(f)
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}")
    return None


def save_cached_results(
    results: Dict, controller: Controller, stop_time: float, step_size: float
) -> None:
    """Save simulation results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(controller, stop_time, step_size)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[CACHE] Saved results to {cache_path}")
    except Exception as e:
        print(f"[CACHE] Failed to save cache: {e}")


def write_fpoints_dat(
    filepath: Path,
    times: List[float],
    values: List[float],
    x_unit: str = "s",
    y_unit: str = "rpm",
) -> None:
    """Write applied control sequence to SIMSEN FPOINTS .DAT file.

    Parameters
    ----------
    filepath : Path
        Output file path
    times : list of float
        Time points [s]
    values : list of float
        Control values at each time point
    x_unit : str
        Unit for time axis (default: 's')
    y_unit : str
        Unit for values (default: 'rpm')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("- DATA :\n")
        for i, (t, v) in enumerate(zip(times, values), start=1):
            f.write(f"x{i} [{x_unit}] : {t:.6f} y{i} [{y_unit}] : {v:.6f}\n")
    print(f"[FPOINTS] Wrote {len(times)} points to {filepath}")


def run_simulation(
    fmu_filename: Path,
    controller: Controller,
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    start_time: float = 0.0,
    stop_time: float = 60.0,
    step_size: float = 0.1,
    N_P_init: float = -313.2579,
) -> Dict[str, Any]:
    """Run FMI simulation with a controller.

    Parameters
    ----------
    fmu_filename : Path
        Path to the FMU file
    controller : Controller
        Pump speed controller (implements Controller interface)
    fpoints_y_T : FPoints
        Turbine guide vane trajectory (for state feedback)
    fpoints_N_T : FPoints
        Turbine speed trajectory (for state feedback)
    start_time, stop_time : float
        Simulation time range [s]
    step_size : float
        Communication step size [s]
    N_P_init : float
        Initial pump speed [rpm]

    Returns
    -------
    dict
        Simulation results with timestamps, inputs, outputs
    """
    model_description = read_model_description(fmu_filename, validate=False)

    # Collect value references
    in_vars = {}
    out_vars = {}
    for var in model_description.modelVariables:
        if var.causality == "input":
            in_vars[var.name] = var.valueReference
        elif var.causality == "output":
            out_vars[var.name] = var.valueReference

    print(f"\nFMU: {model_description.modelIdentifier}")
    print(f"Controller: {controller.name}")
    print(f"Inputs: {list(in_vars.keys())}")
    print(f"Outputs: {list(out_vars.keys())}")

    # Reset controller
    controller.reset()

    unzipdir = extract(fmu_filename)
    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.modelIdentifier,
        instanceName="simsen",
    )

    # Initialize FMU
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()

    # Set initial values
    input_names = ["PUMP1-N", "PUMP2-N", "TURB-N", "TURB-y"]
    fmu.setReal(
        [in_vars[k] for k in input_names],
        [N_P_init, N_P_init, fpoints_N_T(0), fpoints_y_T(0)],
    )

    fmu.exitInitializationMode()

    # Simulation loop
    timestamps: List[float] = []
    inputs: Dict[str, List[float]] = {n: [] for n in in_vars}
    outputs: Dict[str, List[float]] = {n: [] for n in out_vars}

    time = start_time
    N_P_current = N_P_init

    predicted_trajectories: dict[float, tuple[ArrayLike, ArrayLike]] = {}
    predicted_power_P1: dict[float, tuple[ArrayLike, ArrayLike]] = {}
    predicted_power_P2: dict[float, tuple[ArrayLike, ArrayLike]] = {}
    diagnostics_log: List[dict] = []

    while time < stop_time:
        # Read state feedback
        # Note: FPUMP1-H/FPUMP2-H for pumps, FTURB1-N/y are inputs not outputs
        H_T = fmu.getReal([out_vars["FTURB1-H"]])[0]
        H_P1 = fmu.getReal([out_vars["FPUMP1-H"]])[0]
        H_P2 = fmu.getReal([out_vars["FPUMP2-H"]])[0]
        Q_T = fmu.getReal([out_vars["FTURB1-Q"]])[0]
        Q_P1 = fmu.getReal([out_vars["FPUMP1-Q"]])[0]
        Q_P2 = fmu.getReal([out_vars["FPUMP2-Q"]])[0]
        H_tank = fmu.getReal([out_vars["STANK-H"]])[0]

        # N_T and y_T are controlled via FPOINTS (not FMU outputs), read from trajectory
        N_T = fpoints_N_T(time)
        y_T = fpoints_y_T(time)

        state = ControllerState(
            H_T=H_T,
            H_P1=H_P1,
            H_P2=H_P2,
            N_T=N_T,
            y_T=y_T,
            N_P=N_P_current,
            Q_T=Q_T,
            Q_P1=Q_P1,
            Q_P2=Q_P2,
            H_tank=H_tank,
        )

        # Compute pump speed from controller
        N_P_current = controller.compute_pump_speed(time, state)
        # exit()

        if (traj := controller.predicted_trajectory) is not None:
            times = [time + step_size * i for i in range(len(traj))]
            predicted_trajectories[time] = (times, traj)

        # Also store nonlinear predictions if available
        if hasattr(controller, "predicted_trajectory_nonlinear"):
            traj_nl = controller.predicted_trajectory_nonlinear
            if traj_nl is not None:
                times_nl = [time + step_size * i for i in range(len(traj_nl))]
                predicted_trajectories[f"{time}_nonlinear"] = (times_nl, traj_nl)

        # Store power predictions if available (from NMPC)
        if hasattr(controller, "predicted_power_P1"):
            power_P1 = controller.predicted_power_P1
            if power_P1 is not None:
                times_p1 = [time + step_size * i for i in range(len(power_P1))]
                predicted_power_P1[time] = (
                    times_p1,
                    power_P1 / 1000.0,
                )  # Convert W to kW

        if hasattr(controller, "predicted_power_P2"):
            power_P2 = controller.predicted_power_P2
            if power_P2 is not None:
                times_p2 = [time + step_size * i for i in range(len(power_P2))]
                predicted_power_P2[time] = (
                    times_p2,
                    power_P2 / 1000.0,
                )  # Convert W to kW

        # Set inputs
        fmu.setReal(
            [in_vars[k] for k in input_names],
            [N_P_current, N_P_current, N_T, y_T],
        )

        # Perform one step
        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

        # Advance time
        time += step_size
        timestamps.append(time)

        # Read all inputs and outputs
        input_values = fmu.getReal([in_vars[n] for n in in_vars])
        for i, name in enumerate(in_vars):
            inputs[name].append(input_values[i])

        output_values = fmu.getReal([out_vars[n] for n in out_vars])
        for i, name in enumerate(out_vars):
            outputs[name].append(output_values[i])

        # Print status every second
        if int(time * 10 + 0.5) % 10 == 0:
            status_line = (
                f"[{controller.name}] t={time:6.1f}s | "
                f"N_P={N_P_current:7.1f} rpm | "
                f"N_T={inputs['TURB-N'][-1]:.1f} rpm | "
                f"H_T={outputs['FTURB1-H'][-1]:.3f} m"
            )
            print(status_line)

    fmu.terminate()
    fmu.freeInstance()

    # Save diagnostics to file
    if diagnostics_log:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        diag_file = (
            output_dir / f"mpc_diagnostics_{controller.name.replace(' ', '_')}.txt"
        )
        with open(diag_file, "w") as f:
            f.write(f"MPC Diagnostics for {controller.name}\n")
            f.write("=" * 80 + "\n\n")
            # Write header
            f.write(
                f"{'time':>6} | {'H_T_act':>8} | {'H_T_op':>8} | {'pred_next':>9} | "
            )
            f.write(
                f"{'x_est[0]':>9} | {'x_est[1]':>9} | {'x_op[0]':>9} | {'x_op[1]':>9} | "
            )
            f.write(f"{'dx[0]':>9} | {'dx[1]':>9}\n")
            f.write("-" * 120 + "\n")
            for d in diagnostics_log:
                x_est = d.get("x_est", [None, None])
                x_op = d.get("x_op", [None, None])
                dx0 = (x_est[0] - x_op[0]) if x_est[0] and x_op[0] else None
                dx1 = (x_est[1] - x_op[1]) if x_est[1] and x_op[1] else None
                f.write(
                    f"{d['time']:6.1f} | {d.get('H_T_actual', 0):8.4f} | {d.get('H_T_op', 0):8.4f} | "
                )
                f.write(f"{d.get('pred_H_T_next', 0):9.4f} | ")
                f.write(f"{x_est[0] or 0:9.5f} | {x_est[1] or 0:9.5f} | ")
                f.write(f"{x_op[0] or 0:9.5f} | {x_op[1] or 0:9.5f} | ")
                f.write(f"{dx0 or 0:+9.5f} | {dx1 or 0:+9.5f}\n")
            # Write A, B, C matrices from last step
            last_diag = diagnostics_log[-1]
            f.write("\n" + "=" * 80 + "\n")
            f.write("Last A, B, C matrices:\n")
            f.write(f"A = {last_diag.get('A')}\n")
            f.write(f"B = {last_diag.get('B')}\n")
            f.write(f"C = {last_diag.get('C')}\n")
        print(f"\nDiagnostics saved to: {diag_file}")

    return {
        "timestamps": timestamps,
        "inputs": inputs,
        "outputs": outputs,
        "controller_name": controller.name,
        "predicted_trajectories": predicted_trajectories,
        "predicted_power_P1": predicted_power_P1,
        "predicted_power_P2": predicted_power_P2,
        "diagnostics_log": diagnostics_log,
    }


def plot_comparison_matplotlib(
    results_list: List[Dict[str, Any]],
    H_ref: FPoints,
    stop_time: float,
    step_size: float,
    experiment_name: str = "",
    save_path: Optional[str] = None,
    N_P_min: float = -400.0,
    N_P_max: float = -200.0,
    P_max: float = 300000.0,
):
    """Create comparison plots using matplotlib."""
    import matplotlib.pyplot as plt

    if len(results_list) < 1:
        print("Need at least 1 result")
        return

    colors = ["blue", "red", "green", "orange", "purple"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"PF3 Control: {experiment_name}\nTime-varying H_ref, dt={step_size}s")

    # Row 0: Turbine N_T and y_T (left), Pump N_P (right)
    ax_turb_N = axes[0, 0]
    ax_turb_y = ax_turb_N.twinx()  # Create secondary y-axis for y_T
    ax_N_P = axes[0, 1]

    # Row 1: Turbine Head H_T (left), Pump Power Predictions (right)
    ax_H_T = axes[1, 0]
    ax_P_pred = axes[1, 1]

    # Row 2: Power plots (keep as is)
    ax_P_T = axes[2, 0]
    ax_P_P = axes[2, 1]

    for i, results in enumerate(results_list):
        t = results["timestamps"]
        H_T = results["outputs"]["FTURB1-H"]
        N_P = results["inputs"]["PUMP1-N"]
        name = results["controller_name"]
        predicted_trajectories = results["predicted_trajectories"]
        color = colors[i % len(colors)]

        # Get turbine inputs (N_T and y_T)
        N_T = results["inputs"]["TURB-N"]
        y_T = results["inputs"]["TURB-y"]

        # Plot turbine N_T on left y-axis
        ax_turb_N.plot(t, N_T, color=color, lw=1.5, label=f"N_T ({name})")

        # Plot turbine y_T on right y-axis (dashed line)
        ax_turb_y.plot(t, y_T, color=color, lw=1.5, ls="--", label=f"y_T ({name})")

        # Plot predictions every 0.5 seconds
        # Separate linearized (numeric keys) from nonlinear (string keys with "_nonlinear")
        plot_interval = 0.5  # seconds
        linear_preds = {
            k: v
            for k, v in predicted_trajectories.items()
            if isinstance(k, (int, float))
        }
        nonlinear_preds = {
            k: v
            for k, v in predicted_trajectories.items()
            if isinstance(k, str) and "_nonlinear" in k
        }

        for time_k, v in linear_preds.items():
            # Plot if time is approximately a multiple of plot_interval
            if (
                abs(time_k % plot_interval) < 0.01
                or abs(time_k % plot_interval - plot_interval) < 0.01
            ):
                ax_H_T.plot(v[0], v[1], color=color, ls=":", lw=1, alpha=0.5)
                # Add dot at state estimation point (first point of prediction)
                ax_H_T.plot(v[0][0], v[1][0], "o", color=color, ms=4, alpha=0.7)

        # Plot nonlinear ROM predictions in green (every 0.5s)
        for time_key, v in nonlinear_preds.items():
            # Extract time from key like "2.5_nonlinear"
            time_k = float(time_key.replace("_nonlinear", ""))
            if (
                abs(time_k % plot_interval) < 0.01
                or abs(time_k % plot_interval - plot_interval) < 0.01
            ):
                ax_H_T.plot(v[0], v[1], color="green", ls="--", lw=1, alpha=0.7)
                # Add dot at state estimation point (first point of prediction)
                ax_H_T.plot(v[0][0], v[1][0], "o", color="green", ms=4, alpha=0.7)

        ax_H_T.plot(t, H_T, color=color, lw=1.5, label=f"H_T ({name})")
        ax_N_P.plot(t, N_P, color=color, lw=1.5, label=f"N_P ({name})")

        # Plot power from FMU outputs (mechanical power Pm - in W, convert to kW)
        # Turbine power
        if "FTURB1-Pm" in results["outputs"]:
            P_T = np.array(results["outputs"]["FTURB1-Pm"]) / 1000.0  # Convert W to kW
            ax_P_T.plot(t, P_T, color=color, lw=1.5, label=f"P_T ({name})")

        # Pump power
        if "FPUMP1-Pm" in results["outputs"]:
            P_P1 = np.array(results["outputs"]["FPUMP1-Pm"]) / 1000.0  # Convert W to kW
            ax_P_P.plot(t, P_P1, color=color, lw=1.5, label=f"P_P1 ({name})")

        if "FPUMP2-Pm" in results["outputs"]:
            P_P2 = np.array(results["outputs"]["FPUMP2-Pm"]) / 1000.0  # Convert W to kW
            ax_P_P.plot(t, P_P2, color=color, lw=1.5, label=f"P_P2 ({name})")

        # Plot power predictions (NMPC internal predictions)
        predicted_power_P1 = results.get("predicted_power_P1", {})
        predicted_power_P2 = results.get("predicted_power_P2", {})

        # Plot P_P1 predictions every 0.5 seconds
        for time_k, v in predicted_power_P1.items():
            if isinstance(time_k, (int, float)):
                if (
                    abs(time_k % plot_interval) < 0.01
                    or abs(time_k % plot_interval - plot_interval) < 0.01
                ):
                    ax_P_pred.plot(v[0], v[1], color=color, ls=":", lw=1, alpha=0.5)
                    ax_P_pred.plot(v[0][0], v[1][0], "o", color=color, ms=3, alpha=0.7)

        # Plot P_P2 predictions every 0.5 seconds (dashed)
        for time_k, v in predicted_power_P2.items():
            if isinstance(time_k, (int, float)):
                if (
                    abs(time_k % plot_interval) < 0.01
                    or abs(time_k % plot_interval - plot_interval) < 0.01
                ):
                    ax_P_pred.plot(v[0], v[1], color=color, ls="--", lw=1, alpha=0.5)
                    ax_P_pred.plot(v[0][0], v[1][0], "s", color=color, ms=3, alpha=0.7)

        # Plot actual pump power on prediction subplot for comparison
        if "FPUMP1-Pm" in results["outputs"]:
            P_P1 = np.array(results["outputs"]["FPUMP1-Pm"]) / 1000.0  # Convert W to kW
            ax_P_pred.plot(t, P_P1, color=color, lw=1.5, label=f"P_P1 ({name})")

        if "FPUMP2-Pm" in results["outputs"]:
            P_P2 = np.array(results["outputs"]["FPUMP2-Pm"]) / 1000.0  # Convert W to kW
            ax_P_pred.plot(
                t, P_P2, color=color, lw=1.5, ls="--", label=f"P_P2 ({name})"
            )

    # Plot time-varying reference (evaluate at time points from first result)
    if results_list:
        t_ref = results_list[0]["timestamps"]
        H_ref_traj = np.array([H_ref(t_i) for t_i in t_ref])
        ax_H_T.plot(t_ref, H_ref_traj, color="black", ls="--", lw=2, label="H_ref")

    # Row 0: Turbine N_T and y_T
    ax_turb_N.set_xlabel("Time [s]")
    ax_turb_N.set_ylabel("N_T [rpm]", color="black")
    ax_turb_y.set_ylabel("y_T [-]", color="black")
    ax_turb_N.set_title("Turbine Speed N_T and Guide Vane y_T")
    ax_turb_N.tick_params(axis="y", labelcolor="black")
    ax_turb_y.tick_params(axis="y", labelcolor="black")
    ax_turb_N.legend(loc="upper left", fontsize=8)
    ax_turb_y.legend(loc="upper right", fontsize=8)
    ax_turb_N.grid(True, alpha=0.3)

    ax_N_P.set_xlabel("Time [s]")
    ax_N_P.set_ylabel("N_P [rpm]")
    ax_N_P.set_title("Pump Speed N_P")
    ax_N_P.legend(loc="best", fontsize=8)
    ax_N_P.grid(True, alpha=0.3)

    # Row 1: Turbine Head H_T
    ax_H_T.set_xlabel("Time [s]")
    ax_H_T.set_ylabel("H_T [m]")
    ax_H_T.set_title("Turbine Head H_T")
    ax_H_T.legend(loc="best", fontsize=8)
    ax_H_T.grid(True, alpha=0.3)

    # Row 1: Pump Power Predictions (NMPC)
    ax_P_pred.set_xlabel("Time [s]")
    ax_P_pred.set_ylabel("P_P [kW]")
    ax_P_pred.set_title("Pump Power (with NMPC Predictions)")
    ax_P_pred.legend(loc="best", fontsize=7)
    ax_P_pred.grid(True, alpha=0.3)

    # Row 2: Power plots
    ax_P_T.set_xlabel("Time [s]")
    ax_P_T.set_ylabel("P_T [kW]")
    ax_P_T.set_title("Turbine Power")
    ax_P_T.legend(loc="best", fontsize=8)
    ax_P_T.grid(True, alpha=0.3)

    ax_P_P.set_xlabel("Time [s]")
    ax_P_P.set_ylabel("P_P [kW]")
    ax_P_P.set_title("Pump Power")
    ax_P_P.legend(loc="best", fontsize=7)
    ax_P_P.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("plots/fmi-pf3")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir
            / f"{experiment_name}_t{stop_time}_dt{step_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    plt.savefig(str(output_file), dpi=300)
    print(f"\n[{experiment_name}] Plot saved to: {output_file}")
    plt.show()
    return output_file


def print_performance_summary(
    results_list: List[Dict[str, Any]],
    H_ref: FPoints,
):
    """Print performance metrics for all controllers."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\nReference: Time-varying H_ref (FPoints)")

    # Header
    metrics = [
        "Final H_T [m]",
        "Final error [mm]",
        "SS error (last 5s) [mm]",
        "Max |error| [mm]",
        "RMS error [mm]",
        "N_P range [rpm]",
    ]

    print(f"\n{'Metric':<30}", end="")
    for results in results_list:
        print(f"{results['controller_name']:>20}", end="")
    print()
    print("-" * (30 + 20 * len(results_list)))

    for metric in metrics:
        print(f"{metric:<30}", end="")
        for results in results_list:
            t = results["timestamps"]
            H_T = np.array(results["outputs"]["FTURB1-H"])
            N_P = np.array(results["inputs"]["PUMP1-N"])
            H_ref_traj = np.array([H_ref(t_i) for t_i in t])
            error = H_ref_traj - H_T

            if metric == "Final H_T [m]":
                print(f"{H_T[-1]:>20.4f}", end="")
            elif metric == "Final error [mm]":
                print(f"{error[-1]*1000:>20.2f}", end="")
            elif metric == "SS error (last 5s) [mm]":
                print(f"{np.mean(error[-50:])*1000:>20.2f}", end="")
            elif metric == "Max |error| [mm]":
                print(f"{np.max(np.abs(error))*1000:>20.2f}", end="")
            elif metric == "RMS error [mm]":
                print(f"{np.sqrt(np.mean(error**2))*1000:>20.2f}", end="")
            elif metric == "N_P range [rpm]":
                print(f"[{min(N_P):.1f}, {max(N_P):.1f}]".rjust(20), end="")
        print()

    print("=" * (30 + 20 * len(results_list)))


def create_pf3_system() -> PF3StaticModelB2:
    """Create PF3 system model with correct SIMSEN parameters."""
    turbine = Characteristics(
        d_ref=0.3477,
        h_n=5.0,
        q_n=0.1959,
        t_n=224.057,
        n_n=369.3346,
        char_file=Path("data/pf3/missing_files/STORE_4_quadrant_characteristic.txt"),
    )
    pump = Characteristics(
        d_ref=0.535,
        h_n=50.0,
        q_n=1.30,
        t_n=1280.3,
        n_n=790.0,
        char_file=Path("data/pf3/missing_files/PF3_FP_D_535_ext.txt"),
    )
    return PF3StaticModelB2(
        turbine=turbine,
        hydraulic_elements_dir=Path("data/pf3/passive_elements"),
        pump=pump,
        eps_rel=0.02,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PF3 Controller Comparison")
    parser.add_argument("--stop-time", type=float, default=60.0)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading from cache (still saves results)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Custom path to save the figure (default: auto-generated in plots/fmi-pf3/)",
    )
    args = parser.parse_args()

    data_dir = Path("data/pf3")

    # Simulation settings
    experiment_name = "fast_transition"
    fmu_filename = data_dir / "PF3_FMI.fmu"
    start_time = 0.0
    stop_time = args.stop_time
    step_size = args.step_size

    # Controller settings
    # Time-varying reference: step change from 5.0m to 6.0m at t=10s
    fpoints_H_ref = FPoints.constant(5.0)
    fpoints_H_ref = FPoints(data_dir / "H_REF_STEP.DAT")
    # For constant reference, use: fpoints_H_ref = FPoints.constant(H_REF)
    N_P_INIT = -313.2579
    N_T_INIT = 369.3346
    N_y_INIT = 0.4706
    N_P_MIN = -400.0
    N_P_MAX = -200.0

    # CasADi lumped model (for nonlinear MPC)
    casadi_model = PF3LumpedCasadi(data_dir / "passive_elements")

    # Load turbine FPOINTS (needed for state feedback since FMU doesn't expose them)
    fpoints_y_T = FPoints(data_dir / "REGY")
    fpoints_N_T = FPoints(data_dir / "REGN")

    def zero_fun(x):
        return 0.0

    fpoints_y_T_0 = zero_fun
    fpoints_N_T_0 = zero_fun

    # =========================================================================
    # Define controllers to compare
    # =========================================================================

    # State estimator (pluggable - can swap different estimation strategies)
    # Option 1: TurbineInversionEstimator - estimates Q from H_T inversion
    # state_estimator = TurbineInversionEstimator(Q_bounds=(0.05, 0.4))

    # Option 2: OracleEstimator - uses actual FMU values (for debugging)
    state_estimator = OracleEstimator()

    # PF3 system model for PI gain scheduling
    pf3_system = create_pf3_system()

    controllers = [
        # # Open-loop: trajectory following (baseline)
        # FPointsController(FPoints(data_dir / "REGP")),
        # FPointsController(FPoints(data_dir / "REGP_NMPC")),
        # # Gain-scheduled PI control
        # PIController(
        #     H_ref=H_REF,
        #     pf3_system=pf3_system,
        #     dt=step_size,
        #     u_nominal=N_P_INIT,
        # ),
        # # Fixed gain PI controller.
        # PIController(
        #     H_ref=H_REF,
        #     pf3_system=pf3_system,
        #     dt=step_size,
        #     u_nominal=N_P_INIT,
        #     gain_scheduling=False,
        # ),
        NMPCCasadi(
            H_ref=fpoints_H_ref,
            model=casadi_model,
            fpoints_y_T=fpoints_y_T,
            fpoints_N_T=fpoints_N_T,
            params=MPCParams(
                dt=step_size,
                N_p=50,
                Q_H_T=10.0,
                Q_terminal=100.0,
                R_du=0.01,
                u_min=N_P_MIN,
                u_max=N_P_MAX,
                P_max=7500.0,
            ),
            u_nominal=N_P_INIT,
        ),
    ]

    # =========================================================================
    # Run simulations for each controller
    # =========================================================================
    results_list = []
    USE_CACHE = not args.no_cache

    for i, controller in enumerate(controllers):
        print("\n" + "=" * 70)
        print(f"SIMULATION {i+1}: {controller.name}")
        print("=" * 70)

        # Try to load from cache
        cached = None
        if USE_CACHE:
            cached = load_cached_results(controller, stop_time, step_size)

        if cached is not None:
            results = cached
            print(f"[{controller.name}] Using cached results!")
        else:
            results = run_simulation(
                fmu_filename=fmu_filename,
                controller=controller,
                fpoints_y_T=fpoints_y_T,
                fpoints_N_T=fpoints_N_T,
                start_time=start_time,
                stop_time=stop_time,
                step_size=step_size,
                N_P_init=N_P_INIT,
            )
            # Cache results
            save_cached_results(results, controller, stop_time, step_size)

            print(f"\n[{controller.name}] Simulation complete!")

            # Wait for SIMSEN to restart (except after last simulation)
            if i < len(controllers) - 1:
                print("\n" + "-" * 70)
                print("Waiting 5 seconds for SIMSEN server to restart...")
                print("-" * 70)
                time_module.sleep(5)

        results_list.append(results)

    # =========================================================================
    # Performance Comparison
    # =========================================================================
    print_performance_summary(results_list, fpoints_H_ref)

    # =========================================================================
    # Create Comparison Plots
    # =========================================================================
    plot_comparison_matplotlib(
        results_list,
        fpoints_H_ref,
        stop_time,
        step_size,
        experiment_name,
        save_path=args.save,
        N_P_min=N_P_MIN,
        N_P_max=N_P_MAX,
        P_max=300000.0,  # 300 kW (from MPCParams)
    )

    # =========================================================================
    # Export Control Sequences to FPOINTS .DAT Files
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPORTING CONTROL SEQUENCES TO .DAT FILES")
    print("=" * 80)

    output_dir = Path("output/fpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    for results in results_list:
        controller_name = results["controller_name"]
        # Create safe filename from controller name
        safe_name = (
            controller_name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

        # Export pump speeds (PUMP1-N, PUMP2-N)
        write_fpoints_dat(
            filepath=output_dir / f"REGP_{safe_name}.DAT",
            times=results["timestamps"],
            values=results["inputs"]["PUMP1-N"],
            x_unit="s",
            y_unit="rpm",
        )

        # Also export turbine inputs for reference (if you want to replay them)
        write_fpoints_dat(
            filepath=output_dir / f"REGN_{safe_name}.DAT",
            times=results["timestamps"],
            values=results["inputs"]["TURB-N"],
            x_unit="s",
            y_unit="rpm",
        )

        write_fpoints_dat(
            filepath=output_dir / f"REGY_{safe_name}.DAT",
            times=results["timestamps"],
            values=results["inputs"]["TURB-y"],
            x_unit="s",
            y_unit="-",
        )

    print("\nDone!")
