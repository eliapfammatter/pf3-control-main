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

from numpy.typing import ArrayLike

# Add project root to path (allows running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import pickle
import subprocess
import time as time_module
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

from src.controllers import Controller, ControllerState
from src.gvo_kinematics import GVOKinematics, compute_motor_speed
from src.gvo_trajectory_optimizer import optimize_y_T_trajectory
from src.helpers import FPoints, plot_efficiency_hillchart, plot_trajectory_on_hillchart
from src.hydraulic_elements.pump_turbine import PumpTurbine
from src.nmpc_casadi import MPCParams, NMPCCasadi
from src.pf3_lumped_casadi import PF3LumpedCasadi
from src.pl_controller import PLController, PLControllerParams, PLNULLController


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


def write_reproducibility_info(output_path: Path, argv: List[str]) -> None:
    """Write reproducibility info: git hash, dirty status, and command."""
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        git_hash = "unknown"

    try:
        git_status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        is_dirty = len(git_status) > 0
    except subprocess.CalledProcessError:
        is_dirty = None

    with open(output_path, "w") as f:
        f.write("# Reproducibility Info\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(f"Git hash: {git_hash}\n")
        f.write(f"Git dirty: {is_dirty}\n\n")
        f.write(f"Command:\n```\npython {' '.join(argv)}\n```\n")
    print(f"[REPRO] Wrote to {output_path}")


def run_simulation(
    fmu_filename: Path,
    controller: Controller,
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    pl_controller: PLController,
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
    print(f"Simulation Time: {stop_time}")

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
    H_ref_values: List[float] = []

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
        Pm_T = fmu.getReal([out_vars["FTURB1-Pm"]])[0]

        (fpoints_N_T, fpoints_y_T, fpoints_H_ref) = pl_controller(time, Pm_T)
        controller.update_fpoints(fpoints_H_ref, fpoints_y_T, fpoints_N_T)

        # N_T, y_T, H_ref are controlled via FPOINTS (not FMU outputs), read from trajectory
        N_T = fpoints_N_T(time)
        y_T = fpoints_y_T(time)
        H_ref = fpoints_H_ref(time)
        H_ref_values.append(H_ref)

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

    return {
        "timestamps": timestamps,
        "inputs": inputs,
        "outputs": outputs,
        "controller_name": controller.name,
        "predicted_trajectories": predicted_trajectories,
        "predicted_power_P1": predicted_power_P1,
        "predicted_power_P2": predicted_power_P2,
        "diagnostics_log": diagnostics_log,
        "H_ref": H_ref_values,
    }


def plot_comparison_matplotlib(
    results_list: List[Dict[str, Any]],
    stop_time: float,
    step_size: float,
    output_dir: Path,
    experiment_name: str = "",
    N_P_min: float = -400.0,
    N_P_max: float = -200.0,
    P_max: float = 300000.0,
    Pm_BEP: float = 0.0,
    gvo_kin: Optional[GVOKinematics] = None,
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

    # Row 2: Turbine Power (left), Motor Speed (right)
    ax_P_T = axes[2, 0]
    ax_motor = axes[2, 1]

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

        # Motor speed (from y_T via GVO kinematics)
        if gvo_kin is not None:
            motor_speed = compute_motor_speed(np.array(t), np.array(y_T), gvo_kin)
            # motor_speed has N-1 values, pad with 0 at start
            motor_speed = np.insert(motor_speed, 0, 0)
            ax_motor.plot(t, motor_speed, color=color, lw=1.5, label=f"Motor ({name})")

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

    # Plot time-varying reference from results (includes PLController adjustments)
    if results_list:
        t_ref = results_list[0]["timestamps"]
        H_ref_traj = results_list[0]["H_ref"]
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
    ax_H_T.ticklabel_format(useOffset=False, axis="y")

    # Row 1: Pump Power Predictions (NMPC)
    ax_P_pred.set_xlabel("Time [s]")
    ax_P_pred.set_ylabel("P_P [kW]")
    ax_P_pred.set_title("Pump Power (with NMPC Predictions)")
    ax_P_pred.legend(loc="best", fontsize=7)
    ax_P_pred.grid(True, alpha=0.3)

    # Row 2: Power plots
    # Add reference lines for BEP and part load power
    if Pm_BEP > 0:
        Pm_BEP_kW = Pm_BEP / 1000.0
        ax_P_T.axhline(
            y=Pm_BEP_kW,
            color="black",
            ls="--",
            lw=1.5,
            label=f"P_BEP ({Pm_BEP_kW:.1f} kW)",
        )
        ax_P_T.axhline(
            y=Pm_BEP_kW * 0.8,
            color="gray",
            ls="--",
            lw=1.5,
            label=f"P_PL{80:.0f}% ({Pm_BEP_kW*0.8:.1f} kW)",
        )
        ax_P_T.axhline(
            y=Pm_BEP_kW * 0.6,
            color="gray",
            ls="--",
            lw=1.5,
            label=f"P_PL{60:.0f}% ({Pm_BEP_kW*0.6:.1f} kW)",
        )

    ax_P_T.set_xlabel("Time [s]")
    ax_P_T.set_ylabel("P_T [kW]")
    ax_P_T.set_title("Turbine Power")
    ax_P_T.legend(loc="best", fontsize=8)
    ax_P_T.grid(True, alpha=0.3)

    ax_motor.set_xlabel("Time [s]")
    ax_motor.set_ylabel("Motor Speed [rpm]")
    ax_motor.set_title("GVO Servomotor Speed")
    ax_motor.legend(loc="best", fontsize=7)
    ax_motor.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"timeseries_{experiment_name}.png"
    plt.savefig(str(output_file), dpi=300)
    print(f"\n[{experiment_name}] Plot saved to: {output_file}")
    # plt.show()
    return output_file


class OperatingPoint:
    def __init__(self, char, N, Q, H):
        self.N = N
        self.Q = Q
        self.H = H
        self.char = char

    @property
    def y(self):
        return self.char.y_from_NQH(self.N, self.Q, self.H)


def generate_op_steps(char, swirl, start_point: OperatingPoint):
    A = start_point

    # B: Same N as A, reduce Q to get swirl, same H
    B_Q11 = char.Q11_from_N11S(char.N11(A.N, A.H), swirl)
    B_Q = char.Q(B_Q11, A.H)
    B = OperatingPoint(char, A.N, B_Q, A.H)

    # C: Same y as B, same H as B, but S=0 (zero swirl)
    C_N = char.N_from_yHS(B.y, B.H, S_target=0.0)
    C_Q, _ = char.QT_from_yNH(B.y, C_N, B.H)
    C = OperatingPoint(char, C_N, C_Q, B.H)

    # D: Same N as C, reduce Q to get swirl, same H
    D_Q11 = char.Q11_from_N11S(char.N11(C.N, C.H), swirl)
    D_Q = char.Q(D_Q11, C.H)
    D = OperatingPoint(char, C.N, D_Q, C.H)

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
    }


def generate_op_sweep(char, swirl, start_point: OperatingPoint):
    """Generate operating points A, B, D for sweep trajectory.

    Parameters
    ----------
    char : Characteristic
        Turbine characteristic curves
    swirl : float
        Target swirl number for part-load points
    start_point : OperatingPoint
        BEP operating point (point A)

    Returns
    -------
    dict
        Operating points:
        - A: BEP (same as start_point)
        - B: Same N11 as A, Q11 such that swirl matches target
        - D: N11 = 0.5 * A.N11, Q11 such that swirl matches target
    """
    A = start_point

    # B: Same N11 as A (same N, same H), Q11 from target swirl
    A_N11 = char.N11(A.N, A.H)
    B_Q11 = char.Q11_from_N11S(A_N11, swirl)
    B_Q = char.Q(B_Q11, A.H)
    B = OperatingPoint(char, A.N, B_Q, A.H)

    # D: N11 = 0.5 * A.N11, Q11 from target swirl
    # Since N11 = N / sqrt(H) and H is constant, N_D = 0.5 * N_A
    D_N11 = 0.7 * A_N11
    D_N = 0.7 * A.N
    D_Q11 = char.Q11_from_N11S(D_N11, swirl)
    D_Q = char.Q(D_Q11, A.H)
    D = OperatingPoint(char, D_N, D_Q, A.H)

    return {
        "A": A,
        "B": B,
        "D": D,
    }


def generate_fpoints_sweep(sequence: List[OperatingPoint], zoh):
    """Generate sweep trajectory: stepwise B→D→B, wait, then linear B→D→B.

    Timeline (in units of zoh):
    - 0 to 0.5: at A
    - 0.5: step to B
    - 1.5 to 2.5: stepwise B→D (10 steps)
    - 2.5 to 3.5: stepwise D→B (10 steps)
    - 3.5 to 4.0: wait at B
    - 4.0 to 5.0: linear B→D
    - 5.0 to 6.0: linear D→B
    - 6.5: step to A
    Total: 7*zoh
    """
    A, B, D = sequence[0], sequence[1], sequence[2]

    # Build arrays for from_arrays (linear interpolation between points)
    # Use duplicate times for step changes
    times = []
    h_ref = []
    y_t = []
    n_t = []

    def add_point(t, op):
        times.append(t)
        h_ref.append(op.H)
        y_t.append(op.y)
        n_t.append(op.N)

    def add_step(t, op_before, op_after):
        """Add a step change (two points at same time)."""
        add_point(t, op_before)
        add_point(t, op_after)

    # Start at A
    add_point(0.0, A)

    # Step to B at 0.5*zoh
    add_step(0.5 * zoh, A, B)

    # Stepwise B→D (10 steps from 1.5*zoh to 2.5*zoh)
    for i in range(10):
        frac = i / 10.0
        t = 1.5 * zoh + zoh * frac
        op = OperatingPoint(
            B.char,
            B.N + (D.N - B.N) * frac,
            B.Q + (D.Q - B.Q) * frac,
            B.H + (D.H - B.H) * frac,
        )
        if i == 0:
            add_step(t, B, op)
        else:
            prev_frac = (i - 1) / 10.0
            prev_op = OperatingPoint(
                B.char,
                B.N + (D.N - B.N) * prev_frac,
                B.Q + (D.Q - B.Q) * prev_frac,
                B.H + (D.H - B.H) * prev_frac,
            )
            add_step(t, prev_op, op)

    # Step to D at 2.5*zoh
    add_step(
        2.5 * zoh,
        OperatingPoint(
            B.char,
            B.N + (D.N - B.N) * 0.9,
            B.Q + (D.Q - B.Q) * 0.9,
            B.H + (D.H - B.H) * 0.9,
        ),
        D,
    )

    # Stepwise D→B (10 steps from 2.5*zoh to 3.5*zoh)
    for i in range(1, 10):
        frac = i / 10.0
        t = 2.5 * zoh + zoh * frac
        op = OperatingPoint(
            D.char,
            D.N + (B.N - D.N) * frac,
            D.Q + (B.Q - D.Q) * frac,
            D.H + (B.H - D.H) * frac,
        )
        prev_frac = (i - 1) / 10.0
        prev_op = OperatingPoint(
            D.char,
            D.N + (B.N - D.N) * prev_frac,
            D.Q + (B.Q - D.Q) * prev_frac,
            D.H + (B.H - D.H) * prev_frac,
        )
        add_step(t, prev_op, op)

    # Back at B at 3.5*zoh
    add_step(
        3.5 * zoh,
        OperatingPoint(
            D.char,
            D.N + (B.N - D.N) * 0.9,
            D.Q + (B.Q - D.Q) * 0.9,
            D.H + (B.H - D.H) * 0.9,
        ),
        B,
    )

    # Wait at B until 4.0*zoh (just hold, no new point needed - linear interp handles it)

    # Linear B→D from 4.0*zoh to 5.0*zoh
    add_point(4.0 * zoh, B)
    add_point(5.0 * zoh, D)

    # Linear D→B from 5.0*zoh to 6.0*zoh
    add_point(6.0 * zoh, B)

    # Step to A at 6.5*zoh
    add_step(6.5 * zoh, B, A)

    # Hold A until end
    add_point(1e10, A)

    fpoints_H_ref = FPoints.from_arrays(np.array(times), np.array(h_ref))
    fpoints_y_T = FPoints.from_arrays(np.array(times), np.array(y_t))
    fpoints_N_T = FPoints.from_arrays(np.array(times), np.array(n_t))

    return fpoints_y_T, fpoints_N_T, fpoints_H_ref


def generate_fpoints_steps(sequence: List[OperatingPoint], zoh):
    h_ref = list()
    y_t = list()
    n_t = list()
    step_times = list()

    for i, op in enumerate(sequence[1:]):
        step_times.append(zoh * (i + 1))
        h_ref.append(op.H)
        y_t.append(op.y)
        n_t.append(op.N)

    fpoints_H_ref = FPoints.steps_abs(sequence[0].H, step_times, h_ref)
    fpoints_y_T = FPoints.steps_abs(sequence[0].y, step_times, y_t)
    fpoints_N_T = FPoints.steps_abs(sequence[0].N, step_times, n_t)

    return fpoints_y_T, fpoints_N_T, fpoints_H_ref


def float_to_label(value: float) -> str:
    """Transform float [-1, 1] to label like 'p08', 'm07'."""
    prefix = "p" if value >= 0 else "m"
    digits = int(round(abs(value) * 10))
    return f"{prefix}{digits:02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PF3 Controller Comparison")
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument(
        "--swirl",
        type=str,
        choices=["S100", "S95", "S90", "S85", "S80", "S75", "S70", "S65", "S60"],
        default="S80",
    )
    parser.add_argument(
        "--zoh", type=float, default=20.0, help="Zero-order hold time between steps [s]"
    )
    parser.add_argument(
        "--head-step",
        type=float,
        default=0.8,
        help="Percentage in head change on the step (+ or - possible)",
    )
    parser.add_argument(
        "--gvo-speed-max",
        type=float,
        default=3000.0,
        help="Max GVO servomotor speed [rpm]",
    )
    parser.add_argument("--experiment-day", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=False)
    parser.add_argument(
        "--type", type=str, choices=["steps", "head", "sweep"], default="steps"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() at the end"
    )
    args = parser.parse_args()

    if args.type == "steps":
        experiment_name = args.swirl + args.sequence
    elif args.type == "head":
        experiment_name = args.type + float_to_label(args.head_step)
    elif args.type == "sweep":
        experiment_name = args.swirl + args.type
    else:
        raise ValueError("this --type is unknown")

    experiment_day = args.experiment_day
    output_dir = Path(f"experiment-series/{experiment_day}/setup/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write reproducibility info
    write_reproducibility_info(output_dir / "repro.md", sys.argv)

    data_dir = Path("data/pf3")

    # Simulation settings
    fmu_filename = data_dir / "PF3_FMI.fmu"
    start_time = 0.0
    if args.type == "steps":
        stop_time = len(args.sequence) * args.zoh
    elif args.type == "head":
        stop_time = 3 * args.zoh
    elif args.type == "sweep":
        stop_time = 7 * args.zoh
    else:
        raise ValueError("this --type is unknown")

    step_size = args.step_size

    # Controller settings
    N_P_MIN = -10000.0
    N_P_MAX = 0

    # controller initial values.
    N_P_INIT = -443.077342
    N_T_INIT = 517.7283
    y_T_INIT = 0.4706
    Pm_INIT = 24499.3318961
    Q_T_INIT = 0.27901345531
    H_REF = 10

    fturb = PumpTurbine.from_dat(Path("data/pf3/passive_elements/FTURB1.DAT"))
    char = fturb.characteristic

    # swirl number for fturb at X% BEP, reducing GVO (flow):
    swirl_values = {
        "S60": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.179751169236),
        "S65": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.189373456833),
        "S70": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.198995744431),
        "S75": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.213017177111),
        "S80": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.227038609792),
        "S85": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.240032321172),
        "S90": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.253026032551),
        "S95": fturb.characteristic.swirl_number(N=N_T_INIT, Q=0.266019743930),
        "S100": fturb.characteristic.swirl_number(N=N_T_INIT, Q=Q_T_INIT),
    }

    A = OperatingPoint(char, N_T_INIT, Q_T_INIT, H_REF)

    zoh = args.zoh
    swirl = swirl_values[args.swirl]

    if args.type == "steps":
        operating_points = generate_op_steps(char, swirl, A)

        fpoints_y_T, fpoints_N_T, fpoints_H_ref = generate_fpoints_steps(
            [operating_points[p] for p in list(args.sequence)], zoh
        )

    elif args.type == "head":
        fpoints_y_T = FPoints.constant(y_T_INIT)
        fpoints_N_T = FPoints.constant(N_T_INIT)
        fpoints_H_ref = FPoints.steps_abs(
            H_REF,
            [zoh, zoh * 2],
            [H_REF * args.head_step, H_REF],
        )
    elif args.type == "sweep":
        operating_points = generate_op_sweep(char, swirl, A)
        fpoints_y_T, fpoints_N_T, fpoints_H_ref = generate_fpoints_sweep(
            [operating_points[p] for p in list("ABD")], zoh
        )
    else:
        raise ValueError("this --type is unknown")

    # Pre-filter y_T trajectory before simulation (FPoints -> FPoints)
    fpoints_y_T = optimize_y_T_trajectory(
        fpoints_y_T, 0.4, stop_time, motor_speed_max=args.gvo_speed_max
    )

    # pl_controller = PLController(
    #     fpoints_y_T=fpoints_y_T,
    #     fpoints_N_T=fpoints_N_T,
    #     fpoints_H_ref=fpoints_H_ref,
    #     Pm_BEP=Pm_INIT,
    #     part_load=0.7,
    #     params=PLControllerParams(variable="y_T", K_p=1e-5),
    #     # params=PLControllerParams(variable="N_T", K_p=1e-2),
    #     # params=PLControllerParams(variable="H_ref", K_p=1e-4),
    # )

    pl_controller = PLNULLController(
        fpoints_y_T=fpoints_y_T,
        fpoints_N_T=fpoints_N_T,
        fpoints_H_ref=fpoints_H_ref,
    )

    fturb = PumpTurbine.from_dat(Path("data/pf3/passive_elements/FTURB1.DAT"))
    pump = PumpTurbine.from_dat(Path("data/pf3/passive_elements/PUMP1.DAT"))
    [etaT, NT, QT, TT, yT] = fturb.characteristic.bep_turb_at_head(H_REF)
    [NP, TP] = pump.characteristic.NT_from_QH(-QT / 2, H_REF)

    print("\n" + "=" * 60)
    print(
        f"STEADY-STATE VALUES @ H_ref = {H_REF} m\nassuming same flow through both pumps."
    )
    print("=" * 60)
    print("\nTURBINE (FTURB1) - BEP:")
    print(f"  N_T  = {NT:12.4f} rpm")
    print(f"  Q_T  = {QT:12.6f} m³/s")
    print(f"  T_T  = {TT:12.4f} Nm")
    print(f"  η_T  = {etaT:12.4f}")
    print(f"  y_T  = {yT:12.4f}")
    print("\nPUMPS (PUMP1, PUMP2) - each at Q_P = Q_T/2:")
    print(f"  N_P  = {NP:12.4f} rpm")
    print(f"  Q_P  = {-QT/2:12.6f} m³/s  (per pump)")
    print(f"  T_P  = {TP:12.4f} Nm")
    print("=" * 60 + "\n")

    # exit()

    controller = NMPCCasadi(
        H_ref=fpoints_H_ref,
        model=PF3LumpedCasadi(data_dir / "passive_elements"),
        fpoints_y_T=fpoints_y_T,
        fpoints_N_T=fpoints_N_T,
        params=MPCParams(
            dt=step_size,
            N_p=50,
            Q_H_T=5.0,
            Q_terminal=100.0,
            R_du=0.005,
            u_min=N_P_MIN,
            u_max=N_P_MAX,
            P_max=300000.0,
        ),
        u_nominal=N_P_INIT,
    )

    # =========================================================================
    # Run simulation
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"SIMULATION: {controller.name}")
    print("=" * 70)

    results = run_simulation(
        fmu_filename=fmu_filename,
        controller=controller,
        fpoints_y_T=fpoints_y_T,
        fpoints_N_T=fpoints_N_T,
        pl_controller=pl_controller,
        start_time=start_time,
        stop_time=stop_time,
        step_size=step_size,
        N_P_init=N_P_INIT,
    )

    print(f"\n[{controller.name}] Simulation complete!")

    # =========================================================================
    # Create Comparison Plots
    # =========================================================================
    gvo_kin = GVOKinematics()
    plot_comparison_matplotlib(
        [results],
        stop_time,
        step_size,
        output_dir,
        experiment_name,
        N_P_min=N_P_MIN,
        N_P_max=N_P_MAX,
        P_max=300000.0,  # 300 kW (from MPCParams)
        Pm_BEP=Pm_INIT,
        gvo_kin=gvo_kin,
    )

    # =========================================================================
    # Plot Hillchart with Trajectory
    # =========================================================================
    import matplotlib.pyplot as plt

    char = fturb.characteristic
    N_T_arr = np.array(results["inputs"]["TURB-N"])
    Q_T_arr = np.array(results["outputs"]["FTURB1-Q"])
    H_T_arr = np.array(results["outputs"]["FTURB1-H"])

    fig_hill, ax_hill = plt.subplots(figsize=(12, 9))
    fig_hill, ax_hill = plot_efficiency_hillchart(char, fig=fig_hill, ax=ax_hill)
    fig_hill, ax_hill, cbar = plot_trajectory_on_hillchart(
        char,
        N_T_arr,
        Q_T_arr,
        H_T_arr,
        fig=fig_hill,
        ax=ax_hill,
        cmap="coolwarm",
        label=experiment_name,
    )
    ax_hill.set_title(f"Hillchart with Trajectory: {experiment_name}")

    hillchart_path = output_dir / f"hillchart_{experiment_name}.png"
    fig_hill.savefig(hillchart_path, dpi=300)
    print(f"\n[Hillchart] Saved to: {hillchart_path}")

    if not args.no_show:
        plt.show()

    # =========================================================================
    # Export Control Sequences to FPOINTS .DAT Files
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPORTING CONTROL SEQUENCES TO .DAT FILES")
    print("=" * 80)

    # Export pump speeds (PUMP1-N, PUMP2-N)
    write_fpoints_dat(
        filepath=output_dir / f"REGP_{experiment_name}",
        times=results["timestamps"],
        values=results["inputs"]["PUMP1-N"],
        x_unit="s",
        y_unit="rpm",
    )

    # Also export turbine inputs for reference (if you want to replay them)
    write_fpoints_dat(
        filepath=output_dir / f"REGN_{experiment_name}",
        times=results["timestamps"],
        values=results["inputs"]["TURB-N"],
        x_unit="s",
        y_unit="rpm",
    )

    write_fpoints_dat(
        filepath=output_dir / f"REGY_{experiment_name}",
        times=results["timestamps"],
        values=results["inputs"]["TURB-y"],
        x_unit="s",
        y_unit="-",
    )

    # =========================================================================
    # Export control.csv (time, N_T, motor_speed_rpm, N_P) at 0.4s sampling
    # =========================================================================
    t = np.array(results["timestamps"])
    N_T = np.array(results["inputs"]["TURB-N"])
    y_T = np.array(results["inputs"]["TURB-y"])
    N_P = np.array(results["inputs"]["PUMP1-N"])

    # Save original arrays for operating points extraction
    t_orig, N_T_orig, y_T_orig, N_P_orig = t, N_T, y_T, N_P

    # Create time grid starting at 0 with 0.4s spacing
    t_csv = np.arange(0, t[-1] + 0.01, 0.4)

    # Interpolate values to the new time grid
    N_T_csv = np.interp(t_csv, t, N_T)
    y_T_csv = np.interp(t_csv, t, y_T)
    N_P_csv = np.interp(t_csv, t, N_P)

    # Compute motor speed from resampled data (correct 0.4s intervals)
    motor_speed = compute_motor_speed(t_csv, y_T_csv, gvo_kin)
    motor_speed = np.insert(motor_speed, 0, motor_speed[0])  # Pad with first value

    # Round N_T, motor_speed, N_P to integers
    N_T_int = np.round(N_T_csv).astype(int)
    motor_speed_int = np.round(motor_speed).astype(int)
    N_P_int = -np.round(N_P_csv).astype(int)

    control_data = np.column_stack([t_csv, N_T_int, motor_speed_int, N_P_int])
    control_csv_path = output_dir / f"control_{experiment_name}.csv"
    np.savetxt(
        control_csv_path,
        control_data,
        delimiter=";",
        # header="time;N_T;motor_speed_rpm;N_P",
        comments="",
        fmt=["%.2f", "%d", "%d", "%d"],
    )
    print(f"[CONTROL] Wrote {len(t_csv)} points to {control_csv_path}")

    # =========================================================================
    # Export operating points (extract from simulation at steady-state times)
    # =========================================================================
    if args.type == "steps":
        op_names = list(dict.fromkeys(args.sequence))
        # Sample times: shortly before each transition (A, B, C, D)
        sample_times = [
            zoh * (i + 1) - 8 for i in range(len(args.sequence))
        ]  # 8s before each transition

        op_file_path = output_dir / f"operating_points_{experiment_name}.md"
        with open(op_file_path, "w") as f:
            f.write("# Steady-state Operating Points\n\n")
            for name, sample_t in zip(op_names, sample_times):
                # Find closest index to sample_t in original data
                idx = np.argmin(np.abs(t_orig - sample_t))
                alpha_deg = gvo_kin.y_to_alpha(y_T_orig[idx])
                f.write(f"## {name}\n\n")
                f.write("### Turbine\n")
                f.write(f"N = {N_T_orig[idx]:.0f}\n")
                f.write(f"y = {y_T_orig[idx]:.4f}\n")
                f.write(f"alpha = {alpha_deg:.1f} deg\n\n")
                f.write("### Pumps\n")
                f.write(f"N = {N_P_orig[idx]:.0f}\n\n")
        print(f"[OPERATING POINTS] Wrote to {op_file_path}")

    # =========================================================================
    # Export full simulation results for comparison with measurements
    # =========================================================================
    sim_results_path = output_dir / f"simulation_{experiment_name}.pkl"
    with open(sim_results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[SIMULATION] Wrote full results to {sim_results_path}")

    print("\nDone!")
