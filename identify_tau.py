"""
Identify time constant τ from FMU step response.

Uses the pump speed step test FMU to fit the first-order model time constant.

Usage:
    python identify_tau.py [--stop-time 10] [--plot]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from scipy.optimize import curve_fit

from src.characteristics import Characteristics
from src.helpers import FPoints
from src.pf3_static_model_b2 import PF3StaticModelB2


DATA_DIR = Path("data/pf3/test-step/pump-n")


def run_fmu_step_response(stop_time: float, step_size: float = 0.01) -> dict:
    """Run FMU with pump speed step input."""

    fmu_file = DATA_DIR / "pf3_step_pump_n.fmu"
    fpoints_y_T = FPoints(DATA_DIR / "REGY")
    fpoints_N_T = FPoints(DATA_DIR / "REGN")
    fpoints_N_P = FPoints(DATA_DIR / "REGP")

    model_description = read_model_description(fmu_file, validate=False)

    in_vars = {}
    out_vars = {}
    for var in model_description.modelVariables:
        if var.causality == "input":
            in_vars[var.name] = var.valueReference
        elif var.causality == "output":
            out_vars[var.name] = var.valueReference

    print(f"FMU: {model_description.modelIdentifier}")
    print(f"Inputs: {list(in_vars.keys())}")

    unzipdir = extract(fmu_file)
    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.modelIdentifier,
        instanceName="simsen",
    )

    fmu.instantiate()
    fmu.setupExperiment(startTime=0.0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    # Set initial pump speed
    pump_input_names = [k for k in in_vars.keys() if "PUMP" in k and "-N" in k]
    N_P_init = fpoints_N_P(0.0)
    if pump_input_names:
        fmu.setReal([in_vars[k] for k in pump_input_names], [N_P_init] * len(pump_input_names))

    # Storage
    timestamps = []
    H_T_list = []
    N_P_list = []

    time = 0.0
    while time < stop_time:
        N_P = fpoints_N_P(time)

        if pump_input_names:
            fmu.setReal([in_vars[k] for k in pump_input_names], [N_P] * len(pump_input_names))

        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
        time += step_size

        timestamps.append(time)
        N_P_list.append(N_P)
        H_T_list.append(fmu.getReal([out_vars["FTURB1-H"]])[0])

    fmu.terminate()
    fmu.freeInstance()

    return {
        "t": np.array(timestamps),
        "H_T": np.array(H_T_list),
        "N_P": np.array(N_P_list),
    }


def first_order_response(t, H_0, delta_H, tau, t_step):
    """First-order step response: H(t) = H_0 + ΔH·(1 - exp(-(t-t_step)/τ))"""
    return np.where(
        t >= t_step,
        H_0 + delta_H * (1.0 - np.exp(-(t - t_step) / tau)),
        H_0,
    )


def fit_tau(t, H_T, t_step: float) -> dict:
    """Fit time constant τ from step response data."""

    # Extract data after step
    mask = t >= t_step
    t_after = t[mask]
    H_after = H_T[mask]

    # Initial and final values
    H_0 = np.mean(H_T[t < t_step]) if np.any(t < t_step) else H_T[0]
    H_ss = np.mean(H_after[-10:])  # Average last 10 points for steady state
    delta_H_actual = H_ss - H_0

    # Curve fitting
    def model(t_vals, delta_H, tau):
        return first_order_response(t_vals, H_0, delta_H, tau, t_step)

    try:
        # Handle sign of delta_H for bounds
        if delta_H_actual < 0:
            bounds = ([delta_H_actual * 2, 0.01], [delta_H_actual * 0.5, 10.0])
        else:
            bounds = ([delta_H_actual * 0.5, 0.01], [delta_H_actual * 2, 10.0])

        popt, pcov = curve_fit(
            model, t_after, H_after,
            p0=[delta_H_actual, 0.5],
            bounds=bounds,
            maxfev=5000,
        )
        delta_H_fit, tau_fit = popt
        tau_std = np.sqrt(np.diag(pcov))[1]

        # Fit quality
        H_fit = model(t_after, delta_H_fit, tau_fit)
        rmse = np.sqrt(np.mean((H_after - H_fit) ** 2))
        r_squared = 1 - np.sum((H_after - H_fit)**2) / np.sum((H_after - np.mean(H_after))**2)

    except Exception as e:
        print(f"Curve fitting failed: {e}")
        tau_fit, tau_std, delta_H_fit, rmse, r_squared = None, None, None, None, None

    # 63.2% criterion
    H_63 = H_0 + 0.632 * (H_ss - H_0)
    idx_63 = np.argmin(np.abs(H_after - H_63))
    tau_63 = t_after[idx_63] - t_step

    return {
        "tau_fit": tau_fit,
        "tau_63": tau_63,
        "tau_std": tau_std,
        "H_0": H_0,
        "H_ss": H_ss,
        "delta_H_actual": delta_H_actual,
        "delta_H_fit": delta_H_fit,
        "rmse": rmse,
        "r_squared": r_squared,
    }


def main():
    parser = argparse.ArgumentParser(description="Identify τ from FMU step response")
    parser.add_argument("--stop-time", type=float, default=10.0, help="Simulation stop time [s]")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    t_step = 2.0  # Step occurs at t=2s (from REGP)
    N_P_0 = -313.2579
    N_P_1 = -300.0
    delta_N_P = N_P_1 - N_P_0  # +13.26 rpm

    print("=" * 60)
    print("TIME CONSTANT IDENTIFICATION FROM FMU STEP RESPONSE")
    print("=" * 60)
    print(f"\nStep: N_P = {N_P_0:.1f} → {N_P_1:.1f} rpm (ΔN_P = {delta_N_P:.2f} rpm) at t = {t_step}s")

    # Run FMU
    print(f"\nRunning FMU simulation (0 to {args.stop_time}s)...")
    result = run_fmu_step_response(args.stop_time)
    print(f"  Got {len(result['t'])} points")

    # Fit τ
    print("\nFitting time constant τ...")
    fit = fit_tau(result["t"], result["H_T"], t_step)

    # Compute K_P from actual response
    K_P_measured = fit["delta_H_actual"] / delta_N_P

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTime Constant τ:")
    if fit["tau_fit"] is not None:
        print(f"  Curve fit:       τ = {fit['tau_fit']:.4f} s ± {fit['tau_std']:.4f} s")
    else:
        print(f"  Curve fit:       (failed)")
    print(f"  63.2% criterion: τ = {fit['tau_63']:.4f} s")

    print(f"\nHead Change:")
    print(f"  H_0 (before step):  {fit['H_0'] * 1000:.2f} mm")
    print(f"  H_ss (after step):  {fit['H_ss'] * 1000:.2f} mm")
    print(f"  ΔH_T (actual):      {fit['delta_H_actual'] * 1000:.2f} mm")

    print(f"\nStatic Gain (from response):")
    print(f"  K_P = ΔH_T / ΔN_P = {K_P_measured:.6e} m/rpm")

    if fit["r_squared"] is not None:
        print(f"\nFit Quality:")
        print(f"  R² = {fit['r_squared']:.6f}")
        print(f"  RMSE = {fit['rmse'] * 1000:.3f} mm")

    tau_best = fit["tau_fit"] if fit["tau_fit"] else fit["tau_63"]
    print("\n" + "=" * 60)
    print(f"RECOMMENDED τ = {tau_best:.4f} s")
    print("=" * 60)

    # Plot
    if args.plot or args.save:
        fig, axes = plt.subplots(1, 2, figsize=(14, 3))

        ax = axes[0]
        ax.plot(result["t"], result["H_T"] * 1000, "b-", lw=1.5, label="FMU")

        if fit["tau_fit"]:
            H_fit = first_order_response(result["t"], fit["H_0"], fit["delta_H_fit"], fit["tau_fit"], t_step)
            ax.plot(result["t"], H_fit * 1000, "r--", lw=1.5, label=f"First-order fit (τ={fit['tau_fit']:.3f}s)")

        ax.axhline(fit["H_ss"] * 1000, color="gray", ls=":", alpha=0.7)
        ax.axvline(t_step, color="green", ls="--", alpha=0.5, label=f"Step at t={t_step}s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("H_T [mm]")
        ax.set_title(f"Step Response: ΔN_P = {delta_N_P:.1f} rpm")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(result["t"], result["N_P"], "k-", lw=1.5)
        ax.axvline(t_step, color="green", ls="--", alpha=0.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("N_P [rpm]")
        ax.set_title("Pump Speed Input")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if args.save:
            plt.savefig(args.save, dpi=150)
            print(f"\nPlot saved to: {args.save}")

        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
