"""
Plotting Functions for PF3 Experiment Data.

Contains:
- Hillchart (N11-Q11 diagram) with trajectory
- Time series comparison plots
- PCB pressure sensor plots with spectrograms
- Runner sensor plots
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from src.helpers import plot_efficiency_hillchart, plot_trajectory_on_hillchart
from src.hydraulic_elements.pump_turbine import PumpTurbine

from .calibration import PXI_SAMPLE_RATE
from .signal_processing import (
    add_gvo_overlay,
    compute_spectrogram_normalized,
    plot_spectrogram_on_axis,
)

if TYPE_CHECKING:
    from .data_types import PlatformData, PXIMeasurementData, SimulationData


def plot_hillchart(
    data: "PlatformData",
    output_path: Optional[Path] = None,
    title: str = "",
    fturb_dat_path: Path = Path("data/pf3/passive_elements/FTURB1.DAT"),
    downsample: int = 1000,
) -> plt.Figure:
    """
    Create hillchart (N11-Q11 diagram) with measured trajectory.

    Args:
        data: MeasurementData object
        output_path: Path to save figure (optional)
        title: Figure title
        fturb_dat_path: Path to FTURB1.DAT file
        downsample: Downsample factor for trajectory points (default: 1000)

    Returns:
        matplotlib Figure
    """
    # Load turbine characteristic
    fturb = PumpTurbine.from_dat(fturb_dat_path)
    char = fturb.characteristic

    # Get values from MeasurementSeries
    N_T = data.N_T.val
    Q_T = data.Q_T.val
    H_T = data.H_T.val

    # Downsample for cleaner plot
    if downsample > 1:
        N_T = N_T[::downsample]
        Q_T = Q_T[::downsample]
        H_T = H_T[::downsample]

    # Create hillchart
    fig, ax = plt.subplots(figsize=(12, 9))
    fig, ax = plot_efficiency_hillchart(char, fig=fig, ax=ax)

    # Plot trajectory
    fig, ax, cbar = plot_trajectory_on_hillchart(
        char,
        N_T,
        Q_T,
        H_T,
        fig=fig,
        ax=ax,
        cmap="coolwarm",
        label=title,
    )

    ax.set_title(f"Hillchart with Trajectory: {title}")

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved hillchart to: {output_path}")

    return fig


def plot_results_aligned(
    platform: "PlatformData",
    sim: Optional["SimulationData"] = None,
    pxi: Optional["PXIMeasurementData"] = None,
    output_path: Optional[Path] = None,
    title: str = "",
    filter_size: int = 51,
    filter_type: str = "median",
    xlim: Optional[tuple[float | None, float | None]] = None,
) -> plt.Figure:
    """
    Create 3x2 plot grid with aligned data.

    Platform, Sim, and PXI data should have masks/offsets set for alignment.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"PF3 Measurement Results: {title}", fontsize=12)

    def apply_filter(arr: np.ndarray, size: int, ftype: str) -> np.ndarray:
        if ftype == "none" or size <= 1 or len(arr) < size:
            return arr
        if ftype == "median":
            if size % 2 == 0:
                size += 1
            return median_filter(arr, size=size)
        elif ftype == "moving_avg":
            kernel = np.ones(size) / size
            return np.convolve(arr, kernel, mode="same")
        return arr

    # Extract platform data (masks/offsets applied via MeasurementSeries)
    t_daq = platform.N_T.time  # Use any DAQ series for time
    N_T = platform.N_T.val
    H_T = platform.H_T.val
    Q_T = platform.Q_T.val
    T_T = platform.T_T.val
    P_T = platform.P_T.val
    t_pump = platform.N_P1.time
    N_P1 = platform.N_P1.val
    N_P2 = platform.N_P2.val
    t_gvo = platform.y_T.time
    y_T = platform.y_T.val

    # Colors
    c_meas = "blue"
    c_filt = "red"
    c_sim = "green"
    c_pxi = "red"

    # ---------- Row 0: Turbine N_T/y_T and Pump N_P ----------
    ax_NT = axes[0, 0]
    ax_yT = ax_NT.twinx()

    ax_NT.plot(t_daq, N_T, color=c_meas, lw=0.5, alpha=0.3, label="N_T (raw)")
    ax_NT.plot(
        t_daq,
        apply_filter(N_T, filter_size, filter_type),
        color=c_meas,
        lw=1.5,
        label="N_T (platform)",
    )
    ax_yT.plot(t_gvo, y_T, color="orange", lw=1.5, ls="--", label="y_T")

    if pxi is not None:
        ax_NT.plot(
            pxi.rpm.time,
            pxi.rpm.val,
            color=c_pxi,
            lw=1.5,
            alpha=0.7,
            label="N_T (PXI)",
        )

    if sim is not None:
        ax_NT.plot(
            sim.N_T.time,
            sim.N_T.val,
            color=c_sim,
            lw=1.5,
            ls="--",
            label="N_T (sim)",
        )
        ax_yT.plot(
            sim.y_T.time,
            sim.y_T.val,
            color=c_sim,
            lw=1.5,
            ls=":",
            label="y_T (sim)",
        )

    ax_NT.set_xlabel("Time [s]")
    ax_NT.set_ylabel("N_T [rpm]", color=c_meas)
    ax_yT.set_ylabel("y_T [-]", color="orange")
    ax_NT.set_title("Turbine Speed N_T and Guide Vane y_T")
    ax_NT.legend(loc="upper left", fontsize=8)
    ax_yT.legend(loc="upper right", fontsize=8)
    ax_NT.grid(True, alpha=0.3)

    # Pump speeds
    ax_NP = axes[0, 1]
    ax_NP.plot(t_pump, N_P1, color="blue", lw=1.5, label="N_P1")
    ax_NP.plot(t_pump, N_P2, color="red", lw=1.5, label="N_P2")
    ax_NP.plot(
        t_pump, (N_P1 + N_P2) / 2, color="black", lw=1.5, ls="--", label="N_P avg"
    )
    if sim is not None:
        ax_NP.plot(
            sim.N_P1.time,
            sim.N_P1.val,
            color=c_sim,
            lw=1.5,
            ls="--",
            label="N_P (sim)",
        )
    ax_NP.set_xlabel("Time [s]")
    ax_NP.set_ylabel("N_P [rpm]")
    ax_NP.set_title("Pump Speeds")
    ax_NP.legend(loc="best", fontsize=8)
    ax_NP.grid(True, alpha=0.3)

    # ---------- Row 1: Head and Discharge ----------
    ax_HT = axes[1, 0]
    ax_HT.plot(t_daq, H_T, color=c_meas, lw=0.5, alpha=0.3, label="H_T (raw)")
    ax_HT.plot(
        t_daq,
        apply_filter(H_T, filter_size, filter_type),
        color=c_filt,
        lw=1.5,
        label="H_T (meas)",
    )
    if sim is not None:
        ax_HT.plot(
            sim.H_T.time,
            sim.H_T.val,
            color=c_sim,
            lw=1.5,
            ls="--",
            label="H_T (sim)",
        )
    ax_HT.set_xlabel("Time [s]")
    ax_HT.set_ylabel("H_T [m]")
    ax_HT.set_title("Turbine Head H_T")
    ax_HT.legend(loc="best", fontsize=8)
    ax_HT.grid(True, alpha=0.3)
    ax_HT.ticklabel_format(useOffset=False, axis="y")

    ax_QT = axes[1, 1]
    ax_QT.plot(t_daq, Q_T, color=c_meas, lw=0.5, alpha=0.3, label="Q_T (raw)")
    ax_QT.plot(
        t_daq,
        apply_filter(Q_T, filter_size, filter_type),
        color=c_filt,
        lw=1.5,
        label="Q_T (meas)",
    )
    if sim is not None:
        ax_QT.plot(
            sim.Q_T.time,
            sim.Q_T.val,
            color=c_sim,
            lw=1.5,
            ls="--",
            label="Q_T (sim)",
        )
    ax_QT.set_xlabel("Time [s]")
    ax_QT.set_ylabel("Q_T [m^3/s]")
    ax_QT.set_title("Turbine Discharge Q_T")
    ax_QT.legend(loc="best", fontsize=8)
    ax_QT.grid(True, alpha=0.3)

    # ---------- Row 2: Power and Torque ----------
    ax_PT = axes[2, 0]
    P_T_kW = P_T / 1000
    ax_PT.plot(t_daq, P_T_kW, color=c_meas, lw=0.5, alpha=0.3, label="P_T (raw)")
    ax_PT.plot(
        t_daq,
        apply_filter(P_T_kW, filter_size, filter_type),
        color=c_filt,
        lw=1.5,
        label="P_T (meas)",
    )
    if sim is not None:
        ax_PT.plot(
            sim.P_T.time,
            sim.P_T.val / 1000,
            color=c_sim,
            lw=1.5,
            ls="--",
            label="P_T (sim)",
        )
    ax_PT.set_xlabel("Time [s]")
    ax_PT.set_ylabel("P_T [kW]")
    ax_PT.set_title("Turbine Power P_T")
    ax_PT.legend(loc="best", fontsize=8)
    ax_PT.grid(True, alpha=0.3)

    ax_TT = axes[2, 1]
    ax_TT.plot(t_daq, T_T, color=c_meas, lw=0.5, alpha=0.3, label="T_T (raw)")
    ax_TT.plot(
        t_daq,
        apply_filter(T_T, filter_size, filter_type),
        color=c_filt,
        lw=1.5,
        label="T_T (meas)",
    )
    ax_TT.set_xlabel("Time [s]")
    ax_TT.set_ylabel("T_T [Nm]")
    ax_TT.set_title("Turbine Torque T_T")
    ax_TT.legend(loc="best", fontsize=8)
    ax_TT.grid(True, alpha=0.3)

    # Apply xlim to all axes if specified
    if xlim is not None:
        for ax in axes.flat:
            ax.set_xlim(xlim)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved plot to: {output_path}")

    return fig


def plot_pcb_sensors_aligned(
    pxi: "PXIMeasurementData",
    platform: "PlatformData",
    output_path: Optional[Path] = None,
    title: str = "",
    downsample: int = 100,
    filter_size: int = 51,
    xlim: Optional[tuple[float | None, float | None]] = None,
) -> plt.Figure:
    """
    Create 2x2 plot with draft tube PCB sensor time series and spectrograms.

    Layout:
        [0,0] PCB7448 time series (cone 1)     [0,1] PS1 time series (runner)
        [1,0] PCB7448 spectrogram              [1,1] PS1 spectrogram

    Spectrograms show frequency normalized by turbine rotation frequency (f/f_n)
    with guide vane opening (y_T) trajectory overlaid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Pressure Sensors (Draft Tube & Runner): {title}", fontsize=12)

    # Get raw PCB data (full rate for spectrogram, masks applied via MeasurementSeries)
    t_raw = pxi.pcb_7448.time
    pcb_7448_raw = pxi.pcb_7448.val

    # Downsample for time series plot only
    if downsample > 1:
        t_ds = t_raw[::downsample]
        pcb_7448_ds = pcb_7448_raw[::downsample]
    else:
        t_ds = t_raw
        pcb_7448_ds = pcb_7448_raw

    # Get turbine speed for frequency normalization
    N_T_mean = np.mean(pxi.rpm.val)
    f_n = N_T_mean / 60.0  # Turbine rotation frequency [Hz]

    # Get PS1 data (runner-mounted)
    ps1_raw = pxi.ps1.val
    if downsample > 1:
        ps1_ds = ps1_raw[::downsample]
    else:
        ps1_ds = ps1_raw

    sensors = [
        ("PCB7448", "Draft tube cone", pcb_7448_raw, pcb_7448_ds),
        ("PS1", "Runner", ps1_raw, ps1_ds),
    ]

    for col, (name, location, data_raw, data_ds) in enumerate(sensors):
        # ---------- Time series (top row) ----------
        ax_ts = axes[0, col]
        ax_ts.plot(t_ds, data_ds, "b-", lw=0.3, alpha=0.7, label="raw")
        # Running average (window ~1 second after downsampling)
        window = max(int(PXI_SAMPLE_RATE / downsample), 101)
        if window % 2 == 0:
            window += 1
        data_smooth = np.convolve(data_ds, np.ones(window) / window, mode="same")
        ax_ts.plot(t_ds, data_smooth, "r-", lw=1.0, label="running avg")
        ax_ts.set_xlabel("Time [s]")
        ax_ts.set_ylabel("Pressure [bar]")
        ax_ts.set_title(f"{name}: {location}")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc="upper right", fontsize=8)
        stats_text = f"mean: {data_ds.mean():.3f}\nstd: {data_ds.std():.3f}"
        ax_ts.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_ts.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # ---------- Spectrogram (bottom row) ----------
        ax_spec = axes[1, col]
        spec_data = compute_spectrogram_normalized(
            data_raw, PXI_SAMPLE_RATE, f_n, t_offset=t_raw[0]
        )
        im = plot_spectrogram_on_axis(
            ax_spec, spec_data, title=f"{name} Spectrogram (f_n = {f_n:.1f} Hz)"
        )
        fig.colorbar(
            im, ax=ax_spec, label="PSD [dB]", orientation="horizontal", pad=0.15
        )
        add_gvo_overlay(ax_spec, platform.y_T.time, platform.y_T.val)

    # Apply xlim to all axes if specified
    if xlim is not None:
        for ax in axes.flat:
            ax.set_xlim(xlim)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved PCB plot to: {output_path}")

    return fig


def plot_runner_sensors_aligned(
    pxi: "PXIMeasurementData",
    platform: "PlatformData",
    output_path: Optional[Path] = None,
    title: str = "",
    downsample: int = 100,
    xlim: Optional[tuple[float | None, float | None]] = None,
) -> plt.Figure:
    """
    Create 2x4 plot with runner pressure sensor time series and spectrograms.

    Layout:
        [0,0] PS1 time series    [0,1] PS2 time series    [0,2] PS3 time series    [0,3] PS5 time series
        [1,0] PS1 spectrogram    [1,1] PS2 spectrogram    [1,2] PS3 spectrogram    [1,3] PS5 spectrogram

    Spectrograms show frequency normalized by turbine rotation frequency (f/f_n)
    with guide vane opening (y_T) trajectory overlaid.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Runner Pressure Sensors (PS1/2/3/5): {title}", fontsize=12)

    # Get time from first sensor (all PCB sensors share same time base)
    t_raw = pxi.ps1.time

    # Downsample for time series plot
    if downsample > 1:
        t_ds = t_raw[::downsample]
    else:
        t_ds = t_raw

    # Get turbine speed for frequency normalization
    N_T_mean = np.mean(pxi.rpm.val)
    f_n = N_T_mean / 60.0  # Turbine rotation frequency [Hz]

    # Define sensors to plot (name, MeasurementSeries)
    sensors = [
        ("PS1", pxi.ps1),
        ("PS2", pxi.ps2),
        ("PS3", pxi.ps3),
        ("PS5", pxi.ps5),
    ]

    for col, (name, series) in enumerate(sensors):
        # Get values from MeasurementSeries
        data_raw = series.val

        # Downsample for time series
        if downsample > 1:
            data_ds = data_raw[::downsample]
        else:
            data_ds = data_raw

        # ---------- Time series (top row) ----------
        ax_ts = axes[0, col]
        ax_ts.plot(t_ds, data_ds, "b-", lw=0.3, alpha=0.7)
        ax_ts.set_xlabel("Time [s]")
        ax_ts.set_ylabel("Pressure [bar]")
        ax_ts.set_title(f"{name}")
        ax_ts.grid(True, alpha=0.3)
        stats_text = f"mean: {data_ds.mean():.3f}\nstd: {data_ds.std():.3f}"
        ax_ts.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_ts.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # ---------- Spectrogram (bottom row) ----------
        ax_spec = axes[1, col]
        spec_data = compute_spectrogram_normalized(
            data_raw, PXI_SAMPLE_RATE, f_n, t_offset=t_raw[0]
        )
        im = plot_spectrogram_on_axis(
            ax_spec, spec_data, title=f"{name} (f_n = {f_n:.1f} Hz)"
        )
        add_gvo_overlay(ax_spec, platform.y_T.time, platform.y_T.val)

    # Add single colorbar for spectrograms
    fig.colorbar(
        im,
        ax=axes[1, :],
        label="PSD [dB]",
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
    )

    # Apply xlim to all axes if specified
    if xlim is not None:
        for ax in axes.flat:
            ax.set_xlim(xlim)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved runner sensors plot to: {output_path}")

    return fig


def print_summary(data: "PlatformData") -> None:
    """Print summary statistics of the measurement data."""
    print("\n" + "=" * 60)
    print("MEASUREMENT SUMMARY")
    print("=" * 60)

    t = data.N_T.time
    print(f"Duration: {t[-1] - t[0]:.1f} s")

    # Get values (masks applied automatically)
    N_T = data.N_T.val
    H_T = data.H_T.val
    Q_T = data.Q_T.val
    T_T = data.T_T.val
    P_T = data.P_T.val
    N_P1 = data.N_P1.val
    N_P2 = data.N_P2.val

    print(f"\nDAQ samples: {len(N_T)} (rate: {data.N_T.freq:.0f} Hz)")
    print(f"Pump samples: {len(N_P1)}")

    print("\n--- Turbine ---")
    print(
        f"N_T:  {N_T.mean():8.1f} +/- {N_T.std():6.1f} rpm  "
        f"[{N_T.min():.1f} - {N_T.max():.1f}]"
    )
    print(
        f"H_T:  {H_T.mean():8.3f} +/- {H_T.std():6.3f} m    "
        f"[{H_T.min():.3f} - {H_T.max():.3f}]"
    )
    print(
        f"Q_T:  {Q_T.mean():8.4f} +/- {Q_T.std():6.4f} m^3/s "
        f"[{Q_T.min():.4f} - {Q_T.max():.4f}]"
    )
    print(
        f"T_T:  {T_T.mean():8.1f} +/- {T_T.std():6.1f} Nm   "
        f"[{T_T.min():.1f} - {T_T.max():.1f}]"
    )
    print(
        f"P_T:  {P_T.mean()/1000:8.2f} +/- {P_T.std()/1000:6.2f} kW   "
        f"[{P_T.min()/1000:.2f} - {P_T.max()/1000:.2f}]"
    )

    print("\n--- Pumps ---")
    print(
        f"N_P1: {N_P1.mean():8.1f} +/- {N_P1.std():6.1f} rpm  "
        f"[{N_P1.min():.1f} - {N_P1.max():.1f}]"
    )
    print(
        f"N_P2: {N_P2.mean():8.1f} +/- {N_P2.std():6.1f} rpm  "
        f"[{N_P2.min():.1f} - {N_P2.max():.1f}]"
    )

    print("=" * 60)
