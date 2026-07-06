"""
Transient PF3 TDMS Reader - Python Translation
===============================================

Reads PF3 test rig transient measurement TDMS files, applies calibrations,
and compares with SIMSEN simulation data.

Original MATLAB script: Transient_PF3_TDMS_read_fast_v7.m
Author: Nathan Veuthey - PTMH EPFL (December 2025)
Python translation: March 2026

Dependencies:
    pip install nptdms numpy scipy matplotlib

Usage:
    python transient_pf3_tdms_reader.py --tdms path/to/file.tdms --simsen path/to/folder
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile
from scipy.interpolate import interp1d

# =============================================================================
# Constants
# =============================================================================

G = 9.8063  # [m/s^2] PF3 acceleration due to gravity
RHO = 998.3  # [kg/m^3] PF3 water density

# Section areas for kinetic energy correction
S1 = 0.115654  # [m^2] upstream head measurement section
S2 = 0.299895  # [m^2] downstream head measurement section

# Atmospheric pressure calculation constants
R_AIR = 287.05  # [J/kg/K] dry air specific gas constant
ALT_PULLY = 456  # [m.s.m.]
ALT_PTMH = 403  # [m.s.m] PTMH altitude according to Swisstopo

# Default atmospheric conditions (from MeteoSuisse 14.01.2026 at 10h)
T_PULLY_DEFAULT = 8  # [°C]
P_ATM_PULLY_DEFAULT = 966.1  # [hPa]
THETA_DEFAULT = 18.2  # [°C] measured water temperature


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CalibrationCoeffs:
    """Quadratic calibration coefficients: y = A2*x^2 + A1*x + A0"""

    A2: float = 0.0
    A1: float = 1.0
    A0: float = 0.0

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Apply calibration to raw values."""
        return self.A2 * raw**2 + self.A1 * raw + self.A0


@dataclass
class DischargeCalibration:
    """Separate calibration for positive and negative flow directions."""

    pos: CalibrationCoeffs
    neg: CalibrationCoeffs

    def apply(self, raw: np.ndarray, is_positive: np.ndarray) -> np.ndarray:
        """Apply calibration based on flow direction."""
        result = np.zeros_like(raw)
        result[is_positive] = self.pos.apply(raw[is_positive])
        result[~is_positive] = self.neg.apply(raw[~is_positive])
        return result


@dataclass
class ChannelData:
    """Data for a single TDMS channel."""

    name: str
    time: np.ndarray
    raw: np.ndarray
    cal: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if len(self.cal) == 0:
            self.cal = self.raw.copy()


@dataclass
class VISData:
    """Data from a SIMSEN VIS file."""

    Time: np.ndarray
    Q: np.ndarray | None = None
    H: np.ndarray | None = None
    T: np.ndarray | None = None
    N: np.ndarray | None = None
    y: np.ndarray | None = None


@dataclass
class SequenceParams:
    """Sequence metadata."""

    direction: str  # 'P2T' or 'T2P'
    duration: float  # seconds
    delay: float  # seconds before sequence start


# =============================================================================
# Calibration Definitions
# =============================================================================


def get_calibration_coefficients() -> dict[str, CalibrationCoeffs | DischargeCalibration]:
    """
    Return calibration coefficients for all channels.

    Returns dict mapping channel field names to CalibrationCoeffs.
    """
    cal = {}

    # Torques (mV/V -> Nm)
    cal["FrictionTorque_mV_V_"] = CalibrationCoeffs(A2=-10.63833, A1=-132.36, A0=0.94167)
    cal["GuideVaneTorque1_mV_V_"] = CalibrationCoeffs(A2=0.01416, A1=-13.46, A0=4.87069)
    cal["GuideVaneTorque2_mV_V_"] = CalibrationCoeffs(A2=0.01403, A1=-13.52, A0=-4.70623)
    cal["GuideVaneTorque3_mV_V_"] = CalibrationCoeffs(A2=0.01180, A1=-14.14, A0=-2.73726)
    cal["GuideVaneTorque4_mV_V_"] = CalibrationCoeffs(A2=0.02992, A1=-15.57, A0=30.48780)

    # Axial thrust (A -> N)
    # Note: calibration was done for mA, but measurement is in A
    cal["AxialThrust_A_"] = CalibrationCoeffs(
        A2=0.18162 * 1_000_000,  # because measured in A not mA
        A1=609.87 * 1_000,
        A0=-2352.81,
    )

    # Pressure sensors - NOTE: Cone and VlSpc columns are inverted in TDMS!
    # So calibration coefficients are purposely switched
    cal["PressureCone_A_"] = CalibrationCoeffs(
        A2=-0.000001 * 1_000_000,
        A1=0.249933 * 1_000,
        A0=-0.997087,
    )
    cal["PressureVlSpc_A_"] = CalibrationCoeffs(
        A2=0.000021 * 1_000_000,
        A1=0.192588 * 1_000,
        A0=-2.794222,
    )

    # Sigma sensor (A -> bar)
    cal["Sigma_A_"] = CalibrationCoeffs(
        A2=-0.000001 * 1_000_000,
        A1=0.122690 * 1_000,
        A0=-1.471776,
    )

    # Head (A -> bar) - New calibration from 17.12.2025
    cal["Head_A_"] = CalibrationCoeffs(
        A2=0.000008 * 1_000_000,
        A1=0.143690 * 1_000,
        A0=-0.574846,
    )

    # Main torque (Hz -> Nm)
    cal["MainTorque_Hz_"] = CalibrationCoeffs(A2=0.0, A1=0.10, A0=-6006.44275)

    # Discharge requires two sets based on model speed sign
    cal["Dischage_A_"] = DischargeCalibration(
        pos=CalibrationCoeffs(
            A2=0.0 * 1_000_000,
            A1=0.076699 * 1_000,
            A0=-0.306466,
        ),
        neg=CalibrationCoeffs(
            A2=-0.000015 * 1_000_000,
            A1=0.07703 * 1_000,
            A0=-0.30759,
        ),
    )

    # Model speed - no calibration, just sign correction and Hz->rpm
    cal["ModelSpeed_Hz_"] = CalibrationCoeffs(A2=0.0, A1=1.0, A0=0.0)

    # Pump speeds - typically already in rpm, identity calibration
    cal["PumpSpeed1_min_1_"] = CalibrationCoeffs()
    cal["PumpSpeed2_min_1_"] = CalibrationCoeffs()

    # GVO - identity (already in degrees)
    cal["GVO___"] = CalibrationCoeffs()

    # Discharge direction - identity
    cal["DischargeDirection_V_"] = CalibrationCoeffs()

    return cal


# =============================================================================
# SIMSEN VIS File Reader
# =============================================================================


def load_simsen_vis_files(filenames: list[Path]) -> dict[str, VISData]:
    """
    Load SIMSEN VIS files and return structured data.

    Args:
        filenames: List of paths to VIS files

    Returns:
        Dict mapping component names (e.g., 'FTURB1') to VISData objects
    """
    vis_data = {}

    for fname in filenames:
        if not fname.exists():
            print(f"Warning: Could not open file: {fname}")
            continue

        base_name = fname.stem  # e.g., 'FTURB1'

        with open(fname, "r") as f:
            lines = f.readlines()

        # Skip first 4 header lines, line 5 has variable names
        if len(lines) < 6:
            print(f"Warning: File {fname} has too few lines")
            continue

        var_names = lines[4].strip().split()

        # Skip unit line (line 6), read numerical data from line 7 onwards
        data_lines = lines[6:]
        data = []
        for line in data_lines:
            values = line.strip().split()
            if values:
                data.append([float(v) for v in values])

        if not data:
            print(f"Warning: No data in file {fname}")
            continue

        data_array = np.array(data)

        # Create VISData object
        vis = VISData(Time=np.array([]))

        for i, var_name in enumerate(var_names):
            # Clean variable name (make valid Python identifier)
            clean_var = make_valid_name(var_name)
            if hasattr(vis, clean_var):
                setattr(vis, clean_var, data_array[:, i])

        vis_data[base_name] = vis

    return vis_data


def make_valid_name(name: str) -> str:
    """Convert string to valid Python identifier."""
    # Replace invalid characters with underscore
    valid = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure doesn't start with number
    if valid and valid[0].isdigit():
        valid = "_" + valid
    return valid


# =============================================================================
# TDMS File Reader
# =============================================================================


def read_tdms_file(
    tdms_path: Path, downsample: int = 1
) -> tuple[dict[str, ChannelData], dict[str, np.ndarray]]:
    """
    Read TDMS file and extract channel data.

    Args:
        tdms_path: Path to TDMS file
        downsample: Downsampling factor (keep 1 point every N)

    Returns:
        Tuple of (channel_data dict, timestamp dict)
    """
    tdms_file = TdmsFile.read(tdms_path)

    # Get all channels from all groups
    all_channels = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            # Use channel name as key
            name = channel.name
            data = channel[:]
            if downsample > 1:
                data = data[::downsample]
            all_channels[name] = data

    # Separate timestamp channels from data channels
    timestamps = {}
    data_channels = {}

    for name, data in all_channels.items():
        if "timestamp" in name.lower():
            timestamps[name] = np.array(data, dtype=float)
        else:
            data_channels[name] = np.array(data, dtype=float)

    return data_channels, timestamps


def identify_timestamp_roles(timestamp_names: list[str]) -> dict[str, str]:
    """
    Identify which timestamp channel belongs to which subsystem.

    Returns dict with keys 'pump', 'gvo', 'daq' mapping to channel names.
    """
    roles = {}
    for name in timestamp_names:
        ln = name.lower()
        if "pump" in ln:
            roles["pump"] = name
        elif "gvo" in ln:
            roles["gvo"] = name
        elif "daq" in ln:
            roles["daq"] = name
    return roles


def choose_timestamp_for_channel(
    channel_name: str,
    all_channel_names: list[str],
    ts_roles: dict[str, str],
    timestamp_names: list[str],
) -> str:
    """
    Choose appropriate timestamp channel for a data channel.

    Rules:
    - Pump speed channels use pump timestamp
    - GVO channels use gvo timestamp
    - Everything else uses daq timestamp
    - Fallback: use nearest preceding timestamp channel
    """
    ln = channel_name.lower()

    if "pump speed" in ln or ("pump" in ln and "gvo" not in ln):
        if "pump" in ts_roles:
            return ts_roles["pump"]
    elif "gvo" in ln:
        if "gvo" in ts_roles:
            return ts_roles["gvo"]
    else:
        if "daq" in ts_roles:
            return ts_roles["daq"]

    # Fallback: find nearest preceding timestamp
    try:
        var_idx = all_channel_names.index(channel_name)
        ts_indices = [all_channel_names.index(ts) for ts in timestamp_names if ts in all_channel_names]
        preceding = [i for i in ts_indices if i < var_idx]
        if preceding:
            return all_channel_names[max(preceding)]
    except ValueError:
        pass

    # Ultimate fallback
    return timestamp_names[0] if timestamp_names else ""


def build_relative_times(timestamps: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    """
    Convert absolute timestamps to relative times from earliest.

    Returns:
        Tuple of (relative_times dict, t0_epoch)
    """
    # Find minimum across all timestamps
    all_mins = [ts.min() for ts in timestamps.values() if len(ts) > 0]
    t0_epoch = min(all_mins) if all_mins else 0.0

    # Compute relative times
    t_rel = {name: ts - t0_epoch for name, ts in timestamps.items()}

    return t_rel, t0_epoch


# =============================================================================
# Signal Processing
# =============================================================================


def correct_model_speed_sign(
    raw_speed: np.ndarray, sequence_direction: str
) -> np.ndarray:
    """
    Correct model speed sign based on sequence direction.

    For P2T: negate values before minimum
    For T2P: negate values after minimum
    """
    corrected = raw_speed.copy()
    idx_min = np.argmin(corrected)

    if sequence_direction == "P2T":
        if idx_min > 0:
            corrected[:idx_min] = -corrected[:idx_min]
    else:  # T2P
        if idx_min < len(corrected) - 1:
            corrected[idx_min + 1 :] = -corrected[idx_min + 1 :]

    return corrected


def match_time_to_data(
    time_vector: np.ndarray, data_length: int, channel_name: str = ""
) -> np.ndarray:
    """
    Match timestamp vector length to data vector length.

    Handles cases where they differ by truncation or interpolation.
    """
    nt = len(time_vector)

    if nt == data_length:
        return time_vector.copy()
    elif nt >= data_length:
        return time_vector[:data_length].copy()
    else:
        if nt >= 2:
            # Interpolate timestamps
            print(
                f"Warning: Timestamp had fewer samples than {channel_name}; interpolating."
            )
            return np.interp(
                np.linspace(0, 1, data_length),
                np.linspace(0, 1, nt),
                time_vector,
            )
        else:
            # Synthetic time
            print(f"Warning: Using synthetic time for {channel_name}")
            return np.arange(data_length, dtype=float)


def compute_head_meters(
    head_bar: np.ndarray, discharge_m3s: np.ndarray, rho: float = RHO
) -> np.ndarray:
    """
    Convert head from bar to meters including kinetic energy correction.

    H = (E_p + E_k) / g
    where E_p = p / rho (potential energy)
          E_k = Q^2 / 2 * (1/S1^2 - 1/S2^2) (kinetic energy)
    """
    head_pa = head_bar * 1e5
    E_p = head_pa / rho  # Specific hydraulic potential energy
    E_k = discharge_m3s**2 / 2 * (1 / S1**2 - 1 / S2**2)  # Specific kinetic energy
    E = E_p + E_k  # Total specific hydraulic energy
    return E / G


def compute_water_density(theta: float) -> float:
    """
    Compute water density from temperature using Kell equation.

    Args:
        theta: Water temperature in °C

    Returns:
        Water density in kg/m³
    """
    numerator = (
        999.83952
        + 16.945176 * theta
        - 7.9870401e-3 * theta**2
        - 46.170461e-6 * theta**3
        + 105.56302e-9 * theta**4
        - 280.54253e-12 * theta**5
    )
    denominator = 1 + 16.897850e-3 * theta
    return numerator / denominator


def compute_atmospheric_pressure_ptmh(
    T_pully_celsius: float = T_PULLY_DEFAULT,
    p_atm_pully_hpa: float = P_ATM_PULLY_DEFAULT,
) -> float:
    """
    Compute atmospheric pressure at PTMH from Pully measurements.

    Returns pressure in bar.
    """
    T_pully_K = T_pully_celsius + 273.15
    p_atm_ptmh = p_atm_pully_hpa * np.exp(
        -G * (ALT_PULLY - ALT_PTMH) / R_AIR / T_pully_K
    )
    return p_atm_ptmh / 1000  # Convert hPa to bar


def compute_sigma(
    H: np.ndarray,
    Q: np.ndarray,
    delta_p_sigma_bar: np.ndarray,
    theta: float = THETA_DEFAULT,
    p_atm_bar: float | None = None,
) -> np.ndarray:
    """
    Compute cavitation number sigma = NPSH / H.

    Args:
        H: Head in meters
        Q: Discharge in m³/s
        delta_p_sigma_bar: Suction head pressure difference in bar
        theta: Water temperature in °C
        p_atm_bar: Atmospheric pressure in bar (computed if None)

    Returns:
        Sigma (cavitation number)
    """
    if p_atm_bar is None:
        p_atm_bar = compute_atmospheric_pressure_ptmh()

    rho_t = compute_water_density(theta)
    p_amb_pa = p_atm_bar * 1e5

    H_amb = p_amb_pa / rho_t / G
    H_va = 10 ** (2.7862 + 0.0312 * theta - 0.000104 * theta**2) / rho_t / G
    H_s = delta_p_sigma_bar * 1e5 / rho_t / G

    NPSH = H_amb - H_va - H_s + (Q / S2) ** 2 / (2 * G)
    sigma = NPSH / H

    return sigma


# =============================================================================
# Data Processing Pipeline
# =============================================================================


def process_tdms_data(
    tdms_path: Path,
    sequence_params: SequenceParams,
    downsample: int = 1,
) -> dict[str, ChannelData]:
    """
    Main processing pipeline for TDMS data.

    Reads file, applies calibrations, corrects signs, converts units.

    Args:
        tdms_path: Path to TDMS file
        sequence_params: Sequence metadata
        downsample: Downsampling factor

    Returns:
        Dict mapping channel names to processed ChannelData
    """
    # Read TDMS file
    raw_channels, timestamps = read_tdms_file(tdms_path, downsample)

    if not timestamps:
        raise ValueError("No timestamp channels detected in TDMS file")

    # Build relative times
    t_rel, t0_epoch = build_relative_times(timestamps)

    # Identify timestamp roles
    timestamp_names = list(timestamps.keys())
    ts_roles = identify_timestamp_roles(timestamp_names)

    # Get calibration coefficients
    cal_coeffs = get_calibration_coefficients()

    # All channel names for timestamp matching
    all_names = list(raw_channels.keys()) + timestamp_names

    # Process each channel
    data: dict[str, ChannelData] = {}

    for channel_name, raw_values in raw_channels.items():
        field_name = make_valid_name(channel_name)

        # Choose appropriate timestamp
        ts_name = choose_timestamp_for_channel(
            channel_name, all_names, ts_roles, timestamp_names
        )
        ts_field = make_valid_name(ts_name)
        time_vector = t_rel.get(ts_name, np.arange(len(raw_values), dtype=float))

        # Match time vector length to data
        time_matched = match_time_to_data(time_vector, len(raw_values), channel_name)

        # Apply default calibration if available
        cal = cal_coeffs.get(field_name)
        if isinstance(cal, CalibrationCoeffs):
            calibrated = cal.apply(raw_values)
        else:
            calibrated = raw_values.copy()

        data[field_name] = ChannelData(
            name=channel_name,
            time=time_matched,
            raw=raw_values,
            cal=calibrated,
        )

    # Special processing: Model speed sign correction
    if "ModelSpeed_Hz_" in data:
        ms = data["ModelSpeed_Hz_"]
        corrected_raw = correct_model_speed_sign(ms.raw, sequence_params.direction)
        # Apply calibration to corrected raw
        cal = cal_coeffs.get("ModelSpeed_Hz_", CalibrationCoeffs())
        if isinstance(cal, CalibrationCoeffs):
            calibrated = cal.apply(corrected_raw)
        else:
            calibrated = corrected_raw
        ms.raw = corrected_raw
        ms.cal = calibrated
    else:
        raise ValueError('Model speed channel "ModelSpeed_Hz_" not found')

    # Special processing: Discharge calibration based on model speed
    if "Dischage_A_" not in data:
        raise ValueError('Discharge channel "Dischage_A_" not found')
    if "DischargeDirection_V_" not in data:
        raise ValueError('Discharge direction channel "DischargeDirection_V_" not found')

    q_data = data["Dischage_A_"]
    dir_data = data["DischargeDirection_V_"]
    ms_data = data["ModelSpeed_Hz_"]

    # Determine high level threshold for direction
    dir_raw = dir_data.raw
    hi_level = np.median(dir_raw[dir_raw > np.mean(dir_raw)])
    if np.isnan(hi_level):
        hi_level = np.max(dir_raw)
    invert_mask = dir_raw >= hi_level * 0.8

    # Interpolate model speed onto discharge timebase
    interp_func = interp1d(
        ms_data.time, ms_data.cal, kind="linear", fill_value="extrapolate"
    )
    model_speed_at_q = interp_func(q_data.time)

    # Apply pos/neg calibration
    is_positive = model_speed_at_q >= 0
    discharge_cal = cal_coeffs["Dischage_A_"]
    if isinstance(discharge_cal, DischargeCalibration):
        q_calibrated = discharge_cal.apply(q_data.raw, is_positive)
    else:
        q_calibrated = q_data.raw.copy()

    # Note: The original MATLAB has a line that doesn't actually invert:
    # Q_corrected(invert_mask) = Q_corrected(invert_mask);
    # This appears to be a no-op, so we keep the same behavior

    q_data.cal = q_calibrated

    # Head: convert bar to meters with kinetic energy correction
    if "Head_A_" in data:
        head_data = data["Head_A_"]
        head_cal = cal_coeffs.get("Head_A_")
        if isinstance(head_cal, CalibrationCoeffs):
            head_bar = head_cal.apply(head_data.raw)
            # Need discharge on same timebase for kinetic correction
            q_interp = interp1d(
                q_data.time, q_data.cal, kind="linear", fill_value="extrapolate"
            )
            q_at_head = q_interp(head_data.time)
            head_data.cal = compute_head_meters(head_bar, q_at_head)

    # Re-apply calibration for other sensors that might not have been applied
    for field_name, channel in data.items():
        cal = cal_coeffs.get(field_name)
        if field_name in ["ModelSpeed_Hz_", "Dischage_A_", "Head_A_"]:
            continue  # Already processed
        if isinstance(cal, CalibrationCoeffs) and cal.A1 != 1.0 or cal.A0 != 0.0 or cal.A2 != 0.0:
            channel.cal = cal.apply(channel.raw)

    return data


# =============================================================================
# Plotting Functions
# =============================================================================


def pad_range(y: np.ndarray, padding: float = 0.1) -> tuple[float, float]:
    """Compute axis limits with padding."""
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    margin = padding * (y_max - y_min) if y_max != y_min else 1.0
    return y_min - margin, y_max + margin


def plot_comparison(
    data: dict[str, ChannelData],
    vis_data: dict[str, VISData],
    sequence_params: SequenceParams,
    output_dir: Path,
) -> None:
    """
    Create comparison plots: measurement vs simulation.

    Args:
        data: Processed TDMS channel data
        vis_data: SIMSEN VIS data
        sequence_params: Sequence metadata
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    win_start = sequence_params.delay
    win_end = sequence_params.delay + sequence_params.duration

    # Extract VIS data
    if "FTURB1" not in vis_data:
        print("Warning: FTURB1 not found in VIS data, skipping simulation comparison")
        vis_fturb = None
    else:
        vis_fturb = vis_data["FTURB1"]

    if "PUMP1" not in vis_data:
        print("Warning: PUMP1 not found in VIS data")
        vis_pump = None
    else:
        vis_pump = vis_data["PUMP1"]

    # Set up matplotlib style
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
    })

    # 1) GVO comparison
    if "GVO___" in data and vis_fturb is not None and vis_fturb.y is not None:
        gvo = data["GVO___"]
        idx = (gvo.time >= win_start) & (gvo.time <= win_end)
        if np.any(idx):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                gvo.time[idx] - win_start,
                gvo.cal[idx],
                linewidth=1.3,
                label="Measurement",
            )
            ax.plot(
                vis_fturb.Time,
                vis_fturb.y * 34,
                linewidth=1.3,
                label="Command",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("GVO [°]")
            ax.legend(loc="upper center")
            ax.set_title(f"GVO: measurement vs command (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "GVO_cmd_vs_meas.png", dpi=600)
            plt.close(fig)

    # 2) Model speed vs command
    if "ModelSpeed_Hz_" in data and vis_fturb is not None and vis_fturb.N is not None:
        ms = data["ModelSpeed_Hz_"]
        idx = (ms.time >= win_start) & (ms.time <= win_end)
        if np.any(idx):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                ms.time[idx] - win_start,
                ms.cal[idx],
                linewidth=1.3,
                label="Measurement",
            )
            ax.plot(vis_fturb.Time, vis_fturb.N, linewidth=1.3, label="Command")
            yl, yh = pad_range(np.concatenate([ms.cal[idx], vis_fturb.N]))
            ax.set_ylim(yl, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Model speed [rpm]")
            ax.legend(loc="lower right")
            ax.set_title(f"Model speed: measurement vs command (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "ModelSpeed_cmd_vs_meas.png", dpi=600)
            plt.close(fig)

    # 3) Pump speeds average vs command
    if (
        "PumpSpeed1_min_1_" in data
        and "PumpSpeed2_min_1_" in data
        and vis_pump is not None
        and vis_pump.N is not None
    ):
        p1 = data["PumpSpeed1_min_1_"]
        p2 = data["PumpSpeed2_min_1_"]
        idx = (p1.time >= win_start) & (p1.time <= win_end)

        # Clean outliers from VIS data
        N_FP_VIS = vis_pump.N.copy()
        outlier_mask = (N_FP_VIS == -600) | (N_FP_VIS == -50)
        outlier_indices = np.where(outlier_mask)[0]
        for oi in outlier_indices:
            if oi > 0:
                N_FP_VIS[oi] = N_FP_VIS[oi - 1]

        if np.any(idx):
            avg_pump_rpm = (p1.cal[idx] + p2.cal[idx]) / 2
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                p1.time[idx] - win_start,
                avg_pump_rpm,
                linewidth=1.3,
                label="Measurement avg pumps",
            )
            ax.plot(vis_fturb.Time, -N_FP_VIS, linewidth=1.3, label="Command")
            yl, yh = pad_range(np.concatenate([avg_pump_rpm, -N_FP_VIS]))
            ax.set_ylim(yl, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Pumps speed [rpm]")
            ax.legend(loc="lower right")
            ax.set_title(f"Pump speeds: measurement vs command (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "Pumps_cmd_vs_meas.png", dpi=600)
            plt.close(fig)

    # 4) Discharge comparison
    if "Dischage_A_" in data and vis_fturb is not None and vis_fturb.Q is not None:
        q = data["Dischage_A_"]
        idx = (q.time >= win_start) & (q.time <= win_end)
        if np.any(idx):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                q.time[idx] - win_start,
                q.cal[idx],
                linewidth=1.3,
                label="Measurement",
            )
            ax.plot(vis_fturb.Time, vis_fturb.Q, linewidth=1.3, label="Simulation")
            # Moving average
            window = min(200, len(q.cal[idx]))
            if window > 1:
                mov_mean = np.convolve(q.cal[idx], np.ones(window) / window, mode="same")
                ax.plot(
                    q.time[idx] - win_start,
                    mov_mean,
                    linewidth=1.3,
                    label="Mov mean",
                )
            yl, yh = pad_range(np.concatenate([q.cal[idx], vis_fturb.Q]))
            ax.set_ylim(yl, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Q [m³/s]")
            ax.legend(loc="lower right")
            ax.set_title(f"Discharge: measurement vs simulation (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "Discharge_meas_vs_sim.png", dpi=600)
            plt.close(fig)

    # 5) Head comparison
    if "Head_A_" in data and vis_fturb is not None and vis_fturb.H is not None:
        h = data["Head_A_"]
        idx = (h.time >= win_start) & (h.time <= win_end)
        if np.any(idx):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                h.time[idx] - win_start,
                h.cal[idx],
                linewidth=1.3,
                label="Measured",
            )
            ax.plot(vis_fturb.Time, vis_fturb.H, linewidth=1.3, label="Simulation")
            window = min(200, len(h.cal[idx]))
            if window > 1:
                mov_mean = np.convolve(h.cal[idx], np.ones(window) / window, mode="same")
                ax.plot(
                    h.time[idx] - win_start,
                    mov_mean,
                    linewidth=1.3,
                    label="Mov mean",
                )
            yl, yh = pad_range(np.concatenate([h.cal[idx], vis_fturb.H]))
            ax.set_ylim(yl, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("H [m]")
            ax.legend(loc="lower right")
            ax.set_title(f"Head: measurement vs simulation (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "Head_meas_vs_sim.png", dpi=600)
            plt.close(fig)

    # 6) Torque comparison
    if (
        "MainTorque_Hz_" in data
        and "FrictionTorque_mV_V_" in data
        and vis_fturb is not None
        and vis_fturb.T is not None
    ):
        mt = data["MainTorque_Hz_"]
        ft = data["FrictionTorque_mV_V_"]
        idx = (mt.time >= win_start) & (mt.time <= win_end)
        if np.any(idx):
            # Total measured torque (note sign convention)
            measured_T_total = -(mt.cal[idx] + ft.cal[idx])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                mt.time[idx] - win_start,
                measured_T_total,
                linewidth=1.3,
                label="T + T_fr - measurement",
            )
            ax.plot(vis_fturb.Time, vis_fturb.T, linewidth=1.3, label="T_h - simulation")
            window = min(200, len(measured_T_total))
            if window > 1:
                mov_mean = np.convolve(
                    measured_T_total, np.ones(window) / window, mode="same"
                )
                ax.plot(mt.time[idx] - win_start, mov_mean, linewidth=1.3, label="mov mean")
            yl, yh = pad_range(np.concatenate([measured_T_total, vis_fturb.T]))
            ax.set_ylim(0, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("T [Nm]")
            ax.legend(loc="upper right")
            ax.set_title(f"Torque: measurement vs simulation (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "Torque_meas_vs_sim.png", dpi=600)
            plt.close(fig)

    # 7) Sigma plot
    if "Head_A_" in data and "Sigma_A_" in data and "Dischage_A_" in data:
        h = data["Head_A_"]
        q = data["Dischage_A_"]
        sigma_sensor = data["Sigma_A_"]

        idx = (h.time >= win_start) & (h.time <= win_end)
        if np.any(idx):
            # Interpolate Q and sigma onto head timebase
            q_interp = interp1d(q.time, q.cal, kind="linear", fill_value="extrapolate")
            sigma_interp = interp1d(
                sigma_sensor.time, sigma_sensor.cal, kind="linear", fill_value="extrapolate"
            )

            Q_at_h = q_interp(h.time[idx])
            delta_p_sigma = sigma_interp(h.time[idx])
            H_windowed = h.cal[idx]

            sigma = compute_sigma(H_windowed, Q_at_h, delta_p_sigma)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(h.time[idx] - win_start, sigma, linewidth=1.3, label="Measured")
            window = min(200, len(sigma))
            if window > 1:
                mov_mean = np.convolve(sigma, np.ones(window) / window, mode="same")
                ax.plot(h.time[idx] - win_start, mov_mean, linewidth=1.3, label="Mov mean")
            yl, yh = pad_range(sigma)
            ax.set_ylim(yl, yh)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"$\sigma$ [-]")
            ax.legend(loc="lower right")
            ax.set_title(f"Sigma (window: {win_start:.1f}--{win_end:.1f} s)")
            ax.grid(True)
            fig.savefig(output_dir / "Sigma.png", dpi=600)
            plt.close(fig)


def plot_time_series(
    data: dict[str, ChannelData],
    sequence_params: SequenceParams,
    output_dir: Path,
) -> None:
    """
    Plot time series for all calibrated signals.

    Args:
        data: Processed TDMS channel data
        sequence_params: Sequence metadata
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    win_start = sequence_params.delay
    win_end = sequence_params.delay + sequence_params.duration

    # Fields to plot with their units
    fields_and_units = {
        "PumpSpeed1_min_1_": "rpm",
        "PumpSpeed2_min_1_": "rpm",
        "GVO___": "°",
        "FrictionTorque_mV_V_": "Nm",
        "AxialThrust_A_": "N",
        "Head_A_": "m",
        "Sigma_A_": "bar",
        "Dischage_A_": "m³/s",
        "DischargeDirection_V_": "[-]",
        "GuideVaneTorque1_mV_V_": "Nm",
        "GuideVaneTorque2_mV_V_": "Nm",
        "GuideVaneTorque3_mV_V_": "Nm",
        "PressureVlSpc_A_": "bar",
        "PressureCone_A_": "bar",
        "ModelSpeed_Hz_": "rpm",
        "MainTorque_Hz_": "Nm",
    }

    for field_name, unit in fields_and_units.items():
        if field_name not in data:
            print(f"Warning: Field {field_name} not found, skipping.")
            continue

        channel = data[field_name]
        t = channel.time
        y = channel.cal

        # Apply time window
        idx = (t >= win_start) & (t <= win_end)
        t_win = t[idx] - win_start
        y_win = y[idx]

        if len(t_win) == 0:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_win, y_win, linewidth=1.3)
        ax.grid(True)

        yl, yh = pad_range(y_win)
        if yl != yh:  # Robustness for broken sensors
            ax.set_ylim(yl, yh)

        # Clean title (escape underscores for matplotlib)
        title = field_name.replace("_", r"\_")
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Value [{unit}]")

        # Save with clean filename
        clean_name = make_valid_name(channel.name)
        fig.savefig(output_dir / f"{clean_name}.png", dpi=600)
        plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Read PF3 TDMS files and compare with SIMSEN simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python transient_pf3_tdms_reader.py --tdms data/764-15-845.tdms --simsen data/PF3-Simsen-Classic

The script will create comparison plots in the output directory.
        """,
    )
    parser.add_argument(
        "--tdms",
        type=Path,
        required=True,
        help="Path to TDMS file",
    )
    parser.add_argument(
        "--simsen",
        type=Path,
        required=True,
        help="Path to folder containing SIMSEN VIS files (FTURB1.VIS, PUMP1.VIS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for plots (default: output)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["P2T", "T2P"],
        default="P2T",
        help="Sequence direction: P2T (Pump to Turbine) or T2P (Turbine to Pump)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
        help="Sequence duration in seconds (default: 40)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=12.0,
        help="Sequence delay (seconds before sequence start) (default: 12)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsampling factor (default: 1, no downsampling)",
    )
    parser.add_argument(
        "--vis-files",
        type=str,
        nargs="+",
        default=["FTURB1.VIS", "PUMP1.VIS"],
        help="VIS filenames to load (default: FTURB1.VIS PUMP1.VIS)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.tdms.exists():
        print(f"Error: TDMS file not found: {args.tdms}")
        return 1

    if not args.simsen.exists():
        print(f"Error: SIMSEN folder not found: {args.simsen}")
        return 1

    # Create sequence params
    sequence_params = SequenceParams(
        direction=args.direction,
        duration=args.duration,
        delay=args.delay,
    )

    # Print atmospheric pressure for reference
    p_atm_ptmh = compute_atmospheric_pressure_ptmh()
    print(f"Computed atmospheric pressure at PTMH: {p_atm_ptmh:.4f} bar")

    # Load SIMSEN VIS files
    vis_paths = [args.simsen / f for f in args.vis_files]
    print(f"Loading SIMSEN VIS files: {vis_paths}")
    vis_data = load_simsen_vis_files(vis_paths)

    if not vis_data:
        print("Warning: No VIS data loaded")

    # Process TDMS data
    print(f"Processing TDMS file: {args.tdms}")
    data = process_tdms_data(args.tdms, sequence_params, args.downsample)

    print(f"Loaded {len(data)} channels from TDMS file")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create comparison plots
    print("Creating comparison plots...")
    plot_comparison(data, vis_data, sequence_params, args.output)

    # Create time series plots
    print("Creating time series plots...")
    plot_time_series(data, sequence_params, args.output)

    print(f"Processing complete. Plots saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
