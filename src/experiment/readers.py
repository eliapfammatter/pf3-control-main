"""
TDMS and Pickle File Readers for PF3 Experiment Data.

Contains:
- read_platform_tdms: Read platform computer TDMS files
- read_pxi_tdms: Read PXI measurement computer TDMS files
- load_simulation: Load simulation results from pickle files
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
from nptdms import TdmsFile

from .calibration import (
    CAL,
    G,
    GVO_TO_Y_FACTOR,
    PCB_INFO,
    PXI_SAMPLE_RATE,
    RHO,
    S1,
    S2,
)
from .data_types import PlatformData, PXIMeasurementData, SimulationData
from .measurement import MeasurementSeries
from .signal_processing import extract_rpm_from_tachometer


def read_platform_tdms(tdms_path: Path) -> PlatformData:
    """
    Read TDMS file and return processed measurement data.

    Args:
        tdms_path: Path to TDMS file

    Returns:
        PlatformData with MeasurementSeries for each channel
    """
    tdms_file = TdmsFile.read(tdms_path)

    # Get the Raw_Double group
    group = tdms_file["Raw_Double"]

    def get_channel(name: str) -> np.ndarray:
        """Get channel data as float array."""
        return np.array(group[name][:], dtype=np.float64)

    # Read timestamps
    ts_daq = get_channel(
        "Timestamp DAQ [seconds since the epoch 01/01/1904 00:00:00.00 UTC]"
    )
    ts_pump = get_channel(
        "Timestamp pump[seconds since the epoch 01/01/1904 00:00:00.00 UTC]"
    )
    ts_gvo = get_channel(
        "Timestamp GVO [seconds since the epoch 01/01/1904 00:00:00.00 UTC]"
    )

    # Convert to relative time
    t0 = min(ts_daq.min(), ts_pump.min(), ts_gvo.min())
    t_daq = ts_daq - t0
    t_pump = ts_pump - t0
    t_gvo = ts_gvo - t0

    # Compute nominal sample rates
    freq_daq = float(1.0 / np.mean(np.diff(ts_daq)))
    freq_pump = float(1.0 / np.mean(np.diff(ts_pump)))
    freq_gvo = float(1.0 / np.mean(np.diff(ts_gvo)))

    # Read DAQ raw channels
    head_A = get_channel("Head [A]")
    discharge_A = get_channel("Dischage [A]")
    model_speed_hz = get_channel("Model Speed [Hz]")
    main_torque_hz = get_channel("Main Torque [Hz]")
    friction_torque_mVV = get_channel("Friction Torque [mV/V]")
    sigma_A = get_channel("Sigma [A]")
    axial_thrust_A = get_channel("Axial Thrust [A]")

    # Read pump/GVO raw channels
    pump_speed_1_raw = get_channel("Pump Speed 1 [min-1]")
    pump_speed_2_raw = get_channel("Pump Speed 2 [min-1]")
    gvo_deg = get_channel("GVO [°]")

    t_start_daq = t_daq[0]

    # Calibration functions (closures for interdependent channels)
    def cal_H_T(raw_head_A: np.ndarray) -> np.ndarray:
        head_bar = CAL["Head [A]"](raw_head_A)
        Q_T = CAL["Dischage [A]"](discharge_A)
        E_p = head_bar * 1e5 / RHO
        E_k = Q_T**2 / 2 * (1 / S1**2 - 1 / S2**2)
        return (E_p + E_k) / G

    def cal_T_T(raw_main_torque: np.ndarray) -> np.ndarray:
        T_main = CAL["Main Torque [Hz]"](raw_main_torque)
        T_friction = CAL["Friction Torque [mV/V]"](friction_torque_mVV)
        return -(T_main + T_friction)

    def cal_P_T(raw_speed: np.ndarray) -> np.ndarray:
        omega = raw_speed * 2 * np.pi / 60
        T_main = CAL["Main Torque [Hz]"](main_torque_hz)
        T_friction = CAL["Friction Torque [mV/V]"](friction_torque_mVV)
        T_T = -(T_main + T_friction)
        return T_T * omega

    return PlatformData(
        # DAQ channels (uniform sampling, with calibration)
        N_T=MeasurementSeries(
            raw_data=model_speed_hz,
            freq=freq_daq,
            t_start=t_start_daq,
            name="N_T",
            unit="rpm",
        ),
        H_T=MeasurementSeries(
            raw_data=head_A,
            freq=freq_daq,
            t_start=t_start_daq,
            name="H_T",
            unit="m",
            calibrate_fn=cal_H_T,
        ),
        Q_T=MeasurementSeries(
            raw_data=discharge_A,
            freq=freq_daq,
            t_start=t_start_daq,
            name="Q_T",
            unit="m^3/s",
            calibrate_fn=CAL["Dischage [A]"],
        ),
        T_T=MeasurementSeries(
            raw_data=main_torque_hz,
            freq=freq_daq,
            t_start=t_start_daq,
            name="T_T",
            unit="Nm",
            calibrate_fn=cal_T_T,
        ),
        P_T=MeasurementSeries(
            raw_data=model_speed_hz,
            freq=freq_daq,
            t_start=t_start_daq,
            name="P_T",
            unit="W",
            calibrate_fn=cal_P_T,
        ),
        sigma=MeasurementSeries(
            raw_data=sigma_A,
            freq=freq_daq,
            t_start=t_start_daq,
            name="sigma",
            unit="bar",
            calibrate_fn=CAL["Sigma [A]"],
        ),
        axial_thrust=MeasurementSeries(
            raw_data=axial_thrust_A,
            freq=freq_daq,
            t_start=t_start_daq,
            name="axial_thrust",
            unit="N",
            calibrate_fn=CAL["Axial Thrust [A]"],
        ),
        # Pump channels (non-uniform, actual timestamps)
        N_P1=MeasurementSeries(
            raw_data=pump_speed_1_raw,
            freq=freq_pump,
            time_raw=t_pump,
            name="N_P1",
            unit="rpm",
            calibrate_fn=lambda x: -x,  # negate for convention
        ),
        N_P2=MeasurementSeries(
            raw_data=pump_speed_2_raw,
            freq=freq_pump,
            time_raw=t_pump,
            name="N_P2",
            unit="rpm",
            calibrate_fn=lambda x: -x,  # negate for convention
        ),
        # GVO channels (non-uniform, actual timestamps)
        y_T=MeasurementSeries(
            raw_data=gvo_deg,
            freq=freq_gvo,
            time_raw=t_gvo,
            name="y_T",
            unit="-",
            calibrate_fn=lambda x: x / GVO_TO_Y_FACTOR,
        ),
        GVO_deg=MeasurementSeries(
            raw_data=gvo_deg,
            freq=freq_gvo,
            time_raw=t_gvo,
            name="GVO_deg",
            unit="deg",
        ),
    )


def read_pxi_tdms(pxi_path: Path) -> PXIMeasurementData:
    """
    Read PXI measurement computer TDMS file (PCB sensors).

    Note: The "Time" channel in these TDMS files has INCORRECT timestamps
    (shows ~4900s with fake 950ms gaps). The actual data is continuous at
    20 kHz, giving ~245s of data for 4.9M samples.

    Returns:
        PXIMeasurementData with MeasurementSeries for each channel
    """
    tdms_file = TdmsFile.read(pxi_path)
    group = tdms_file["PXI-Slots"]

    def get_channel(name: str) -> np.ndarray:
        return np.array(group[name][:], dtype=np.float64)

    # Read raw data (NOTE: Time channel is WRONG in this TDMS!)
    t_raw_wrong = get_channel("Time")
    n1_raw = get_channel("N_1")

    n_samples = len(n1_raw)
    actual_duration = n_samples / PXI_SAMPLE_RATE

    print(
        f"PXI: {n_samples:,} samples, duration: {actual_duration:.1f}s "
        f"(TDMS Time showed {t_raw_wrong[-1]:.1f}s - WRONG)"
    )

    # Read PCB sensor raw data (V)
    pcb_6905_raw = get_channel("PCB6905")
    pcb_6908_raw = get_channel("PCB6908")
    pcb_7448_raw = get_channel("PCB7448")
    pcb_34317_raw = get_channel("PCB34317")
    pcb_34319_raw = get_channel("PCB34319")

    # Read runner-mounted pressure sensors (already calibrated)
    ps1_raw = get_channel("PS1")
    ps2_raw = get_channel("PS2")
    ps3_raw = get_channel("PS3")
    ps5_raw = get_channel("PS5")

    # Extract RPM from N_1 tachometer
    rpm_1hz_val, _, rpm_t_start = extract_rpm_from_tachometer(
        n1_raw, PXI_SAMPLE_RATE, resample_freq=1.0
    )
    print(f"PXI: RPM range: {rpm_1hz_val.min():.0f} - {rpm_1hz_val.max():.0f}")

    # PCB calibration: V -> bar (divide by sensitivity)
    def make_pcb_calibrator(sensor_id: str) -> Callable[[np.ndarray], np.ndarray]:
        sens = PCB_INFO[sensor_id]["sens_V_per_bar"]
        return lambda x: x / sens

    # Create MeasurementSeries for each channel
    return PXIMeasurementData(
        # RPM is already processed (not raw sensor data)
        rpm=MeasurementSeries(
            raw_data=rpm_1hz_val,
            freq=1.0,
            t_start=rpm_t_start,
            name="rpm",
            unit="rpm",
        ),
        # PCB sensors with V -> bar calibration
        pcb_6905=MeasurementSeries(
            raw_data=pcb_6905_raw,
            freq=PXI_SAMPLE_RATE,
            name="PCB6905",
            unit="bar",
            calibrate_fn=make_pcb_calibrator("PCB6905"),
        ),
        pcb_6908=MeasurementSeries(
            raw_data=pcb_6908_raw,
            freq=PXI_SAMPLE_RATE,
            name="PCB6908",
            unit="bar",
            calibrate_fn=make_pcb_calibrator("PCB6908"),
        ),
        pcb_7448=MeasurementSeries(
            raw_data=pcb_7448_raw,
            freq=PXI_SAMPLE_RATE,
            name="PCB7448",
            unit="bar",
            calibrate_fn=make_pcb_calibrator("PCB7448"),
        ),
        pcb_34317=MeasurementSeries(
            raw_data=pcb_34317_raw,
            freq=PXI_SAMPLE_RATE,
            name="PCB34317",
            unit="bar",
            calibrate_fn=make_pcb_calibrator("PCB34317"),
        ),
        pcb_34319=MeasurementSeries(
            raw_data=pcb_34319_raw,
            freq=PXI_SAMPLE_RATE,
            name="PCB34319",
            unit="bar",
            calibrate_fn=make_pcb_calibrator("PCB34319"),
        ),
        # Runner pressure sensors (no calibration)
        ps1=MeasurementSeries(
            raw_data=ps1_raw,
            freq=PXI_SAMPLE_RATE,
            t_start=0.0,
            name="PS1",
            unit="bar",
        ),
        ps2=MeasurementSeries(
            raw_data=ps2_raw,
            freq=PXI_SAMPLE_RATE,
            t_start=0.0,
            name="PS2",
            unit="bar",
        ),
        ps3=MeasurementSeries(
            raw_data=ps3_raw,
            freq=PXI_SAMPLE_RATE,
            t_start=0.0,
            name="PS3",
            unit="bar",
        ),
        ps5=MeasurementSeries(
            raw_data=ps5_raw,
            freq=PXI_SAMPLE_RATE,
            t_start=0.0,
            name="PS5",
            unit="bar",
        ),
    )


def load_simulation_raw(pkl_path: Path) -> dict[str, Any]:
    """Load raw simulation results from pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_simulation(pkl_path: Path) -> SimulationData:
    """Load simulation results as SimulationData.

    Args:
        pkl_path: Path to simulation pickle file

    Returns:
        SimulationData with MeasurementSeries for each channel
    """
    raw = load_simulation_raw(pkl_path)

    timestamps = np.array(raw["timestamps"])
    dt = timestamps[1] - timestamps[0]  # Uniform sampling
    freq = 1.0 / dt
    t_start = timestamps[0]

    def make_series(data: Any, name: str, unit: str) -> MeasurementSeries:
        return MeasurementSeries(
            raw_data=np.array(data),
            freq=freq,
            t_start=t_start,
            name=name,
            unit=unit,
        )

    inputs = raw["inputs"]
    outputs = raw["outputs"]

    return SimulationData(
        # Inputs (pump speeds already negative in simulation, no negation needed)
        N_P1=make_series(inputs["PUMP1-N"], "N_P1", "rpm"),
        N_P2=make_series(inputs["PUMP2-N"], "N_P2", "rpm"),
        N_T=make_series(inputs["TURB-N"], "N_T", "rpm"),
        y_T=make_series(inputs["TURB-y"], "y_T", "-"),
        # Outputs
        Q_T=make_series(outputs["FTURB1-Q"], "Q_T", "m^3/s"),
        H_T=make_series(outputs["FTURB1-H"], "H_T", "m"),
        P_T=make_series(outputs["FTURB1-Pm"], "P_T", "W"),
        T_T=make_series(outputs["FTURB1-T"], "T_T", "Nm"),
        # Reference
        H_ref=make_series(raw.get("H_ref", timestamps * 0), "H_ref", "m"),
        # Metadata
        controller_name=raw.get("controller_name", ""),
    )
