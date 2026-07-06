"""
PF3 Experiment Data Type Containers.

Contains dataclasses for holding measurement data from different sources:
- PlatformData: Platform computer measurements (DAQ + pumps + GVO)
- PXIMeasurementData: PXI computer measurements (PCB sensors)
- SimulationData: Simulation results with same interface as PlatformData
"""

from __future__ import annotations

from dataclasses import dataclass

from .measurement import MaskableDataMixin, MeasurementSeries


@dataclass
class PlatformData(MaskableDataMixin):
    """Container for processed platform measurement data.

    All fields are MeasurementSeries with their own sample rate and lazy time.
    Use crop() to select time range and align time axes.

    Sample rates (typical):
        - DAQ channels (N_T, H_T, Q_T, etc.): ~2000 Hz
        - Pump channels (N_P1, N_P2): ~5 Hz
        - GVO channels (y_T, GVO_deg): ~5 Hz
    """

    # Turbine (DAQ rate)
    N_T: MeasurementSeries  # Turbine speed [rpm]
    H_T: MeasurementSeries  # Turbine head [m]
    Q_T: MeasurementSeries  # Turbine discharge [m^3/s]
    T_T: MeasurementSeries  # Turbine torque [Nm]
    P_T: MeasurementSeries  # Turbine power [W]

    # Other DAQ sensors
    sigma: MeasurementSeries  # Sigma [bar]
    axial_thrust: MeasurementSeries  # Axial thrust [N]

    # Pumps (low rate)
    N_P1: MeasurementSeries  # Pump 1 speed [rpm]
    N_P2: MeasurementSeries  # Pump 2 speed [rpm]

    # Guide vane (low rate)
    y_T: MeasurementSeries  # Guide vane opening [-] (0-1)
    GVO_deg: MeasurementSeries  # Guide vane opening [deg]


@dataclass
class PXIMeasurementData(MaskableDataMixin):
    """Container for PXI measurement computer data (PCB sensors).

    All fields are MeasurementSeries with their own sample rate and lazy time.
    Use crop() to select time range and align time axes.

    Sample rates:
        - PCB/PS sensors: 20 kHz
        - RPM (derived from tachometer): ~1 Hz (variable)
    """

    # Turbine RPM from tachometer (derived, ~1 Hz)
    rpm: MeasurementSeries  # Turbine RPM [rpm]

    # PCB pressure sensors (20 kHz)
    pcb_6905: MeasurementSeries  # Vaneless gap [bar]
    pcb_6908: MeasurementSeries  # Vaneless gap [bar]
    pcb_7448: MeasurementSeries  # Draft tube cone 1 [bar]
    pcb_34317: MeasurementSeries  # Draft tube cone 2 [bar]
    pcb_34319: MeasurementSeries  # Upstream pipe [bar]

    # Runner-mounted pressure sensors (20 kHz)
    ps1: MeasurementSeries  # Runner pressure sensor 1 [bar]
    ps2: MeasurementSeries  # Runner pressure sensor 2 [bar]
    ps3: MeasurementSeries  # Runner pressure sensor 3 [bar]
    ps5: MeasurementSeries  # Runner pressure sensor 5 [bar]


@dataclass
class SimulationData(MaskableDataMixin):
    """Simulation results with same interface as PlatformData.

    All fields are MeasurementSeries with uniform sampling.
    Use crop() to select time range and align time axes.
    """

    # Inputs
    N_P1: MeasurementSeries  # PUMP1-N [rpm]
    N_P2: MeasurementSeries  # PUMP2-N [rpm]
    N_T: MeasurementSeries  # TURB-N [rpm]
    y_T: MeasurementSeries  # TURB-y [-]

    # Outputs
    Q_T: MeasurementSeries  # FTURB1-Q [m^3/s]
    H_T: MeasurementSeries  # FTURB1-H [m]
    P_T: MeasurementSeries  # FTURB1-Pm [W]
    T_T: MeasurementSeries  # FTURB1-T [Nm]

    # Reference
    H_ref: MeasurementSeries  # H reference [m]

    # Metadata
    controller_name: str = ""
