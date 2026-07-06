"""
PF3 Experiment Calibration Coefficients and Constants.

Contains:
- Physical constants (gravity, water density, section areas)
- Sensor calibration coefficients from MATLAB calibration scripts
- PCB pressure sensor sensitivity values
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================

G = 9.8063  # [m/s^2] PF3 acceleration due to gravity
RHO = 998.3  # [kg/m^3] PF3 water density

# Section areas for kinetic energy correction
S1 = 0.115654  # [m^2] upstream head measurement section
S2 = 0.299895  # [m^2] downstream head measurement section

# GVO conversion: GVO [deg] = y_T * 34
GVO_TO_Y_FACTOR = 34.0

# =============================================================================
# PXI Measurement Computer Parameters
# =============================================================================

PXI_SAMPLE_RATE = 20000  # Hz
PXI_BURST_DURATION = 0.05  # seconds (50ms per burst)
PXI_GAP_THRESHOLD = 0.1  # seconds (to detect gaps between bursts)

# =============================================================================
# PCB Sensor Information
# =============================================================================

PCB_INFO = {
    "PCB6905": {"location": "Vaneless gap", "sens_V_per_bar": 0.13560},
    "PCB6908": {"location": "Vaneless gap", "sens_V_per_bar": 0.13351},
    "PCB7448": {"location": "Draft tube cone 1", "sens_V_per_bar": 0.13722},
    "PCB34317": {"location": "Draft tube cone 2", "sens_V_per_bar": 1.50381},
    "PCB34319": {"location": "Upstream pipe", "sens_V_per_bar": 1.42059},
}

# =============================================================================
# Calibration Coefficients
# =============================================================================


@dataclass
class CalCoeffs:
    """Quadratic calibration: y = A2*x^2 + A1*x + A0"""

    A2: float = 0.0
    A1: float = 1.0
    A0: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.A2 * x**2 + self.A1 * x + self.A0


# Calibration coefficients (from Transient_PF3_TDMS_read_fast_v7.m)
CAL = {
    # Head (A -> bar) - New calibration from 17.12.2025
    "Head [A]": CalCoeffs(
        A2=0.000008 * 1_000_000,
        A1=0.143690 * 1_000,
        A0=-0.574846,
    ),
    # Main torque (Hz -> Nm)
    "Main Torque [Hz]": CalCoeffs(A2=0.0, A1=0.10, A0=-6006.44275),
    # Friction torque (mV/V -> Nm)
    "Friction Torque [mV/V]": CalCoeffs(A2=-10.63833, A1=-132.36, A0=0.94167),
    # Discharge (A -> m^3/s) - positive flow calibration
    "Dischage [A]": CalCoeffs(
        A2=0.0,
        A1=0.076699 * 1_000,
        A0=-0.306466,
    ),
    # Sigma (A -> bar)
    "Sigma [A]": CalCoeffs(
        A2=-0.000001 * 1_000_000,
        A1=0.122690 * 1_000,
        A0=-1.471776,
    ),
    # Axial thrust (A -> N)
    "Axial Thrust [A]": CalCoeffs(
        A2=0.18162 * 1_000_000,
        A1=609.87 * 1_000,
        A0=-2352.81,
    ),
    # Pressure sensors (A -> bar) - NOTE: Cone/VlSpc swapped in TDMS!
    "Pressure Cone WK1 Fuji [A]": CalCoeffs(
        A2=-0.000001 * 1_000_000,
        A1=0.249933 * 1_000,
        A0=-0.997087,
    ),
    "Pressure VlSpc WK2 Rosemount [A]": CalCoeffs(
        A2=0.000021 * 1_000_000,
        A1=0.192588 * 1_000,
        A0=-2.794222,
    ),
    # Guide vane torques (mV/V -> Nm)
    "Guide Vane Torque 1 [mV/V]": CalCoeffs(A2=0.01416, A1=-13.46, A0=4.87069),
    "Guide Vane Torque 2 [mV/V]": CalCoeffs(A2=0.01403, A1=-13.52, A0=-4.70623),
    "Guide Vane Torque 3 [mV/V]": CalCoeffs(A2=0.01180, A1=-14.14, A0=-2.73726),
    "Guide Vane Torque 4 [mV/V]": CalCoeffs(A2=0.02992, A1=-15.57, A0=30.48780),
}
