"""
Guide Vane Opening (GVO) Kinematics.

Converts between:
- y: SIMSEN normalized opening [0, 1]
- GVO: Guide vane angle [deg] = y * GVO_MAX
- S: Servo screw position [mm] (nonlinear relationship with GVO)
- motor_turns: Servomotor turns from reference position

Physical chain:
    y -> GVO [deg] -> S [mm] -> screw_turns -> motor_turns

The S <-> alpha relationship is nonlinear and loaded from matlab-scripts/*.mat.
Polynomial fits are used for fast evaluation.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat


class GVOKinematics:
    """Guide vane kinematics: y <-> servomotor position.

    Uses polynomial fits for S <-> alpha conversion (max error ~0.01 deg).
    """

    def __init__(self, mat_dir: Path = Path("data/pf3/calibration")):
        # Load lookup tables
        self.S = loadmat(mat_dir / "S.mat")["S"].flatten()  # screw position [mm]
        self.alpha = loadmat(mat_dir / "alpha.mat")["alpha"].flatten()  # GV angle [deg]

        # Parameters from GVO_kinematcs_v10.py
        self.thread_pitch = 2.0  # mm/turn
        self.reduction_ratio = 15  # motor turns per screw turn
        self.gvo_max = 34.0  # max GVO angle [deg] (y=1 corresponds to 34 deg)

        # Reference position (y=0 -> GVO=0 deg)
        self._S_ref = float(interp1d(self.alpha, self.S, kind="cubic")(0.0))

        # Scipy interpolators for numeric operations
        self._alpha_to_S = interp1d(
            self.alpha, self.S, kind="cubic", fill_value="extrapolate"
        )
        self._S_to_alpha = interp1d(
            self.S, self.alpha, kind="cubic", fill_value="extrapolate"
        )

        # Polynomial fits for fast evaluation
        # S -> alpha: cubic polynomial (max error ~0.01 deg)
        self._poly_S_to_alpha = np.polyfit(self.S, self.alpha, deg=3)
        # alpha -> S: cubic polynomial
        self._poly_alpha_to_S = np.polyfit(self.alpha, self.S, deg=3)

    def y_to_alpha(self, y: float) -> float:
        """Convert y (0-1) to GVO angle [deg]."""
        return self.gvo_max * y

    def y_to_motor_turns(self, y: float) -> float:
        """Convert y (0-1) to servomotor turns from y=0 reference."""
        alpha = self.y_to_alpha(y)
        S = float(self._alpha_to_S(alpha))
        screw_turns = (S - self._S_ref) / self.thread_pitch
        return screw_turns * self.reduction_ratio

    def motor_turns_to_y(self, motor_turns: float) -> float:
        """Convert servomotor turns back to y."""
        screw_turns = motor_turns / self.reduction_ratio
        S = self._S_ref + screw_turns * self.thread_pitch
        gvo = float(self._S_to_alpha(S))
        return gvo / self.gvo_max

    @property
    def motor_turns_min(self) -> float:
        """Motor turns at y=0."""
        return self.y_to_motor_turns(0.0)

    @property
    def motor_turns_max(self) -> float:
        """Motor turns at y=1."""
        return self.y_to_motor_turns(1.0)

    def print_summary(self):
        """Print kinematics summary."""
        print("GVO Kinematics")
        print("=" * 50)
        print(f"GVO max angle: {self.gvo_max} deg")
        print(f"Thread pitch: {self.thread_pitch} mm/turn")
        print(f"Reduction ratio: {self.reduction_ratio}:1")
        print(f"S reference (y=0): {self._S_ref:.2f} mm")
        print(f"\nMotor turns range:")
        print(f"  y=0.0 -> {self.motor_turns_min:.2f} turns")
        print(f"  y=0.5 -> {self.y_to_motor_turns(0.5):.2f} turns")
        print(f"  y=1.0 -> {self.motor_turns_max:.2f} turns")


def compute_motor_speed(
    time: np.ndarray,
    y_T: np.ndarray,
    gvo: GVOKinematics,
) -> np.ndarray:
    """Compute servomotor speed [rpm] from y_T trajectory.

    Parameters
    ----------
    time : np.ndarray
        Time points [s]
    y_T : np.ndarray
        Guide vane opening values (normalized 0-1)
    gvo : GVOKinematics
        Kinematics instance

    Returns
    -------
    np.ndarray
        Motor speed [rpm] (N-1 values for N points)
    """
    motor_turns = np.array([gvo.y_to_motor_turns(y) for y in y_T])
    dt = np.diff(time)
    d_motor_turns = np.diff(motor_turns)
    return d_motor_turns / dt * 60.0
