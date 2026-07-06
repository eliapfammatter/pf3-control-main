"""Part-load controller for tracking a target power fraction.

Adjusts one of the FPOINTS series (y_T, N_T, or H_ref) using proportional
control to track a target part load of Pm_BEP. The correction is computed
relative to the base FPOINTS trajectory and accumulated over time.
"""

from dataclasses import dataclass
from typing import Tuple, Union

from src.helpers import FPoints


class OffsetFPoints:
    """FPoints wrapper that adds a constant offset to the base trajectory."""

    def __init__(self, base: FPoints, offset: float):
        """
        Parameters
        ----------
        base : FPoints
            Base FPOINTS trajectory
        offset : float
            Constant offset to add to all values
        """
        self.base = base
        self.offset = offset

    def __call__(self, time: float) -> float:
        """Evaluate at time with offset applied."""
        return self.base(time) + self.offset


@dataclass
class PLControllerParams:
    """Parameters for part-load controller.

    Attributes
    ----------
    variable : str
        Which FPOINTS to adjust: "y_T", "N_T", or "H_ref"
    K_p : float
        Proportional gain for control law:
        correction = K_p * (Pm_target - Pm_T)
    """

    variable: str
    K_p: float

    def __post_init__(self):
        valid_variables = {"y_T", "N_T", "H_ref"}
        if self.variable not in valid_variables:
            raise ValueError(
                f"variable must be one of {valid_variables}, got '{self.variable}'"
            )


class PLNULLController:
    def __init__(
        self,
        fpoints_y_T: FPoints,
        fpoints_N_T: FPoints,
        fpoints_H_ref: FPoints,
    ):
        self.fpoints_y_T = fpoints_y_T
        self.fpoints_N_T = fpoints_N_T
        self.fpoints_H_ref = fpoints_H_ref

    def __call__(self, time: float, Pm_T: float) -> Tuple[
        Union[FPoints, OffsetFPoints],
        Union[FPoints, OffsetFPoints],
        Union[FPoints, OffsetFPoints],
    ]:
        return (self.fpoints_N_T, self.fpoints_y_T, self.fpoints_H_ref)


class PLController:
    """Part-load controller using proportional control.

    Adjusts one of the three FPOINTS series to track a target mechanical
    power equal to part_load * Pm_BEP.

    Example
    -------
    >>> pl_controller = PLController(
    ...     fpoints_y_T=fpoints_y_T,
    ...     fpoints_N_T=fpoints_N_T,
    ...     fpoints_H_ref=fpoints_H_ref,
    ...     Pm_BEP=24500.0,
    ...     part_load=0.8,
    ...     params=PLControllerParams(variable="y_T", K_p=1e-6),
    ... )
    >>> # In simulation loop:
    >>> (fpoints_N_T, fpoints_y_T, fpoints_H_ref) = pl_controller(time, Pm_T)
    >>> N_T = fpoints_N_T(time)
    >>> y_T = fpoints_y_T(time)
    """

    def __init__(
        self,
        fpoints_y_T: FPoints,
        fpoints_N_T: FPoints,
        fpoints_H_ref: FPoints,
        Pm_BEP: float,
        part_load: float,
        params: PLControllerParams,
    ):
        """
        Parameters
        ----------
        fpoints_y_T : FPoints
            Base turbine guide vane trajectory
        fpoints_N_T : FPoints
            Base turbine speed trajectory
        fpoints_H_ref : FPoints
            Base head reference trajectory
        Pm_BEP : float
            Mechanical power at best efficiency point [W]
        part_load : float
            Target part load ratio (e.g., 0.8 = 80% of Pm_BEP)
        params : PLControllerParams
            Controller parameters (variable selection and gain)
        """
        self.fpoints_y_T = fpoints_y_T
        self.fpoints_N_T = fpoints_N_T
        self.fpoints_H_ref = fpoints_H_ref
        self.Pm_BEP = Pm_BEP
        self.part_load = part_load
        self.params = params

        # Target power
        self.Pm_target = part_load * Pm_BEP

        # Accumulated correction (stateful)
        self.accumulated_correction = 0.0

    def reset(self) -> None:
        """Reset accumulated correction to zero."""
        self.accumulated_correction = 0.0

    def __call__(self, time: float, Pm_T: float) -> Tuple[
        Union[FPoints, OffsetFPoints],
        Union[FPoints, OffsetFPoints],
        Union[FPoints, OffsetFPoints],
    ]:
        """Compute adjusted FPOINTS to track part load.

        Parameters
        ----------
        time : float
            Current simulation time [s]
        Pm_T : float
            Current turbine mechanical power [W]

        Returns
        -------
        tuple of (fpoints_N_T, fpoints_y_T, fpoints_H_ref)
            Adjusted FPOINTS, with the selected variable modified by
            proportional control. Unmodified variables return the
            original FPoints objects.
        """
        # Compute error and accumulate correction
        error = self.Pm_target - Pm_T
        self.accumulated_correction += self.params.K_p * error

        # Apply accumulated correction to selected variable
        if self.params.variable == "y_T":
            return (
                self.fpoints_N_T,
                OffsetFPoints(self.fpoints_y_T, self.accumulated_correction),
                self.fpoints_H_ref,
            )
        elif self.params.variable == "N_T":
            return (
                OffsetFPoints(self.fpoints_N_T, self.accumulated_correction),
                self.fpoints_y_T,
                self.fpoints_H_ref,
            )
        else:  # H_ref
            return (
                self.fpoints_N_T,
                self.fpoints_y_T,
                OffsetFPoints(self.fpoints_H_ref, self.accumulated_correction),
            )

    @property
    def name(self) -> str:
        """Controller name for logging."""
        return f"PLController({self.params.variable}, PL={self.part_load:.0%})"
