"""
Shared Data Structures for the PF3 Simulation Software

This module defines the dataclasses shared across the PF3 codebase: the
per-timestep input/output/state records, the trajectory helpers used to
drive simulations, and the artefact produced by a completed simulation run.

All shared state in this project is passed explicitly via these dataclasses
rather than through global variables, per the project's architecture rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelInputs:
    """Single-timestep control inputs to the PF3 plant.

    Parameters
    ----------
    y_T : float
        Guide vane opening of the turbine, dimensionless (typically in
        [0, 1]).
    N_T : float
        Turbine rotational speed, in revolutions per minute [rpm].
    N_P : float
        Pump rotational speed, in revolutions per minute [rpm].
    """

    y_T: float
    N_T: float
    N_P: float


@dataclass
class ModelOutputs:
    """Single-timestep measured/simulated outputs of the PF3 plant.

    Parameters
    ----------
    H_T : float
        Turbine head, in metres [m].
    Q_T : float
        Turbine flow rate, in cubic metres per second [m3/s].
    H_P1 : float
        Pump 1 head, in metres [m].
    H_P2 : float
        Pump 2 head, in metres [m].
    """

    H_T: float
    Q_T: float
    H_P1: float
    H_P2: float


@dataclass
class ModelState:
    """Complete state of the plant at a single point in time.

    Parameters
    ----------
    t : float
        Simulation time, in seconds [s].
    inputs : ModelInputs
        Inputs applied at time `t`.
    outputs : ModelOutputs
        Outputs produced at time `t`.
    """

    t: float
    inputs: ModelInputs
    outputs: ModelOutputs


@dataclass
class Trajectory:
    """A time series that can be evaluated at arbitrary times by interpolation.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sample times, in seconds [s].
    values : np.ndarray
        Sample values corresponding to `times`, same length as `times`.

    Raises
    ------
    ValueError
        If `times` and `values` do not have the same length.
    """

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate that `times` and `values` are consistent in length.

        Raises
        ------
        ValueError
            If `times` and `values` do not have the same length.
        """
        if len(self.times) != len(self.values):
            raise ValueError("`times` and `values` must have the same length")

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time `t` via linear interpolation.

        Parameters
        ----------
        t : float
            Query time, in seconds [s]. Values outside the range of
            `times` are clamped to the boundary samples (as per
            `numpy.interp`).

        Returns
        -------
        float
            Interpolated value at time `t`.
        """
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """Reference trajectories used to drive a simulation run.

    Parameters
    ----------
    y_T : Trajectory
        Reference guide vane opening trajectory.
    N_T : Trajectory
        Reference turbine speed trajectory.
    H_ref : Trajectory
        Reference head trajectory.
    N_P : Trajectory or None, optional
        Reference pump speed trajectory, by default None.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Recorded result of a completed simulation run.

    Notes
    -----
    Backward compatibility: the `predictions` field defaults to `None` so
    that artefacts produced by earlier code paths (which did not record
    model predictions) remain valid instances of this dataclass. Consumers
    should treat a `None` value as "no predictions available" rather than
    an error.

    Parameters
    ----------
    t : np.ndarray
        Simulation time vector, in seconds [s].
    inputs : dict[str, np.ndarray]
        Recorded input signals keyed by field name (e.g. "y_T", "N_T",
        "N_P"), each an array aligned with `t`.
    outputs : dict[str, np.ndarray]
        Recorded output signals keyed by field name (e.g. "H_T", "Q_T",
        "H_P1", "H_P2"), each an array aligned with `t`.
    metadata : dict
        Free-form metadata describing the run (e.g. solver settings,
        model version, timestamps).
    predictions : dict or None, optional
        Recorded model predictions, if any, by default None.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    predictions: dict | None = None
