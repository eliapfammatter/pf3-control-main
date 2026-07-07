"""Core data structures shared across the PF3 simulation stack.

Defines the plain dataclasses used to pass state between the plant model,
controllers and the simulation orchestrator: model inputs/outputs/state,
reference trajectories and the recorded simulation artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ModelInputs:
    """Inputs applied to the plant at a given time step.

    Parameters
    ----------
    y_T : float
        Turbine guide vane opening, in [-] (fraction of max opening).
    N_T : float
        Turbine rotational speed, in rpm.
    N_P : float
        Pump rotational speed, in rpm.
    """

    y_T: float
    N_T: float
    N_P: float


@dataclass
class ModelOutputs:
    """Outputs computed by the plant at a given time step.

    Parameters
    ----------
    H_T : float
        Turbine head, in m.
    Q_T : float
        Turbine flow rate, in m^3/s.
    H_P1 : float
        Head across pump 1, in m.
    H_P2 : float
        Head across pump 2, in m.
    """

    H_T: float
    Q_T: float
    H_P1: float
    H_P2: float


@dataclass
class ModelState:
    """Snapshot of the plant at a given simulation time.

    Parameters
    ----------
    t : float
        Simulation time, in s.
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
    """A time-indexed signal, linearly interpolated between samples.

    Parameters
    ----------
    times : numpy.ndarray
        Strictly increasing sample times, in s.
    values : numpy.ndarray
        Sample values, same length as `times`.

    Raises
    ------
    ValueError
        If `times` and `values` do not have matching lengths.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length")

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time `t` via linear interpolation.

        Parameters
        ----------
        t : float
            Query time, in s.

        Returns
        -------
        float
            Interpolated value at time `t`.
        """
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """Reference trajectories driving an open-loop controller.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening trajectory.
    N_T : Trajectory
        Turbine speed trajectory.
    H_ref : Trajectory
        Reference head trajectory.
    N_P : Trajectory or None, optional
        Pump speed trajectory. Defaults to None, treated as 0 when absent.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Recorded outcome of a simulation run.

    Parameters
    ----------
    t : numpy.ndarray
        Simulation time vector, in s.
    inputs : dict[str, numpy.ndarray]
        Recorded input signals, keyed by `ModelInputs` field name.
    outputs : dict[str, numpy.ndarray]
        Recorded output signals, keyed by `ModelOutputs` field name.
    metadata : dict[str, typing.Any]
        Run metadata, e.g. git hash, timestamp.
    predictions : dict[str, numpy.ndarray] or None, optional
        Optional model predictions, e.g. from a state estimator. Defaults
        to None.

    Notes
    -----
    `predictions` was added after the initial release of this dataclass.
    It defaults to None and is optional so that code and serialized
    artefacts created before its introduction remain valid without
    modification, preserving backward compatibility.
    """

    t: NDArray[np.float64]
    inputs: dict[str, NDArray[np.float64]]
    outputs: dict[str, NDArray[np.float64]]
    metadata: dict[str, Any]
    predictions: dict[str, NDArray[np.float64]] | None = None
