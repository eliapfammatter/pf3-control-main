"""
PF3 Core Data Types.

Defines the shared dataclasses that flow between the ``Plant`` and
``InputFn`` protocols and the ``run_simulation`` orchestrator:

- ModelInputs: plant excitation signals at a single instant.
- ModelOutputs: plant response signals at a single instant.
- ModelState: full instantaneous state (time, inputs, outputs).
- Trajectory: a time series with linear-interpolation lookup.
- TrajectorySet: the trajectories needed to drive a simulation.
- Artefact: the recorded result of a full simulation run.

All physical quantities are expressed in SI units unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Number of points required for np.interp to produce a meaningful
# interpolation (fewer points make a trajectory lookup ill-defined).
MIN_TRAJECTORY_POINTS = 2


@dataclass
class ModelInputs:
    """Instantaneous plant inputs.

    Parameters
    ----------
    y_T : float
        Turbine guide vane opening [-].
    N_T : float
        Turbine rotational speed [rpm].
    N_P : float
        Pump rotational speed [rpm].
    """

    y_T: float
    N_T: float
    N_P: float


@dataclass
class ModelOutputs:
    """Instantaneous plant outputs.

    Parameters
    ----------
    H_T : float
        Turbine head [m].
    Q_T : float
        Turbine flow rate [m3/s].
    H_P1 : float
        Pump 1 head [m].
    H_P2 : float
        Pump 2 head [m].
    """

    H_T: float
    Q_T: float
    H_P1: float
    H_P2: float


@dataclass
class ModelState:
    """Full instantaneous state of a simulated plant.

    Parameters
    ----------
    t : float
        Simulation time [s].
    inputs : ModelInputs
        Plant inputs at time `t`.
    outputs : ModelOutputs
        Plant outputs at time `t`.
    """

    t: float
    inputs: ModelInputs
    outputs: ModelOutputs


@dataclass
class Trajectory:
    """A time series with linear-interpolation lookup.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sample times [s].
    values : np.ndarray
        Sample values, same length as `times`.

    Raises
    ------
    ValueError
        On construction, if `times` and `values` have different lengths.
        On call, if there are fewer than `MIN_TRAJECTORY_POINTS` samples.
    """

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate that `times` and `values` have matching shapes.

        Raises
        ------
        ValueError
            If `times` and `values` have different lengths.
        """
        if len(self.times) != len(self.values):
            raise ValueError(
                "Trajectory 'times' and 'values' must have equal length: "
                f"got {len(self.times)} and {len(self.values)}."
            )

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time `t` by linear interpolation.

        Parameters
        ----------
        t : float
            Query time [s]. Values outside the sampled range are clamped
            to the first/last sample (standard `np.interp` behaviour).

        Returns
        -------
        float
            Interpolated value at time `t`.

        Raises
        ------
        ValueError
            If the trajectory has fewer than `MIN_TRAJECTORY_POINTS`
            samples to interpolate from.
        """
        if len(self.times) < MIN_TRAJECTORY_POINTS:
            raise ValueError(
                "Trajectory requires at least "
                f"{MIN_TRAJECTORY_POINTS} samples, got {len(self.times)}."
            )
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """The set of trajectories needed to drive a simulation.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening reference trajectory [-].
    N_T : Trajectory
        Turbine speed reference trajectory [rpm].
    H_ref : Trajectory
        Head reference trajectory [m].
    N_P : Trajectory | None, optional
        Pump speed reference trajectory [rpm], by default None.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Recorded result of a full simulation run.

    Notes
    -----
    Backward compatibility: `metadata` and `predictions` are open-ended
    dictionaries so that new fields can be added by producers/consumers
    of this artefact without breaking the dataclass schema or requiring
    a version bump. Existing keys in `inputs`/`outputs` should not be
    renamed or removed once published, to keep older analysis scripts
    working against newer artefacts.

    Parameters
    ----------
    t : np.ndarray
        Simulation time vector [s].
    inputs : dict[str, np.ndarray]
        Recorded input signals, keyed by `ModelInputs` field name.
    outputs : dict[str, np.ndarray]
        Recorded output signals, keyed by `ModelOutputs` field name.
    metadata : dict
        Free-form run metadata (e.g. plant configuration, solver settings).
    predictions : dict | None, optional
        Optional model predictions (e.g. from an observer or controller),
        by default None.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] | None = None
