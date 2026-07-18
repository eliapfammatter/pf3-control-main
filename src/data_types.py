"""Shared data structures for the PF3 hydropower simulation software.

This module defines the dataclasses that flow between the ``Plant`` and
``InputFn`` protocols and the ``run_simulation`` orchestrator: model
inputs/outputs/state, trajectory sampling helpers, and the artefact
produced at the end of a simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelInputs:
    """Inputs applied to the plant at a given simulation step.

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
    """Outputs produced by the plant at a given simulation step.

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
    """Snapshot of the plant state at a given simulation time.

    Parameters
    ----------
    t : float
        Simulation time [s].
    inputs : ModelInputs
        Inputs applied at time ``t``.
    outputs : ModelOutputs
        Outputs produced at time ``t``.
    """

    t: float
    inputs: ModelInputs
    outputs: ModelOutputs


@dataclass
class Trajectory:
    """A sampled time series that can be evaluated at arbitrary times.

    Values outside the sampled time range are clamped to the first or
    last sample, matching the default boundary behaviour of
    ``numpy.interp``.

    Parameters
    ----------
    times : numpy.ndarray
        Sample times [s], expected to be sorted in ascending order.
    values : numpy.ndarray
        Sample values, one per entry in `times`.
    """

    times: np.ndarray
    values: np.ndarray

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time `t` by linear interpolation.

        Parameters
        ----------
        t : float
            Time at which to evaluate the trajectory [s].

        Returns
        -------
        float
            Interpolated value at time `t`, clamped to the sample
            range boundaries if `t` falls outside it.

        Raises
        ------
        ValueError
            If the trajectory has no samples.
        """
        if self.times.size == 0 or self.values.size == 0:
            raise ValueError("Trajectory has no samples to interpolate.")
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """Collection of reference trajectories driving a simulation run.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening reference trajectory [-].
    N_T : Trajectory
        Turbine speed reference trajectory [rpm].
    H_ref : Trajectory
        Head reference trajectory [m].
    N_P : Trajectory or None, optional
        Pump speed reference trajectory [rpm], by default None.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Result of a full simulation run produced by `run_simulation`.

    Notes
    -----
    For backward compatibility with earlier PF3 tooling, the
    ``predictions`` field defaults to ``None`` rather than an empty
    dict. Consumers written against older Artefact producers that did
    not populate predictions should treat ``None`` and ``{}``
    equivalently.

    Parameters
    ----------
    t : numpy.ndarray
        Simulation time vector [s].
    inputs : dict[str, numpy.ndarray]
        Recorded input signals keyed by `ModelInputs` field name.
    outputs : dict[str, numpy.ndarray]
        Recorded output signals keyed by `ModelOutputs` field name.
    metadata : dict
        Free-form metadata about the run (e.g. plant name, solver
        settings, git revision).
    predictions : dict or None, optional
        Optional model predictions keyed by signal name, by default
        None.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    predictions: dict | None = None
