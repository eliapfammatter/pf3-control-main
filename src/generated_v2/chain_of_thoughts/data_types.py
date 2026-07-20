"""
PF3 Simulation Core Data Types.

Defines the shared dataclasses used by the ``Plant`` and ``InputFn``
protocols and by the ``run_simulation`` orchestrator:

- ModelInputs: control inputs applied to the plant at a given instant.
- ModelOutputs: plant outputs produced at a given instant.
- ModelState: timestamped pair of the last known inputs/outputs.
- Trajectory: a time-sampled signal, evaluated by linear interpolation.
- TrajectorySet: the reference trajectories driving a scripted InputFn.
- Artefact: the full time history returned by run_simulation.

These types carry no behavior beyond simple containers and, for
``Trajectory``, interpolation. All shared state is passed explicitly
through these dataclasses; no global state is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelInputs:
    """Control inputs applied to the plant at a given time step.

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
    """Plant outputs produced at a given time step.

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
    """Timestamped snapshot of the plant's last known inputs and outputs.

    Used by ``InputFn`` implementations that need to react to the current
    plant state when computing the next ``ModelInputs``.

    Parameters
    ----------
    t : float
        Simulation time [s] at which this state was recorded.
    inputs : ModelInputs
        Inputs applied to the plant at time `t`.
    outputs : ModelOutputs
        Outputs produced by the plant at time `t`.
    """

    t: float
    inputs: ModelInputs
    outputs: ModelOutputs


@dataclass
class Trajectory:
    """A time-sampled scalar signal, evaluated by linear interpolation.

    Parameters
    ----------
    times : np.ndarray
        Strictly non-decreasing sample times [s], shape ``(N,)``.
    values : np.ndarray
        Sample values, shape ``(N,)``, aligned with `times`.

    Raises
    ------
    ValueError
        If `times` and `values` are not equal-length 1-D arrays.
    """

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the stored sample arrays.

        Raises
        ------
        ValueError
            If `times` and `values` do not have the same length.
        """
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.times.shape != self.values.shape:
            raise ValueError(
                "times and values must have the same shape, got "
                f"{self.times.shape} and {self.values.shape}"
            )

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time `t` by linear interpolation.

        Parameters
        ----------
        t : float
            Query time [s]. Values outside the sampled range are clamped
            to the boundary samples (``np.interp`` default behavior).

        Returns
        -------
        float
            Interpolated signal value at time `t`.
        """
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """Reference trajectories driving a scripted ``InputFn``.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening reference trajectory [-].
    N_T : Trajectory
        Turbine speed reference trajectory [rpm].
    H_ref : Trajectory
        Head reference trajectory [m], e.g. for feedback controllers.
    N_P : Trajectory or None, optional
        Pump speed reference trajectory [rpm]. Default is ``None`` when
        pump speed is not scripted (e.g. held constant or controlled
        externally).
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Full time history produced by ``run_simulation``.

    Notes
    -----
    Backward compatibility: `predictions` was added after the initial
    release of this dataclass and defaults to ``None``. Code consuming
    `Artefact` instances created before this field existed (e.g. loaded
    from disk) must treat a missing/``None`` `predictions` as "no
    prediction data available" rather than as an error.

    Parameters
    ----------
    t : np.ndarray
        Simulation time vector [s], shape ``(N,)``.
    inputs : dict[str, np.ndarray]
        Mapping from ``ModelInputs`` field name to its time history,
        each of shape ``(N,)``.
    outputs : dict[str, np.ndarray]
        Mapping from ``ModelOutputs`` field name to its time history,
        each of shape ``(N,)``.
    metadata : dict
        Free-form run metadata (e.g. plant/controller configuration).
    predictions : dict or None, optional
        Optional mapping of predicted/forecast signals (e.g. from an
        MPC horizon), keyed by signal name. Default is ``None``.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    predictions: dict | None = None
