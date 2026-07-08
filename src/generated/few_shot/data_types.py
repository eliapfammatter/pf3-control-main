"""
Shared data structures for the PF3 hydropower simulation.

Defines the dataclasses that flow between Plant and InputFn
implementations (ModelInputs, ModelOutputs, ModelState), the
trajectory replay helpers (Trajectory, TrajectorySet), and the
simulation result container (Artefact) produced by run_simulation.
"""

from __future__ import annotations

# Dataclass fields below (y_T, N_T, N_P, H_T, Q_T, H_P1, H_P2, H_ref) use
# domain-standard turbine/pump notation mandated by the module contract
# rather than PEP 8 snake_case; the module-wide suppression avoids
# repeating this justification on every affected class/field.
# pylint: disable=invalid-name

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
    """Plant outputs computed at a given simulation step.

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
    """Full plant state at a given simulation time.

    Parameters
    ----------
    t : float
        Simulation time [s].
    inputs : ModelInputs
        Inputs applied at time t.
    outputs : ModelOutputs
        Outputs computed at time t.
    """

    t: float
    inputs: ModelInputs
    outputs: ModelOutputs


@dataclass
class Trajectory:
    """A time-indexed signal, linearly interpolated when called.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sample times [s].
    values : np.ndarray
        Signal values at each sample time, same length as times.
    """

    times: np.ndarray
    values: np.ndarray

    def __call__(self, t: float) -> float:
        """Return the linearly interpolated value at time t.

        Parameters
        ----------
        t : float
            Query time [s]. Values outside the sampled range are
            clamped to the nearest endpoint (np.interp default).

        Returns
        -------
        float
            Interpolated signal value at time t.
        """
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """Reference trajectories used by open-loop InputFn implementations.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening trajectory [-].
    N_T : Trajectory
        Turbine speed trajectory [rpm].
    H_ref : Trajectory
        Reference head trajectory [m].
    N_P : Trajectory | None, optional
        Pump speed trajectory [rpm], by default None.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Container for the full time history produced by run_simulation.

    Notes
    -----
    For backward compatibility with earlier PF3 result formats, the
    ``inputs`` and ``outputs`` fields use plain ``dict[str, np.ndarray]``
    mappings (keyed by field name) rather than arrays of dataclass
    instances, and ``predictions`` defaults to None when a controller
    does not expose internal predictions.

    Parameters
    ----------
    t : np.ndarray
        Simulation time vector [s].
    inputs : dict[str, np.ndarray]
        Time series of each ModelInputs field, keyed by field name.
    outputs : dict[str, np.ndarray]
        Time series of each ModelOutputs field, keyed by field name.
    metadata : dict
        Free-form run metadata (e.g. plant/controller names, config).
    predictions : dict | None, optional
        Optional controller-internal predictions, by default None.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    predictions: dict | None = None
