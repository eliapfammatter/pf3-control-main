"""Shared dataclasses for the PF3 hydropower simulation software.

This module defines the core data structures exchanged between the
:class:`Plant` and :class:`InputFn` protocols and the simulation
orchestrator: model inputs/outputs/state, trajectories, trajectory
sets, and the simulation artefact produced by ``run_simulation``.

All fields use SI units unless noted otherwise.
"""

from __future__ import annotations

# Dataclass fields below (y_T, N_T, N_P, H_T, Q_T, H_P1, H_P2, H_ref) use
# domain-standard turbine/pump notation mandated by the module contract
# rather than PEP 8 snake_case; the module-wide suppression avoids
# repeating this justification on every affected class/field.
# pylint: disable=invalid-name

from dataclasses import dataclass, field

import numpy as np

# Named constants (no magic numbers)
LINEAR_INTERP_MIN_POINTS = 1


@dataclass
class ModelInputs:
    """Inputs applied to the plant model at a given simulation step.

    Parameters
    ----------
    y_T : float
        Turbine guide vane opening [-].
    N_T : float
        Turbine rotational speed [rpm].
    N_P : float
        Pump rotational speed [rpm].

    Notes
    -----
    Field names follow the domain-standard turbine/pump notation
    (``y_T``, ``N_T``, ``N_P``) mandated by the module contract rather
    than PEP 8 snake_case, hence the local ``invalid-name`` suppression.
    """

    y_T: float
    N_T: float
    N_P: float


@dataclass
class ModelOutputs:
    """Outputs produced by the plant model at a given simulation step.

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

    Notes
    -----
    Field names follow the domain-standard turbine/pump notation
    (``H_T``, ``Q_T``, ``H_P1``, ``H_P2``) mandated by the module
    contract rather than PEP 8 snake_case, hence the local
    ``invalid-name`` suppression.
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
    """A time-indexed signal that can be evaluated by interpolation.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing sample times [s].
    values : np.ndarray
        Sample values corresponding to ``times``.
    """

    times: np.ndarray
    values: np.ndarray

    def __call__(self, t: float) -> float:
        """Evaluate the trajectory at time ``t`` by linear interpolation.

        Parameters
        ----------
        t : float
            Query time [s].

        Returns
        -------
        float
            Interpolated value at time ``t``, clamped to the boundary
            values outside the sampled time range.

        Raises
        ------
        ValueError
            If the trajectory has no sample points.
        """
        if self.times.size < LINEAR_INTERP_MIN_POINTS:
            raise ValueError("Trajectory has no sample points to interpolate.")
        return float(np.interp(t, self.times, self.values))


@dataclass
class TrajectorySet:
    """A bundle of reference trajectories for a simulation run.

    Parameters
    ----------
    y_T : Trajectory
        Guide vane opening reference trajectory [-].
    N_T : Trajectory
        Turbine speed reference trajectory [rpm].
    H_ref : Trajectory
        Reference head trajectory [m].
    N_P : Trajectory | None, optional
        Pump speed reference trajectory [rpm], by default None.

    Notes
    -----
    Field names follow the domain-standard turbine/pump notation
    (``y_T``, ``N_T``, ``H_ref``, ``N_P``) mandated by the module
    contract rather than PEP 8 snake_case, hence the local
    ``invalid-name`` suppression.
    """

    y_T: Trajectory
    N_T: Trajectory
    H_ref: Trajectory
    N_P: Trajectory | None = None


@dataclass
class Artefact:
    """Result of a full simulation run produced by ``run_simulation``.

    Notes
    -----
    Backward compatibility: ``predictions`` defaults to ``None`` so that
    artefacts produced by earlier versions of the orchestrator (which
    did not populate model predictions) can still be constructed and
    consumed without modification. Consumers must treat a ``None``
    value as "no predictions available" rather than an error.

    Parameters
    ----------
    t : np.ndarray
        Simulation time vector [s].
    inputs : dict[str, np.ndarray]
        Recorded input signals keyed by field name.
    outputs : dict[str, np.ndarray]
        Recorded output signals keyed by field name.
    metadata : dict
        Free-form metadata describing the simulation run.
    predictions : dict | None, optional
        Recorded model predictions keyed by field name, by default None.
    """

    t: np.ndarray
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    predictions: dict | None = None
