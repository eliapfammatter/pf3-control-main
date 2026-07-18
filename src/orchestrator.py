"""
PF3 Simulation Orchestrator.

Defines the ``Plant`` and ``InputFn`` protocols that every plant model
and input generator must satisfy, and the ``run_simulation`` function
that drives a fixed-step simulation loop and packs the resulting time
history into an ``Artefact``.

No global state is used: all state flows through explicit dataclass
arguments (``ModelInputs``, ``ModelOutputs``, ``ModelState``) and is
accumulated locally within ``run_simulation``.
"""

from __future__ import annotations

import subprocess
from dataclasses import fields
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from src.data_types import Artefact, ModelInputs, ModelOutputs, ModelState


class Plant(Protocol):
    """Interface implemented by simulated pump-turbine plant models."""

    def step(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        """Advance the plant state by one time step.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Time step duration [s].
        inputs : ModelInputs
            Control inputs applied over the interval ``[t, t + dt]``.

        Returns
        -------
        ModelOutputs
            Plant outputs at time ``t + dt``.
        """
        ...

    def reset(self) -> None:
        """Reset the plant to its initial internal state.

        Returns
        -------
        None
        """
        ...


class InputFn(Protocol):
    """Interface implemented by control-input generators."""

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute the plant inputs to apply at time `t`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Last known plant state, or ``None`` on the first call.

        Returns
        -------
        ModelInputs
            Inputs to apply to the plant at time `t`.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state of the input generator.

        Returns
        -------
        None
        """
        ...


def _get_git_hash() -> str:
    """Retrieve the current short git commit hash of the repository.

    Returns
    -------
    str
        The short commit hash, or ``"unknown"`` if it cannot be
        determined (e.g. not running inside a git repository).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _build_metadata(t_span: tuple[float, float], dt: float) -> dict:
    """Assemble run metadata for a simulation `Artefact`.

    Parameters
    ----------
    t_span : tuple[float, float]
        Simulation start and end times ``(t_start, t_end)`` [s].
    dt : float
        Fixed simulation time step [s].

    Returns
    -------
    dict
        Metadata with the git commit hash, an ISO 8601 UTC timestamp,
        and the simulation time span and step size.
    """
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "t_span": t_span,
        "dt": dt,
    }


def _stack_field(history: list[ModelState], attr: str, field_name: str) -> np.ndarray:
    """Stack one field's values across a state history into an array.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, in chronological order.
    attr : str
        Name of the ``ModelState`` attribute to read (``"inputs"`` or
        ``"outputs"``).
    field_name : str
        Name of the field within that attribute's dataclass.

    Returns
    -------
    np.ndarray
        1-D array of the field's values over time, shape ``(N,)``.
    """
    values = [getattr(getattr(state, attr), field_name) for state in history]
    return np.asarray(values, dtype=float)


def _history_to_artefact(history: list[ModelState], metadata: dict) -> Artefact:
    """Convert a recorded state history into a packed `Artefact`.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, in chronological order.
    metadata : dict
        Run metadata to attach to the returned `Artefact`.

    Returns
    -------
    Artefact
        Time vector plus input/output histories packed as arrays.
    """
    t = np.asarray([state.t for state in history], dtype=float)
    inputs = {
        f.name: _stack_field(history, "inputs", f.name) for f in fields(ModelInputs)
    }
    outputs = {
        f.name: _stack_field(history, "outputs", f.name) for f in fields(ModelOutputs)
    }
    return Artefact(t=t, inputs=inputs, outputs=outputs, metadata=metadata)


def run_simulation(
    plant: Plant,
    input_fn: InputFn,
    t_span: tuple[float, float],
    dt: float,
) -> Artefact:
    """Run a fixed-step simulation and pack the result into an `Artefact`.

    Parameters
    ----------
    plant : Plant
        Plant model to simulate.
    input_fn : InputFn
        Callable generating plant inputs at each time step.
    t_span : tuple[float, float]
        Simulation start and end times ``(t_start, t_end)`` [s].
    dt : float
        Fixed simulation time step [s].

    Returns
    -------
    Artefact
        Full recorded time history of inputs and outputs, with
        metadata including the git commit hash and an ISO timestamp.

    Raises
    ------
    ValueError
        If `dt` is not strictly positive or `t_span` is not increasing.
    """
    t_start, t_end = t_span
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    if t_end <= t_start:
        raise ValueError(f"t_span must be increasing, got {t_span}")

    plant.reset()
    input_fn.reset()

    history: list[ModelState] = []
    state: ModelState | None = None
    t = t_start
    while t < t_end:
        inputs = input_fn(t, state)
        outputs = plant.step(t, dt, inputs)
        state = ModelState(t=t, inputs=inputs, outputs=outputs)
        history.append(state)
        t += dt

    metadata = _build_metadata(t_span, dt)
    return _history_to_artefact(history, metadata)
