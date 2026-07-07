"""Simulation orchestrator for the PF3 pump-turbine test rig.

Defines the `Plant` and `InputFn` protocols shared by all plant models and
input generators, together with `run_simulation`, the time-stepping loop
that drives a plant with a given input function and records the results
into an `Artefact`.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from data_types import Artefact, ModelInputs, ModelOutputs, ModelState


class Plant(Protocol):
    """Interface for a plant model driven by the simulation loop."""

    def step(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        """Advance the plant state by one time step.

        Parameters
        ----------
        t : float
            Current simulation time, in s.
        dt : float
            Time step, in s.
        inputs : ModelInputs
            Inputs applied over the step.

        Returns
        -------
        ModelOutputs
            Outputs computed after the step.
        """

    def reset(self) -> None:
        """Reset the plant to its initial state.

        Returns
        -------
        None
        """


class InputFn(Protocol):
    """Interface for a callable that generates plant inputs over time."""

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute the plant inputs at a given time.

        Parameters
        ----------
        t : float
            Current simulation time, in s.
        state : ModelState or None
            Most recent recorded state, or None on the first call.

        Returns
        -------
        ModelInputs
            Inputs to apply to the plant at time `t`.
        """

    def reset(self) -> None:
        """Reset any internal state of the input function.

        Returns
        -------
        None
        """


def _get_git_hash() -> str:
    """Return the current git commit hash, or "unknown" if unavailable.

    Returns
    -------
    str
        Short git commit hash of HEAD, or "unknown" if it cannot be
        determined (e.g. outside a git repository).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _build_metadata() -> dict[str, Any]:
    """Assemble run metadata (git hash and ISO timestamp).

    Returns
    -------
    dict[str, typing.Any]
        Metadata with keys "git_hash" and "timestamp".
    """
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _stack_field(history: list[ModelState], attr: str, field: str) -> np.ndarray:
    """Stack a single field of `inputs` or `outputs` across the history.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, one per time step.
    attr : str
        Name of the `ModelState` attribute to read ("inputs" or "outputs").
    field : str
        Name of the dataclass field to extract from `attr`.

    Returns
    -------
    numpy.ndarray
        Array of the field's values across the history.
    """
    return np.array([getattr(getattr(state, attr), field) for state in history])


def _history_to_artefact(history: list[ModelState]) -> Artefact:
    """Convert a list of recorded states into an `Artefact`.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, one per time step.

    Returns
    -------
    Artefact
        Packed simulation record with time, inputs, outputs and metadata.
    """
    t = np.array([state.t for state in history])
    input_fields = history[0].inputs.__dataclass_fields__
    output_fields = history[0].outputs.__dataclass_fields__
    inputs = {name: _stack_field(history, "inputs", name) for name in input_fields}
    outputs = {name: _stack_field(history, "outputs", name) for name in output_fields}
    return Artefact(t=t, inputs=inputs, outputs=outputs, metadata=_build_metadata())


def run_simulation(
    plant: Plant,
    input_fn: InputFn,
    t_span: tuple[float, float],
    dt: float,
) -> Artefact:
    """Run a closed simulation loop over `t_span` and record the results.

    Parameters
    ----------
    plant : Plant
        Plant model to simulate.
    input_fn : InputFn
        Callable producing the inputs applied at each time step.
    t_span : tuple[float, float]
        Start and end simulation time, in s.
    dt : float
        Time step, in s.

    Returns
    -------
    Artefact
        Recorded simulation trajectory, including metadata.

    Raises
    ------
    ValueError
        If `dt` is not strictly positive or `t_span` is not increasing.
    """
    t_start, t_end = t_span
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive")
    if t_end <= t_start:
        raise ValueError("t_span must be increasing")

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

    return _history_to_artefact(history)
