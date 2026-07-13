"""
Simulation orchestrator for the PF3 pump-turbine test rig.

Defines the ``Plant`` and ``InputFn`` protocols implemented by every plant
model and input-signal generator in this project, together with
``run_simulation``, the fixed-step time loop that drives a ``Plant`` with a
given ``InputFn`` and packs the recorded trajectory into an ``Artefact``.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from src.generated.few_shot.data_types import (
    Artefact,
    ModelInputs,
    ModelOutputs,
    ModelState,
)


class Plant(Protocol):
    """Interface implemented by every plant model driven by the simulator."""

    def step(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        """Advance the plant state by one time step.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Time step duration [s].
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
    """Interface implemented by every plant input-signal generator."""

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute the plant inputs to apply at a given time.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Most recently recorded state, or None on the first call.

        Returns
        -------
        ModelInputs
            Inputs to apply to the plant at time ``t``.
        """

    def reset(self) -> None:
        """Reset any internal state held by the input function.

        Returns
        -------
        None
        """


def _get_git_hash() -> str:
    """Return the current git commit hash of HEAD.

    Returns
    -------
    str
        Short git commit hash of HEAD, or ``"unknown"`` if it cannot be
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
    """Assemble run metadata (git commit hash and ISO-8601 timestamp).

    Returns
    -------
    dict[str, Any]
        Metadata with keys ``"git_hash"`` and ``"timestamp"``.
    """
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _validate_time_span(t_span: tuple[float, float], dt: float) -> None:
    """Validate the simulation time span and step size.

    Parameters
    ----------
    t_span : tuple[float, float]
        Start and end simulation time [s].
    dt : float
        Time step duration [s].

    Raises
    ------
    ValueError
        If ``dt`` is not strictly positive or ``t_span`` is not increasing.
    """
    t_start, t_end = t_span
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive")
    if t_end <= t_start:
        raise ValueError("t_span end must be strictly greater than its start")


def _step_count(t_span: tuple[float, float], dt: float) -> int:
    """Compute the number of simulation steps covering ``t_span``.

    Parameters
    ----------
    t_span : tuple[float, float]
        Start and end simulation time [s].
    dt : float
        Time step duration [s].

    Returns
    -------
    int
        Number of steps of size ``dt`` between ``t_span[0]`` and ``t_span[1]``.
    """
    t_start, t_end = t_span
    return max(1, round((t_end - t_start) / dt))


def _run_loop(
    plant: Plant, input_fn: InputFn, t_span: tuple[float, float], dt: float
) -> list[ModelState]:
    """Drive ``plant`` with ``input_fn`` over ``t_span`` and record states.

    Parameters
    ----------
    plant : Plant
        Plant model to simulate. Must already have been reset.
    input_fn : InputFn
        Callable producing the inputs applied at each time step. Must
        already have been reset.
    t_span : tuple[float, float]
        Start and end simulation time [s].
    dt : float
        Time step duration [s].

    Returns
    -------
    list[ModelState]
        One recorded state per simulated time step, in order.
    """
    t_start, _ = t_span
    history: list[ModelState] = []
    state: ModelState | None = None
    for i in range(_step_count(t_span, dt)):
        t = t_start + i * dt
        inputs = input_fn(t, state)
        outputs = plant.step(t, dt, inputs)
        state = ModelState(t=t, inputs=inputs, outputs=outputs)
        history.append(state)
    return history


def _stack_field(history: list[ModelState], attr: str, field_name: str) -> np.ndarray:
    """Stack a single dataclass field across the recorded state history.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, one per time step.
    attr : str
        Name of the ``ModelState`` attribute to read (``"inputs"`` or
        ``"outputs"``).
    field_name : str
        Name of the dataclass field to extract from ``attr``.

    Returns
    -------
    np.ndarray
        Array of the field's values across the history.
    """
    return np.array([getattr(getattr(s, attr), field_name) for s in history])


def _pack_signals(history: list[ModelState], attr: str) -> dict[str, np.ndarray]:
    """Pack every dataclass field of ``attr`` across history into arrays.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, one per time step.
    attr : str
        Name of the ``ModelState`` attribute to pack (``"inputs"`` or
        ``"outputs"``).

    Returns
    -------
    dict[str, np.ndarray]
        Recorded signals keyed by dataclass field name.
    """
    field_names = getattr(history[0], attr).__dataclass_fields__
    return {name: _stack_field(history, attr, name) for name in field_names}


def _history_to_artefact(history: list[ModelState]) -> Artefact:
    """Convert a list of recorded states into an ``Artefact``.

    Parameters
    ----------
    history : list[ModelState]
        Recorded simulation states, one per time step.

    Returns
    -------
    Artefact
        Packed simulation record with time, inputs, outputs and metadata.
    """
    t = np.array([s.t for s in history])
    return Artefact(
        t=t,
        inputs=_pack_signals(history, "inputs"),
        outputs=_pack_signals(history, "outputs"),
        metadata=_build_metadata(),
    )


def run_simulation(
    plant: Plant, input_fn: InputFn, t_span: tuple[float, float], dt: float
) -> Artefact:
    """Run a closed-loop simulation over ``t_span`` and record the results.

    Parameters
    ----------
    plant : Plant
        Plant model to simulate.
    input_fn : InputFn
        Callable producing the inputs applied at each time step.
    t_span : tuple[float, float]
        Start and end simulation time [s].
    dt : float
        Time step duration [s].

    Returns
    -------
    Artefact
        Recorded simulation trajectory, including run metadata.

    Raises
    ------
    ValueError
        If ``dt`` is not strictly positive or ``t_span`` is not increasing.
    """
    _validate_time_span(t_span, dt)
    plant.reset()
    input_fn.reset()
    history = _run_loop(plant, input_fn, t_span, dt)
    return _history_to_artefact(history)
