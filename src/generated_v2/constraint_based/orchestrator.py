"""
PF3 Simulation Orchestrator.

Defines the ``Plant`` and ``InputFn`` protocols and the ``run_simulation``
entry point that drives a fixed-step, open- or closed-loop simulation loop:
at each time step an ``InputFn`` produces ``ModelInputs`` from the current
time and ``ModelState``, the ``Plant`` advances by ``dt`` and returns
``ModelOutputs``, and the resulting ``ModelState`` history is packed into
an ``Artefact`` for downstream analysis and storage.

All shared state is passed explicitly through dataclass arguments; no
global state is used.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import numpy as np

from src.data_types import Artefact, ModelInputs, ModelOutputs, ModelState

_GIT_HASH_UNKNOWN = "unknown"
_GIT_HASH_LENGTH = ["rev-parse", "--short", "HEAD"]


@runtime_checkable
class Plant(Protocol):
    """Protocol for simulated or physical plants driven by ``run_simulation``."""

    def step(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        """Advance the plant by one time step.

        Parameters
        ----------
        t : float
            Current simulation time [s], before the step is taken.
        dt : float
            Step size [s].
        inputs : ModelInputs
            Inputs applied over the interval ``[t, t + dt)``.

        Returns
        -------
        ModelOutputs
            Plant outputs at time ``t + dt``.
        """
        ...

    def reset(self) -> None:
        """Reset the plant to its initial state.

        Returns
        -------
        None
        """
        ...


@runtime_checkable
class InputFn(Protocol):
    """Protocol for callables that produce ``ModelInputs`` at each time step."""

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute the inputs to apply at time `t`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Most recent plant state, or ``None`` on the first call.

        Returns
        -------
        ModelInputs
            Inputs to apply at time `t`.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state held by the input function.

        Returns
        -------
        None
        """
        ...


def _git_hash() -> str:
    """Return the current git commit hash, or "unknown" if unavailable.

    Returns
    -------
    str
        Short git commit hash of ``HEAD``, or ``"unknown"`` if the hash
        cannot be determined (e.g. not running inside a git repository).
    """
    try:
        result = subprocess.run(
            ["git", *_GIT_HASH_LENGTH],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return _GIT_HASH_UNKNOWN


def _time_grid(*, t_span: tuple[float, float], dt: float) -> np.ndarray:
    """Build the fixed-step time grid spanned by a simulation run.

    Parameters
    ----------
    t_span : tuple of float
        Simulation start and end times ``(t0, t1)`` [s].
    dt : float
        Step size [s].

    Returns
    -------
    np.ndarray
        Time samples covering ``[t0, t1]`` inclusive, spaced by `dt`.

    Raises
    ------
    ValueError
        If `dt` is not strictly positive or `t_span` is not increasing.
    """
    t0, t1 = t_span
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    if t1 <= t0:
        raise ValueError(f"t_span must be increasing, got {t_span}")
    n_steps = int(round((t1 - t0) / dt))
    return t0 + dt * np.arange(n_steps + 1)


def _stack_field(*, history: list[ModelState], attr: str, field_name: str) -> np.ndarray:
    """Stack one named field of a ``ModelState`` sub-record across history.

    Parameters
    ----------
    history : list of ModelState
        Time-ordered simulation states, one per time step.
    attr : str
        Name of the ``ModelState`` attribute holding the sub-record
        (``"inputs"`` or ``"outputs"``).
    field_name : str
        Name of the scalar field to extract from the sub-record.

    Returns
    -------
    np.ndarray
        The extracted field values, one per entry in `history`.
    """
    return np.array(
        [getattr(getattr(state, attr), field_name) for state in history],
        dtype=float,
    )


def _history_to_artefact(*, history: list[ModelState], metadata: dict) -> Artefact:
    """Pack a list of ``ModelState`` samples into an ``Artefact``.

    Parameters
    ----------
    history : list of ModelState
        Time-ordered simulation states, one per time step.
    metadata : dict
        Run metadata to attach to the artefact.

    Returns
    -------
    Artefact
        Time, input, and output histories packed as arrays, together with
        `metadata`.
    """
    t = np.array([state.t for state in history], dtype=float)

    inputs = {
        name: _stack_field(history=history, attr="inputs", field_name=name)
        for name in ModelInputs.__dataclass_fields__
    }
    outputs = {
        name: _stack_field(history=history, attr="outputs", field_name=name)
        for name in ModelOutputs.__dataclass_fields__
    }

    return Artefact(t=t, inputs=inputs, outputs=outputs, metadata=metadata)


def _run_metadata(*, t_span: tuple[float, float], dt: float) -> dict:
    """Build the run metadata attached to a simulation's ``Artefact``.

    Parameters
    ----------
    t_span : tuple of float
        Simulation start and end times ``(t0, t1)`` [s].
    dt : float
        Step size [s].

    Returns
    -------
    dict
        Metadata containing the git commit hash, an ISO-8601 UTC
        timestamp, and the run's `dt` and `t_span`.
    """
    return {
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dt": dt,
        "t_span": t_span,
    }


def _simulate(*, plant: Plant, input_fn: InputFn, times: np.ndarray, dt: float) -> list[ModelState]:
    """Advance `plant` over `times`, driven by `input_fn`.

    Parameters
    ----------
    plant : Plant
        The plant (simulated or physical) to simulate.
    input_fn : InputFn
        Callable producing ``ModelInputs`` at each time step.
    times : np.ndarray
        Fixed-step time grid to simulate over.
    dt : float
        Step size [s] passed to `plant.step` at each iteration.

    Returns
    -------
    list of ModelState
        Time-ordered simulation states, one per entry in `times`.
    """
    history: list[ModelState] = []
    state: ModelState | None = None

    for t in times:
        inputs = input_fn(float(t), state)
        outputs = plant.step(float(t), dt, inputs)
        state = ModelState(t=float(t), inputs=inputs, outputs=outputs)
        history.append(state)

    return history


def run_simulation(
    plant: Plant,
    input_fn: InputFn,
    t_span: tuple[float, float],
    dt: float,
) -> Artefact:
    """Run a fixed-step simulation of `plant` driven by `input_fn`.

    Resets `plant` and `input_fn`, then repeatedly computes inputs via
    `input_fn` and advances `plant` by `dt` over ``t_span``, accumulating
    the resulting ``ModelState`` history into an ``Artefact``.

    Parameters
    ----------
    plant : Plant
        The plant (simulated or physical) to simulate.
    input_fn : InputFn
        Callable producing ``ModelInputs`` at each time step.
    t_span : tuple of float
        Simulation start and end times ``(t0, t1)`` [s].
    dt : float
        Step size [s].

    Returns
    -------
    Artefact
        Full time history of inputs and outputs, with metadata containing
        the git commit hash and an ISO-8601 run timestamp.
    """
    plant.reset()
    input_fn.reset()

    times = _time_grid(t_span=t_span, dt=dt)
    history = _simulate(plant=plant, input_fn=input_fn, times=times, dt=dt)
    metadata = _run_metadata(t_span=t_span, dt=dt)

    return _history_to_artefact(history=history, metadata=metadata)
