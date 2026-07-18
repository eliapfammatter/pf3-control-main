"""
PF3 Simulation Orchestrator.

Defines the ``Plant`` and ``InputFn`` protocols that decouple the plant
model from the signal that drives it, and the ``run_simulation`` function
that advances a ``Plant`` through a fixed-step simulation loop, recording
the full input/output time history into an ``Artefact``.

No global state is used: all shared state is threaded explicitly through
function arguments and the ``ModelState``/``Artefact`` dataclasses.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import numpy as np

from src.data_types import Artefact, ModelInputs, ModelOutputs, ModelState


@runtime_checkable
class Plant(Protocol):
    """Protocol for a plant model advanced one fixed time step at a time."""

    def step(self, t: float, dt: float, inputs: ModelInputs) -> ModelOutputs:
        """
        Advance the plant by one time step.

        Parameters
        ----------
        t : float
            Current simulation time [s], at the start of the step.
        dt : float
            Duration of the step [s].
        inputs : ModelInputs
            Control inputs applied over the step.

        Returns
        -------
        ModelOutputs
            Plant outputs resulting from the step.
        """
        ...

    def reset(self) -> None:
        """
        Reset the plant to its initial state.

        Returns
        -------
        None
        """
        ...


@runtime_checkable
class InputFn(Protocol):
    """Protocol for a callable that generates plant inputs over time."""

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """
        Compute the plant inputs to apply at time `t`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Most recently recorded plant state, or ``None`` if no step
            has been taken yet.

        Returns
        -------
        ModelInputs
            Inputs to apply to the plant at time `t`.
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state held by the input function.

        Returns
        -------
        None
        """
        ...


def _get_git_hash() -> str:
    """
    Get the current git commit hash of the repository.

    Returns
    -------
    str
        The current commit hash, or ``"unknown"`` if it cannot be
        determined (e.g. outside a git repository or git unavailable).
    """
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _build_metadata() -> dict:
    """
    Build run metadata containing the git commit hash and a timestamp.

    Returns
    -------
    dict
        Dictionary with keys ``"git_hash"`` and ``"timestamp"`` (ISO 8601,
        UTC).
    """
    return {
        "git_hash": _get_git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _stack_field(records: list, field_name: str) -> np.ndarray:
    """
    Stack a named field from a list of dataclass records into an array.

    Parameters
    ----------
    records : list
        Sequence of ``ModelInputs`` or ``ModelOutputs`` instances sharing
        the field `field_name`.
    field_name : str
        Name of the dataclass field to extract.

    Returns
    -------
    np.ndarray
        1-D array of the extracted field values, shape ``(N,)``.
    """
    return np.array([getattr(record, field_name) for record in records], dtype=float)


def _pack_artefact(
    times: list[float],
    inputs_history: list[ModelInputs],
    outputs_history: list[ModelOutputs],
) -> Artefact:
    """
    Pack recorded simulation history into an ``Artefact``.

    Parameters
    ----------
    times : list of float
        Recorded simulation times [s].
    inputs_history : list of ModelInputs
        Recorded inputs, one per time step, aligned with `times`.
    outputs_history : list of ModelOutputs
        Recorded outputs, one per time step, aligned with `times`.

    Returns
    -------
    Artefact
        Packed time history, with git hash and timestamp metadata.
    """
    input_fields = ModelInputs.__dataclass_fields__.keys()
    output_fields = ModelOutputs.__dataclass_fields__.keys()

    inputs = {name: _stack_field(inputs_history, name) for name in input_fields}
    outputs = {name: _stack_field(outputs_history, name) for name in output_fields}

    return Artefact(
        t=np.array(times, dtype=float),
        inputs=inputs,
        outputs=outputs,
        metadata=_build_metadata(),
    )


def run_simulation(
    plant: Plant,
    input_fn: InputFn,
    t_span: tuple[float, float],
    dt: float,
) -> Artefact:
    """
    Run a fixed-step simulation of `plant` driven by `input_fn`.

    At each time step, `input_fn` is called with the current time and the
    most recently recorded ``ModelState`` to produce ``ModelInputs``,
    which are then applied to `plant` via ``plant.step``. The resulting
    ``ModelState`` history is accumulated and packed into an ``Artefact``.

    Parameters
    ----------
    plant : Plant
        The plant model to simulate.
    input_fn : InputFn
        Callable producing plant inputs at each time step.
    t_span : tuple of float
        Start and end simulation time ``(t_start, t_end)`` [s], inclusive.
    dt : float
        Fixed simulation time step [s].

    Returns
    -------
    Artefact
        Full recorded time history of inputs, outputs, and run metadata.

    Raises
    ------
    ValueError
        If `dt` is not strictly positive, or if `t_span` is decreasing.
    """
    t_start, t_end = t_span
    if dt <= 0.0:
        raise ValueError(f"dt must be strictly positive, got {dt}")
    if t_end < t_start:
        raise ValueError(f"t_span must satisfy t_start <= t_end, got {t_span}")

    plant.reset()
    input_fn.reset()

    n_steps = int(round((t_end - t_start) / dt)) + 1
    times: list[float] = []
    inputs_history: list[ModelInputs] = []
    outputs_history: list[ModelOutputs] = []
    state: ModelState | None = None

    for i in range(n_steps):
        t = t_start + i * dt
        inputs = input_fn(t, state)
        outputs = plant.step(t, dt, inputs)
        state = ModelState(t=t, inputs=inputs, outputs=outputs)

        times.append(t)
        inputs_history.append(inputs)
        outputs_history.append(outputs)

    return _pack_artefact(times, inputs_history, outputs_history)
