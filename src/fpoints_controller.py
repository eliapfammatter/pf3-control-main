"""Open-loop trajectory replay input function for the PF3 simulator.

Defines ``FPointsController``, an ``InputFn`` implementation that replays a
pre-recorded ``TrajectorySet`` by interpolation, without ever reading the
plant state. This is the "FPOINTS" open-loop excitation used to reproduce
a fixed test sequence (guide vane opening, turbine speed, pump speed)
independently of the plant's response.
"""

from __future__ import annotations

# Field names below (y_T, N_T, N_P) use domain-standard turbine/pump
# notation mandated by the module contract rather than PEP 8 snake_case.
# pylint: disable=invalid-name

# pylint: disable=import-error,relative-beyond-top-level
# This module is part of the `generated.constraint_based` package and
# shares its data structures with the sibling `data_types` module via a
# same-package relative import (single dot). When this file is linted in
# isolation (outside of the package tree), static resolution of that
# import is a known false positive; at runtime, within the package, it
# resolves correctly.
from .data_types import ModelInputs, ModelState, TrajectorySet

# Named constants (no magic numbers)
DEFAULT_PUMP_SPEED = 0.0


class FPointsController:
    """Open-loop input function that replays a fixed ``TrajectorySet``.

    Implements the ``InputFn`` protocol by evaluating each trajectory in
    the ``TrajectorySet`` at the requested time and packing the results
    into a ``ModelInputs`` instance. The plant ``state`` argument is
    accepted for protocol compatibility but is always ignored: this
    controller performs pure open-loop trajectory replay and never
    reacts to the simulated system's behaviour.
    """

    def __init__(self, trajectories: TrajectorySet) -> None:
        """
        Parameters
        ----------
        trajectories : TrajectorySet
            Reference trajectories to replay for ``y_T``, ``N_T`` and,
            optionally, ``N_P``.
        """
        self._trajectories = trajectories

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Return the plant inputs at time ``t`` by trajectory replay.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Most recently recorded plant state. Ignored: this controller
            is open-loop.

        Returns
        -------
        ModelInputs
            Inputs interpolated from the trajectory set at time ``t``.
        """
        del state
        return ModelInputs(
            y_T=self._trajectories.y_T(t),
            N_T=self._trajectories.N_T(t),
            N_P=self._interpolate_pump_speed(t),
        )

    def reset(self) -> None:
        """No-op reset: this controller holds no mutable internal state.

        Returns
        -------
        None
        """

    def _interpolate_pump_speed(self, t: float) -> float:
        """Evaluate the pump speed trajectory at time ``t``, if present.

        Parameters
        ----------
        t : float
            Query time [s].

        Returns
        -------
        float
            Interpolated pump speed [rpm], or ``DEFAULT_PUMP_SPEED`` if
            the trajectory set defines no pump speed trajectory.
        """
        if self._trajectories.N_P is None:
            return DEFAULT_PUMP_SPEED
        return self._trajectories.N_P(t)