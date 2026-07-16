"""Open-loop trajectory replay input function for the PF3 simulation software.

Defines ``FPointsController``, an ``InputFn`` implementation that reproduces
a fixed "FPOINTS" excitation sequence by interpolating a pre-recorded
``TrajectorySet`` at each requested simulation time. The controller never
reads the plant state: guide vane opening, turbine speed, and pump speed are
functions of time alone, which is the defining property of open-loop
trajectory replay as used to drive PF3 test-rig simulations against a fixed,
reproducible reference sequence.
"""

from __future__ import annotations

# pylint: disable=import-error,relative-beyond-top-level
# This module is part of the `generated.chain_of_thoughts` package and
# shares its data structures with the sibling `data_types` module via a
# same-package relative import (single dot). When this file is linted in
# isolation (outside of the package tree), static resolution of that import
# is a known false positive; at runtime, within the package, it resolves
# correctly.
from .data_types import ModelInputs, ModelState, TrajectorySet

# Named constant (no magic numbers): pump speed used when the TrajectorySet
# carries no N_P trajectory.
DEFAULT_PUMP_SPEED = 0.0


class FPointsController:
    """Open-loop input function that replays a fixed ``TrajectorySet``.

    Implements the ``InputFn`` protocol by evaluating each reference
    trajectory (``y_T``, ``N_T``, ``N_P``) in the ``TrajectorySet`` at the
    requested time and packing the results into a ``ModelInputs`` instance.
    This is open-loop trajectory replay: the plant ``state`` argument is
    accepted for protocol compatibility but is always ignored, so the
    returned inputs depend on simulation time alone and never react to the
    plant's actual behaviour.
    """

    def __init__(self, trajectories: TrajectorySet) -> None:
        """
        Parameters
        ----------
        trajectories : TrajectorySet
            Reference trajectories to replay for guide vane opening
            (`y_T`), turbine speed (`N_T`) and, optionally, pump speed
            (`N_P`).
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
            Interpolated pump speed [rpm], or ``DEFAULT_PUMP_SPEED`` if the
            trajectory set defines no pump speed trajectory.
        """
        if self._trajectories.N_P is None:
            return DEFAULT_PUMP_SPEED
        return self._trajectories.N_P(t)
