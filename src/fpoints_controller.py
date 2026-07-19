"""
PF3 Fixed-Points Open-Loop Controller.

Defines ``FPointsController``, an ``InputFn`` implementation for
open-loop trajectory replay. A ``TrajectorySet`` recorded or designed
ahead of time (guide vane opening, turbine speed, pump speed) is
interpolated at each query time `t` and returned as ``ModelInputs``.
No measurement of the plant state is used to compute the inputs: the
same time vector `t` always produces the same output, regardless of
how the plant actually behaved. This makes ``FPointsController``
suitable for scripted excitation experiments and for replaying
reference trajectories that a feedback controller is expected to
track.
"""

from __future__ import annotations

from src.data_types import ModelInputs, ModelState, TrajectorySet

DEFAULT_PUMP_SPEED = 0.0


class FPointsController:
    """Open-loop controller replaying a fixed set of reference trajectories.

    ``FPointsController`` implements the ``InputFn`` protocol by
    independently interpolating the guide vane opening (`y_T`),
    turbine speed (`N_T`), and pump speed (`N_P`) trajectories stored
    in a ``TrajectorySet`` at each requested simulation time. This is
    pure open-loop replay: the `state` argument received by
    `__call__` is ignored, so the generated inputs depend only on
    time and never react to the actual plant behavior.

    Parameters
    ----------
    trajectory_set : TrajectorySet
        Reference trajectories for `y_T`, `N_T`, and `N_P` to replay.
        If `trajectory_set.N_P` is ``None``, the pump speed input is
        held at `DEFAULT_PUMP_SPEED` for all times.
    """

    def __init__(self, trajectory_set: TrajectorySet) -> None:
        """Store the trajectory set to replay.

        Parameters
        ----------
        trajectory_set : TrajectorySet
            Reference trajectories for `y_T`, `N_T`, and `N_P`.

        Returns
        -------
        None
        """
        self._trajectory_set = trajectory_set

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute the open-loop plant inputs at time `t`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState or None
            Last known plant state. Ignored, since this controller is
            purely open-loop and does not use feedback.

        Returns
        -------
        ModelInputs
            Inputs interpolated from the stored trajectory set at
            time `t`.
        """
        return self._interpolate(t)

    def reset(self) -> None:
        """Reset internal state.

        No-op: this controller is stateless, since it only interpolates
        the stored trajectory set as a function of time.

        Returns
        -------
        None
        """

    def _interpolate(self, t: float) -> ModelInputs:
        """Interpolate all reference trajectories at time `t`.

        Parameters
        ----------
        t : float
            Query time [s].

        Returns
        -------
        ModelInputs
            Inputs built from the interpolated `y_T`, `N_T`, and
            `N_P` trajectory values at time `t`.
        """
        trajectories = self._trajectory_set
        return ModelInputs(
            y_T=trajectories.y_T(t),
            N_T=trajectories.N_T(t),
            N_P=self._interpolate_pump_speed(t),
        )

    def _interpolate_pump_speed(self, t: float) -> float:
        """Interpolate the pump speed trajectory at time `t`, if present.

        Parameters
        ----------
        t : float
            Query time [s].

        Returns
        -------
        float
            Interpolated pump speed [rpm], or `DEFAULT_PUMP_SPEED` if
            no pump speed trajectory is set.
        """
        n_p = self._trajectory_set.N_P
        return n_p(t) if n_p is not None else DEFAULT_PUMP_SPEED
