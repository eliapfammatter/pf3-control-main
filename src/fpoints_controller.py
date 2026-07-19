"""
PF3 Fixed-Points Open-Loop Controller.

Defines ``FPointsController``, an ``InputFn`` implementation that
replays a pre-recorded ``TrajectorySet`` open-loop: at each query time
`t`, the guide vane opening, turbine speed, and pump speed references
are independently interpolated from their respective trajectories and
returned as ``ModelInputs``. No feedback from the plant state is used.
"""

from __future__ import annotations

from src.data_types import ModelInputs, ModelState, TrajectorySet  # pylint: disable=import-error


class FPointsController:
    """Open-loop controller that replays a fixed set of trajectories.

    ``FPointsController`` implements the ``InputFn`` protocol by
    interpolating the guide vane opening (`y_T`), turbine speed
    (`N_T`), and pump speed (`N_P`) reference trajectories stored in a
    ``TrajectorySet`` at each requested simulation time. This is a pure
    open-loop replay: the plant `state` argument passed to `__call__`
    is ignored, so the generated inputs depend only on time and never
    react to the actual plant behavior.

    Parameters
    ----------
    trajectory_set : TrajectorySet
        Reference trajectories for `y_T`, `N_T`, and `N_P` to replay.
        If `trajectory_set.N_P` is ``None``, the pump speed input is
        held at ``0.0`` for all times.
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
        trajectories = self._trajectory_set
        n_p = trajectories.N_P(t) if trajectories.N_P is not None else 0.0
        return ModelInputs(
            y_T=trajectories.y_T(t),
            N_T=trajectories.N_T(t),
            N_P=n_p,
        )

    def reset(self) -> None:
        """Reset internal state.

        No-op: this controller is stateless, since it only interpolates
        the stored trajectory set as a function of time.

        Returns
        -------
        None
        """
