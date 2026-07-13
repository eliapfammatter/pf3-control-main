"""Open-loop trajectory replay input function for the PF3 simulation software.

Defines ``FPointsController``, an ``InputFn`` implementation that replays a
pre-recorded ``TrajectorySet`` (as read from a SIMSEN FPOINTS file, for
example) by interpolating each reference signal at the requested simulation
time, independent of the plant's current state.
"""

from __future__ import annotations

from src.generated.zero_shot.data_types import ModelInputs, ModelState, TrajectorySet


class FPointsController:
    """Open-loop input function that replays a fixed ``TrajectorySet``.

    Implements the ``InputFn`` protocol. At each call, the guide vane
    opening, turbine speed, and pump speed are obtained by interpolating
    the trajectories recorded in the ``TrajectorySet`` supplied at
    construction time. Because the replay is open-loop, the ``state``
    argument passed to ``__call__`` is accepted for protocol compatibility
    but never used: the returned inputs depend only on time.
    """

    def __init__(self, trajectories: TrajectorySet) -> None:
        """
        Parameters
        ----------
        trajectories : TrajectorySet
            Reference trajectories for guide vane opening (`y_T`), turbine
            speed (`N_T`), and pump speed (`N_P`) to replay open-loop.
        """
        self._trajectories = trajectories

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Compute plant inputs at time `t` by interpolating trajectories.

        Parameters
        ----------
        t : float
            Current simulation time, in seconds [s].
        state : ModelState or None
            Most recently recorded state. Ignored, since replay is
            open-loop.

        Returns
        -------
        ModelInputs
            Inputs interpolated from the trajectory set at time `t`. The
            pump speed `N_P` is 0.0 if no `N_P` trajectory was provided.
        """
        n_p = self._trajectories.N_P(t) if self._trajectories.N_P is not None else 0.0
        return ModelInputs(
            y_T=self._trajectories.y_T(t),
            N_T=self._trajectories.N_T(t),
            N_P=n_p,
        )

    def reset(self) -> None:
        """No-op reset, since the replay carries no internal state.

        Returns
        -------
        None
        """