"""
Open-loop trajectory controller for the PF3 simulation.

Interpolates pre-recorded pump speed and guide-vane trajectories
(TrajectorySet) to produce ModelInputs at each simulation step.
No state feedback is used.
"""

from dataclasses import dataclass

from src.generated.few_shot.data_types import ModelInputs, ModelState, TrajectorySet


@dataclass
class FPointsController:
    """Open-loop InputFn: replays a recorded TrajectorySet.

    Parameters
    ----------
    traj : TrajectorySet
        The reference trajectories to replay.
    """

    traj: TrajectorySet

    def __call__(self, t: float, state: ModelState | None) -> ModelInputs:
        """Return interpolated inputs at time t.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        state : ModelState | None
            Ignored (open-loop controller).

        Returns
        -------
        ModelInputs
            Interpolated y_T, N_T, N_P values.
        """
        return ModelInputs(
            y_T=self.traj.y_T(t),
            N_T=self.traj.N_T(t),
            N_P=self.traj.N_P(t) if self.traj.N_P is not None else 0.0,
        )

    def reset(self) -> None:
        """No internal state to reset for open-loop controller."""
        pass
