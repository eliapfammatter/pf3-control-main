from pathlib import Path

from src.controllers import Controller, ControllerState
from src.helpers import FPoints


class FPointsController(Controller):
    """Open-loop controller that follows a predefined trajectory.

    Reads pump speed from SIMSEN FPOINTS file and ignores state feedback.
    """

    def __init__(self, fpoints: FPoints):
        """
        Parameters
        ----------
        fpoints : FPoints
            FPOINTS trajectory reader for pump speed
        """
        self._fpoints = fpoints
        # Extract file name for cache key differentiation
        self._file_name = Path(fpoints.filepath).name

    def compute_pump_speed(self, time: float, state: ControllerState) -> float:
        """Return pump speed from trajectory (ignores state)."""
        return float(self._fpoints(time))

    def reset(self) -> None:
        """No-op for open-loop controller."""
        pass

    @property
    def name(self) -> str:
        return f"FPOINTS ({self._file_name})"
