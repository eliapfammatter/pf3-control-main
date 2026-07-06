"""
Pump Speed Controllers for PF3 System.

This module provides a common interface for pump speed control strategies.
All controllers implement the `Controller` abstract base class and can be
used interchangeably in the FMI simulation loop.

Controllers:
- FPointsController: Open-loop trajectory following
- GainScheduledPIController: Closed-loop PI with gain scheduling
- DoMPCController: Model Predictive Control (see src/mpc_controller.py)

Example usage:
    # Open-loop (trajectory following):
    controller = FPointsController(FPoints("data/pf3/REGP"))

    # Closed-loop (PI control):
    controller = GainScheduledPIController(
        H_ref=5.0,
        pf3_system=pf3,
        tau_c=2.0,
        u_nominal=-313.0,
    )

    # do-mpc control:
    from src.mpc_controller import DoMPCController, DoMPCParams
    controller = DoMPCController(H_ref=5.0, pf3_system=pf3, ...)

    # All use the same interface:
    N_P = controller.compute_pump_speed(time, state)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControllerState:
    """State feedback for controller computation.

    Contains all measurements available from the FMI simulation.
    """

    H_T: float  # Turbine head [m]
    H_P1: float  # Pump 1 head [m]
    H_P2: float  # Pump 2 head [m]
    N_T: float  # Turbine speed [rpm]
    y_T: float  # Turbine guide vane opening [-]
    N_P: float  # Current pump speed [rpm] (for gain computation)
    Q_T: float  # Turbine flow [m³/s]
    Q_P1: float  # Pump 1 flow [m³/s] (FMU convention: negative)
    Q_P2: float  # Pump 2 flow [m³/s] (FMU convention: negative)
    H_tank: float  # Surge tank level [m]


class Controller(ABC):
    """Abstract base class for pump speed controllers.

    All controllers must implement `compute_pump_speed()` which takes
    the current time and state feedback and returns the pump speed command.
    """

    @abstractmethod
    def compute_pump_speed(self, time: float, state: ControllerState) -> float:
        """Compute pump speed command.

        Parameters
        ----------
        time : float
            Current simulation time [s]
        state : ControllerState
            Current state feedback from FMI simulation

        Returns
        -------
        float
            Pump speed command [rpm]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state (for closed-loop controllers)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller name for logging/plotting."""
        pass

    @property
    def predicted_trajectory(self) -> Optional[np.ndarray]:
        return None
