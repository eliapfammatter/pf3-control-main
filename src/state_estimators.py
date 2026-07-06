"""
State Estimators for MPC Controllers.

This module provides pluggable state estimation strategies for use with
MPC controllers. All estimators implement the StateEstimator protocol.

State estimators reconstruct the full network state from available measurements
(typically H_T from the turbine).

Available Estimators:
- TurbineInversionEstimator: Inverts turbine characteristic to get Q_T,
  then splits pump flows using steady-state ratio.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

from src.controllers import ControllerState


@runtime_checkable
class StateEstimator(Protocol):
    """Protocol for state estimators.

    State estimators reconstruct the full network state vector from
    available measurements (H_T, H_P1, H_P2, etc.).
    """

    def estimate(
        self, state: ControllerState, network: "MPCCompatibleNetwork", time: float = None
    ) -> np.ndarray:
        """
        Estimate full network state from measurements.

        Parameters
        ----------
        state : ControllerState
            Current measurements from FMU (H_T, H_P1, H_P2, y_T, N_T, N_P)
        network : MPCCompatibleNetwork
            The network model (for steady-state computation, characteristics, etc.)
        time : float, optional
            Current time [s] for computing numerical derivatives (if needed)

        Returns
        -------
        np.ndarray
            Estimated state vector matching network.n_states
        """
        ...

    def reset(self) -> None:
        """Reset estimator state (if stateful)."""
        ...


class TurbineInversionEstimator:
    """
    State estimator using turbine characteristic inversion.

    Estimates the network state by:
    1. Inverting the turbine characteristic to find Q_turbine from H_T measurement
    2. Splitting pump flows using steady-state ratio
    3. Assuming Q_penstock ≈ Q_turbine (mass balance)
    4. Using steady-state H_STANK (slow dynamics)

    This estimator is designed for the 5-state lumped network:
        State order: [Q_pen, Q_pump1, Q_pump2, Q_turb, H_tank]
        Indices:     [0,     1,       2,       3,      4     ]

    Limitations:
    - Assumes pump flow split ratio matches steady-state (not true during transients)
    - Assumes Q_pen = Q_turb (ignores tank dynamics)
    - H_STANK taken from steady-state (not measured)
    """

    def __init__(self, Q_bounds: tuple = (0.05, 0.4)):
        """
        Initialize estimator.

        Parameters
        ----------
        Q_bounds : tuple
            (Q_min, Q_max) bounds for turbine flow inversion [m³/s]
        """
        self.Q_bounds = Q_bounds

    def estimate(self, state: ControllerState, network, time: float = None) -> np.ndarray:
        """
        Estimate network state from measurements.

        Parameters
        ----------
        state : ControllerState
            Current measurements (H_T, y_T, N_T, N_P, etc.)
        network : MPCCompatibleNetwork
            Network model with turbine characteristic and steady-state solver
        time : float, optional
            Current time [s] (not used by this estimator)

        Returns
        -------
        np.ndarray
            Estimated 5-state vector [Q_pen, Q_p1, Q_p2, Q_turb, H_tank]
        """
        y_T = float(np.asarray(state.y_T).item())
        N_T = float(np.asarray(state.N_T).item())
        N_P = float(np.asarray(state.N_P).item())
        H_T_measured = float(state.H_T)

        try:
            # Get steady-state for flow ratios and fallback values
            x_ss = network.get_steady_state(y_T, N_T, N_P)

            # For 5-state lumped network:
            # State order: [Q_pen, Q_pump1, Q_pump2, Q_turb, H_tank]
            if len(x_ss) == 5:
                IDX_Q_PEN = 0
                IDX_Q_PUMP1 = 1
                IDX_Q_PUMP2 = 2
                IDX_Q_TURB = 3
                IDX_H_TANK = 4

                # Get pump flow split ratio from steady-state
                Q_p1_ss = x_ss[IDX_Q_PUMP1]
                Q_p2_ss = x_ss[IDX_Q_PUMP2]
                Q_total_ss = Q_p1_ss + Q_p2_ss
                Q_ratio = Q_p1_ss / Q_total_ss if Q_total_ss > 1e-6 else 0.5

                # Invert turbine characteristic to find Q_turb from H_T
                def h_error(Q_turb):
                    return network.turbine.compute_H(y_T, N_T, Q_turb) - H_T_measured

                try:
                    Q_turb_est = brentq(h_error, self.Q_bounds[0], self.Q_bounds[1])
                except ValueError:
                    # Fallback to steady-state if inversion fails
                    Q_turb_est = x_ss[IDX_Q_TURB]

                # Estimate pump flows using steady-state ratio
                Q_p1_est = Q_ratio * Q_turb_est
                Q_p2_est = (1 - Q_ratio) * Q_turb_est

                # Q_penstock ≈ Q_turbine (mass balance, ignoring tank)
                Q_pen_est = Q_turb_est

                # H_STANK from steady-state (slow dynamics, not measured)
                H_tank_est = x_ss[IDX_H_TANK]

                return np.array([Q_pen_est, Q_p1_est, Q_p2_est, Q_turb_est, H_tank_est])

            else:
                # Generic fallback: scale flows proportionally
                # Find turbine flow index (highest |∂H_T/∂x|)
                idx_turb = 0  # Default
                Q_turb_ss = abs(x_ss[idx_turb])

                # Invert to get Q_turb
                def h_error(Q_turb):
                    return network.turbine.compute_H(y_T, N_T, Q_turb) - H_T_measured

                try:
                    Q_turb_est = brentq(h_error, self.Q_bounds[0], self.Q_bounds[1])
                except ValueError:
                    Q_turb_est = Q_turb_ss

                # Scale all flows proportionally
                if Q_turb_ss > 1e-6:
                    scale = Q_turb_est / Q_turb_ss
                    x_est = x_ss.copy()
                    for i in range(len(x_est) - 1):  # Assume last state is not a flow
                        x_est[i] *= scale
                    return x_est

                return x_ss

        except Exception:
            # Ultimate fallback: return steady-state
            return network.get_steady_state(y_T, N_T, N_P)

    def reset(self) -> None:
        """Reset estimator (stateless, nothing to reset)."""
        pass


class OracleEstimator:
    """
    Oracle state estimator that ensures model consistency with FMU measurements.

    Estimates Q_T by inverting H_T from FMU, so that:
        model.compute_turbine_head(estimated_state) = H_T_measured

    This ensures the MPC model's first forward pass gives exactly the
    measured H_T, which is required for correct prediction trajectories.

    For pump flows, uses actual FMU values (Q_P1, Q_P2) since those
    don't affect H_T computation.

    For the 5-state lumped network:
        State order: [Q_pen, Q_pump1, Q_pump2, Q_turb, H_tank]
    """

    def __init__(self, Q_bounds: tuple = (0.05, 0.4)):
        """
        Initialize estimator.

        Parameters
        ----------
        Q_bounds : tuple
            (Q_min, Q_max) bounds for turbine flow inversion [m³/s]
        """
        self.Q_bounds = Q_bounds
        self._Q_T_prev = None
        self._time_prev = None
        self._dQT_dt = 0.0

    def estimate(self, state: ControllerState, network, time: float = None) -> np.ndarray:
        """
        Estimate state such that model(state) = H_T_measured.

        Inverts H_T to find Q_T, uses actual pump flows from FMU.
        Computes dQ_T/dt numerically from consecutive measurements.

        Parameters
        ----------
        state : ControllerState
            Current measurements including H_T, Q_P1, Q_P2, H_tank from FMU
        network : MPCCompatibleNetwork
            Network model (for turbine characteristic inversion)
        time : float, optional
            Current time [s] for computing numerical derivatives

        Returns
        -------
        np.ndarray
            State vector where compute_turbine_head(state) = H_T_measured
        """
        if state.H_tank is None:
            raise ValueError(
                "OracleEstimator requires H_tank in ControllerState. "
                "Make sure FMU STANK-H output is being read."
            )

        y_T = float(np.asarray(state.y_T).item())
        N_T = float(np.asarray(state.N_T).item())
        H_T_measured = float(state.H_T)

        # Invert H_T to find Q_T such that H_char(Q_T) = H_T_measured
        def h_error(Q_turb):
            return network.turbine.compute_H(y_T, N_T, Q_turb) - H_T_measured

        if state.Q_T is not None:
            # Oracle estimator: use the actual FMU turbine flow when available.
            Q_turb = float(state.Q_T)
        else:
            try:
                Q_turb = brentq(h_error, self.Q_bounds[0], self.Q_bounds[1])
            except ValueError:
                raise ValueError(
                    "H_T inversion failed and no FMU Q_T available for fallback"
                )

        # Compute dQ_T/dt numerically from consecutive measurements
        if self._Q_T_prev is not None and self._time_prev is not None and time is not None:
            dt = time - self._time_prev
            if dt > 0:
                self._dQT_dt = (Q_turb - self._Q_T_prev) / dt

        # Store for next iteration
        self._Q_T_prev = Q_turb
        self._time_prev = time

        # Pump flows: use actual FMU values (negated for network convention)
        if state.Q_P1 is not None and state.Q_P2 is not None:
            Q_p1 = -float(state.Q_P1)
            Q_p2 = -float(state.Q_P2)
        else:
            # Fallback: split Q_turb using typical ratio
            Q_p1 = 0.52 * Q_turb
            Q_p2 = 0.48 * Q_turb

        # Q_penstock from mass balance
        Q_pen = Q_p1 + Q_p2

        H_tank = float(state.H_tank)

        return np.array([Q_pen, Q_p1, Q_p2, Q_turb, H_tank])

    def get_dQT_dt(self) -> float:
        """Get the most recently computed dQ_T/dt."""
        return self._dQT_dt

    def reset(self) -> None:
        """Reset estimator state."""
        self._Q_T_prev = None
        self._time_prev = None
        self._dQT_dt = 0.0
