"""
PF3 First-Order Model for PI Controller Design.

Implements the simplified first-order transfer function from Appendix B2:
    G(s) = K_P / (1 + τ_P·s)

State-space form:
    ẋ = -(1/τ_P)·x + (K_P/τ_P)·u
    y = x

where:
    x = H_T (turbine head state)
    u = N_P (pump speed input)
    y = H_T (output)
    K_P = static gain from B2 model
    τ_P = time constant (fitted from step response)
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .pf3_static_model_b2 import PF3StaticModelB2


@dataclass
class FirstOrderParams:
    """Parameters for the first-order model."""

    tau: float  # Time constant [s]
    K_P: float  # Static gain [m/rpm] (can be updated from B2 model)


@dataclass
class OperatingPoint:
    """Operating point for linearization."""

    y_T: float  # Turbine guide vane opening [-]
    N_T: float  # Turbine speed [rpm]
    H_T: float  # Turbine head [m]
    N_P: float  # Pump speed [rpm]
    H_P1: float  # Pump 1 head [m]
    H_P2: float  # Pump 2 head [m]


class PF3FirstOrderModel:
    """
    First-order model: G(s) = K_P / (1 + τ_P·s)

    This is a simplified model for PI controller design that captures
    the dominant dynamics of the pump-turbine system.

    Parameters
    ----------
    static_model : PF3StaticModelB2
        B2 static gain model for computing K_P
    tau : float
        Time constant [s] (fitted from step response)

    Attributes
    ----------
    static_model : PF3StaticModelB2
        Reference to B2 model for gain computation
    tau : float
        Time constant
    _K_P : float or None
        Cached static gain (updated on demand)
    """

    def __init__(self, static_model: PF3StaticModelB2, tau: float):
        self.static_model = static_model
        self.tau = tau
        self._K_P: Optional[float] = None
        self._operating_point: Optional[OperatingPoint] = None

    def update_gain(self, operating_point: OperatingPoint) -> float:
        """
        Update the static gain K_P at the given operating point.

        Parameters
        ----------
        operating_point : OperatingPoint
            Current operating point

        Returns
        -------
        float
            Updated static gain K_P [m/rpm]
        """
        result = self.static_model.compute_G_u(
            y_T=operating_point.y_T,
            N_T=operating_point.N_T,
            H_T=operating_point.H_T,
            N_P=operating_point.N_P,
            H_P1=operating_point.H_P1,
            H_P2=operating_point.H_P2,
        )
        self._K_P = result["G_u"]
        self._operating_point = operating_point
        return self._K_P

    @property
    def K_P(self) -> float:
        """Current static gain. Raises if not yet computed."""
        if self._K_P is None:
            raise ValueError("K_P not computed. Call update_gain() first.")
        return self._K_P

    def dynamics(self, t: float, x: float, u: float) -> float:
        """
        Compute the state derivative.

        ẋ = -(1/τ)·x + (K_P/τ)·u

        Parameters
        ----------
        t : float
            Time [s] (unused, for interface compatibility)
        x : float
            Current state (H_T deviation from nominal)
        u : float
            Input (N_P deviation from nominal)

        Returns
        -------
        float
            State derivative dx/dt
        """
        return -x / self.tau + self.K_P / self.tau * u

    def simulate(
        self,
        t_span: tuple[float, float],
        x0: float,
        u_func: Callable[[float], float],
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
    ) -> dict:
        """
        Simulate the first-order model.

        Parameters
        ----------
        t_span : tuple
            (t_start, t_end) simulation time span [s]
        x0 : float
            Initial state (H_T at t=0)
        u_func : Callable
            Function u(t) returning pump speed at time t [rpm]
        t_eval : np.ndarray, optional
            Times at which to store solution
        method : str
            Integration method (default: "RK45")

        Returns
        -------
        dict
            Dictionary with:
            - 't': time array [s]
            - 'H_T': turbine head array [m]
            - 'N_P': pump speed array [rpm]
        """

        def rhs(t, x):
            u = u_func(t)
            return [self.dynamics(t, x[0], u)]

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        result = solve_ivp(
            rhs,
            t_span,
            [x0],
            method=method,
            t_eval=t_eval,
        )

        return {
            "t": result.t,
            "H_T": result.y[0],
            "N_P": np.array([u_func(ti) for ti in result.t]),
        }

    def step_response(
        self,
        t_span: tuple[float, float],
        H_T_0: float,
        N_P_0: float,
        delta_N_P: float,
        t_step: float = 0.0,
        t_eval: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute step response to a change in pump speed.

        Parameters
        ----------
        t_span : tuple
            (t_start, t_end) simulation time span [s]
        H_T_0 : float
            Initial turbine head [m]
        N_P_0 : float
            Initial pump speed [rpm]
        delta_N_P : float
            Step change in pump speed [rpm]
        t_step : float
            Time at which step occurs [s] (default: 0.0)
        t_eval : np.ndarray, optional
            Times at which to store solution

        Returns
        -------
        dict
            Dictionary with step response data
        """

        def u_func(t):
            if t >= t_step:
                return N_P_0 + delta_N_P
            return N_P_0

        result = self.simulate(t_span, H_T_0, u_func, t_eval)

        # Add theoretical steady-state
        H_T_ss = H_T_0 + self.K_P * delta_N_P
        result["H_T_ss"] = H_T_ss
        result["delta_N_P"] = delta_N_P
        result["t_step"] = t_step

        return result

    def predict_steady_state(self, H_T_0: float, delta_N_P: float) -> float:
        """
        Predict steady-state H_T after a pump speed change.

        Parameters
        ----------
        H_T_0 : float
            Initial turbine head [m]
        delta_N_P : float
            Change in pump speed [rpm]

        Returns
        -------
        float
            Predicted steady-state turbine head [m]
        """
        return H_T_0 + self.K_P * delta_N_P

    def time_to_63_percent(self) -> float:
        """
        Return time to reach 63.2% of final value (= τ by definition).

        Returns
        -------
        float
            Time constant τ [s]
        """
        return self.tau

    def settling_time(self, percent: float = 2.0) -> float:
        """
        Compute settling time to within specified percentage of final value.

        For first-order system: t_s = -τ * ln(percent/100)

        Parameters
        ----------
        percent : float
            Settling band percentage (default: 2%)

        Returns
        -------
        float
            Settling time [s]
        """
        return -self.tau * np.log(percent / 100.0)

    def print_summary(self) -> None:
        """Print model summary."""
        print("=" * 60)
        print("PF3 First-Order Model Summary")
        print("=" * 60)
        print(f"  Transfer function: G(s) = K_P / (1 + τ·s)")
        print(f"  Time constant τ = {self.tau:.4f} s")
        if self._K_P is not None:
            print(f"  Static gain K_P = {self._K_P:.6e} m/rpm")
            print(f"  Settling time (2%) = {self.settling_time(2.0):.2f} s")
        else:
            print("  Static gain K_P = (not computed, call update_gain())")
        print("=" * 60)
