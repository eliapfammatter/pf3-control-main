"""Gain-scheduled PI controller with IMC tuning.

IMC-PI tuning from Rivera et al. (1986), Table I Entry A.
"""

import numpy as np

from src.controllers import Controller, ControllerState
from src.pf3_static_model_b2 import PF3StaticModelB2


class PIController(Controller):
    """PI controller with gain scheduling and IMC tuning.

    From Rivera et al. (1986) Table I Entry A, for a first-order plant model:

        G(s) = K_P / (τ_P s + 1)

    the IMC design yields a PI controller with parameters:

        K_c = τ_P / (K_P × τ_c)
        τ_I = τ_P

    where τ_c is the closed-loop time constant (tuning parameter).
    """

    def __init__(
        self,
        H_ref: float,
        pf3_system: PF3StaticModelB2,
        dt: float = 0.1,
        tau_P: float = 0.491,
        tau_c: float = 10,
        u_nominal: float = -313.0,
        K_P_nominal: float = -0.032,
        tau_gain: float | None = 2,
        gain_scheduling: bool = True,
    ):
        self.H_ref = H_ref
        self.pf3 = pf3_system
        self.tau_P = tau_P
        self.tau_c = tau_c
        self.tau_I = tau_P
        self.dt = dt
        self.u_nominal = u_nominal
        self.K_P_nominal = K_P_nominal
        self.gain_scheduling = gain_scheduling

        # Gain filter (slow adaptation)
        self.tau_gain = tau_gain
        self._alpha_gain = dt / (self.tau_gain + dt) if self.tau_gain else None

        # State
        self._integral = 0.0
        self._K_P_filtered = K_P_nominal

    def reset(self) -> None:
        self._integral = 0.0
        self._K_P_filtered = self.K_P_nominal

    @property
    def name(self) -> str:
        if self.gain_scheduling:
            return f"PI dynamic gain (τ_c={self.tau_c}s)"
        return f"PI fixed gain"

    def _get_K_P(self, state: ControllerState) -> float:
        """Get plant gain K_P, low-pass filtered."""
        K_P_instant = None
        warning = None

        # Check operating point validity
        if not self.gain_scheduling:
            K_P_instant = None
        else:
            # Try to compute gain
            try:
                result = self.pf3.compute_G_u(
                    y_T=state.y_T,
                    N_T=state.N_T,
                    H_T=state.H_T,
                    N_P=state.N_P,
                    H_P1=state.H_P1,
                    H_P2=state.H_P2,
                )
                K_P_instant = result["G_u"]
            except Exception as e:
                warning = f"WARN: compute_G_u failed: {e}"

        # Fall back to nominal if invalid
        if K_P_instant is None:
            K_P_instant = self.K_P_nominal

        # Low-pass filter for slow adaptation
        if self._alpha_gain is not None:
            self._K_P_filtered = (
                self._alpha_gain * K_P_instant
                + (1 - self._alpha_gain) * self._K_P_filtered
            )
        else:
            self._K_P_filtered = K_P_instant

        if warning:
            print(
                f"{warning} | using nominal={self.K_P_nominal:.6f}, filtered={self._K_P_filtered:.6f}"
            )

        return self._K_P_filtered

    def compute_pump_speed(self, time: float, state: ControllerState) -> float:
        """Compute PI control output.

        IMC-PI law:
            u = K_c × e + (K_c / τ_I) × ∫e dt

        with K_c = τ_P / (K_P × τ_c), τ_I = τ_P
        """
        K_P = self._get_K_P(state)

        # IMC tuning: K_c = τ_P / (K_P × τ_c)
        K_c = self.tau_P / (K_P * self.tau_c)

        # Error
        error = self.H_ref - state.H_T

        # PI law
        self._integral += error * self.dt
        u = K_c * error + (K_c / self.tau_I) * self._integral

        # Add nominal offset
        u_total = self.u_nominal + u

        return u_total
