"""
MPC Network Adapter.

Generic adapter that wraps a HydraulicNetwork to implement the MPCCompatibleNetwork
protocol required by MPCIncidenceNetwork.

This separates topology knowledge (factory functions) from control interface
(this adapter), so any HydraulicNetwork can be used for MPC.

Usage:
    from src.pf3_lumped_network import build_pf3_lumped_mpc

    # Convenience function wires everything together:
    adapter = build_pf3_lumped_mpc(interp_method="structured")
"""

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

from src.characteristics import Characteristics
from src.incidence_network import HydraulicNetwork


class MPCNetworkAdapter:
    """
    Adapts any HydraulicNetwork to the MPCCompatibleNetwork interface.

    Provides:
        n_states, n_inputs, turbine
        ode_rhs(t, state, inputs)
        linearize(state, inputs, eps)
        compute_turbine_head(state, inputs, dQ_dt)
        get_steady_state(y_T, N_T, N_P, x0)
        get_initial_state()

    Parameters
    ----------
    network : HydraulicNetwork
        Built network (from a factory function).
    turbine_char : PumpTurbine
        Turbine characteristic for compute_H and turbine inversion.
    turbine_Q_name : str
        Branch name for turbine flow in the network state vector.
        Used to look up Q_T index via network._Q_indices.
    turbine_L : float
        Turbine hydraulic inductance [s²/m²] for the dynamic term
        H_total = H_char + L × dQ/dt. Set to 0.0 for static H only.
    input_keys : tuple
        MPC input keys in order, defining columns of B and D matrices.
    input_defaults : dict, optional
        Extra keys added to inputs before passing to network.ode_rhs().
        E.g., {"y_P1": 1.0, "y_P2": 1.0} for fixed pump guide vanes.
    """

    def __init__(
        self,
        network: HydraulicNetwork,
        turbine_char: Characteristics,
        turbine_Q_name: str,
        turbine_L: float = 0.0,
        input_keys: Tuple[str, ...] = ("y_T", "N_T", "N_P"),
        input_defaults: Optional[Dict[str, float]] = None,
        pump_char: Optional[Characteristics] = None,
    ):
        self.network = network
        self.turbine_char = turbine_char
        self.pump_char = pump_char
        self._turbine_Q_name = turbine_Q_name
        self._turbine_L = turbine_L
        self._input_keys = input_keys
        self._input_defaults = input_defaults or {}

        # Look up turbine flow index in state vector
        Q_indices = network._Q_indices
        if turbine_Q_name not in Q_indices:
            available = list(Q_indices.keys())
            raise KeyError(
                f"Branch '{turbine_Q_name}' not found in network. "
                f"Available Q branches: {available}"
            )
        self._turb_Q_idx = Q_indices[turbine_Q_name]

        # Cache state size
        self._n_states = len(network.get_initial_state())

    # =========================================================================
    # MPCCompatibleNetwork protocol
    # =========================================================================

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_inputs(self) -> int:
        return len(self._input_keys)

    @property
    def turbine(self) -> Characteristics:
        return self.turbine_char

    def _expand_inputs(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Expand MPC inputs with defaults (e.g., add y_P1, y_P2)."""
        return {**self._input_defaults, **inputs}

    def ode_rhs(
        self,
        t: float,
        state: np.ndarray,
        inputs: Dict[str, float],
    ) -> np.ndarray:
        """Compute state derivatives."""
        return self.network.ode_rhs(t, state, self._expand_inputs(inputs))

    def compute_turbine_head(
        self,
        state: np.ndarray,
        inputs: Dict[str, float],
        dQT_dt: float,
    ) -> float:
        """
        Compute turbine head H_T from state and inputs.

        H_total = H_char(y_T, N_T, Q_T) + L × dQ/dt
        """
        Q_T = state[self._turb_Q_idx]
        y_T = inputs["y_T"]
        N_T = inputs["N_T"]
        H_char = self.turbine_char.compute_H(y_T, N_T, Q_T)
        return H_char + self._turbine_L * dQT_dt

    def get_initial_state(self) -> np.ndarray:
        """Get initial state from DAT files."""
        return self.network.get_initial_state()

    def get_steady_state(
        self,
        y_T: float = 0.47059,
        N_T: float = 369.3346,
        N_P: float = -313.2579,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """Find steady-state where ode_rhs = 0."""
        if x0 is None:
            x0 = self.get_initial_state()

        inputs = {"y_T": y_T, "N_T": N_T, "N_P": N_P}

        def residual(x):
            return self.ode_rhs(0, x, inputs)

        x_ss, info, ier, msg = fsolve(residual, x0, full_output=True)

        if ier != 1:
            warnings.warn(f"Steady-state solver: {msg}")

        return x_ss

    def linearize(
        self,
        state: np.ndarray,
        inputs: Dict[str, float],
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize at operating point via finite differences.

        Returns (A, B, C, D) where:
            dx/dt = A @ dx + B @ du
            dy    = C @ dx + D @ du
        """
        n_x = self._n_states
        n_u = len(self._input_keys)

        # A matrix: d(f)/d(x)
        f0 = self.ode_rhs(0, state, inputs)
        dQT_dt_0 = f0[self._turb_Q_idx]
        A = np.zeros((n_x, n_x))
        for i in range(n_x):
            x_pert = state.copy()
            x_pert[i] += eps
            A[:, i] = (self.ode_rhs(0, x_pert, inputs) - f0) / eps

        # B matrix: d(f)/d(u)
        B = np.zeros((n_x, n_u))
        for j, key in enumerate(self._input_keys):
            u_pert = inputs.copy()
            u_pert[key] = inputs.get(key, 0) + eps
            B[:, j] = (self.ode_rhs(0, state, u_pert) - f0) / eps

        # C matrix: d(y)/d(x) where y = H_T
        # IMPORTANT: Must recompute dQT_dt at each perturbation to capture dynamic term
        H_T_0 = self.compute_turbine_head(state, inputs, dQT_dt_0)
        C = np.zeros((1, n_x))
        for i in range(n_x):
            x_pert = state.copy()
            x_pert[i] += eps
            f_pert = self.ode_rhs(0, x_pert, inputs)
            dQT_dt_pert = f_pert[self._turb_Q_idx]
            H_T_pert = self.compute_turbine_head(x_pert, inputs, dQT_dt_pert)
            C[0, i] = (H_T_pert - H_T_0) / eps

        # D matrix: d(y)/d(u) — direct feedthrough
        # IMPORTANT: Must recompute dQT_dt for input perturbations too
        D = np.zeros((1, n_u))
        for j, key in enumerate(self._input_keys):
            u_pert = inputs.copy()
            u_pert[key] = inputs.get(key, 0) + eps
            f_pert = self.ode_rhs(0, state, u_pert)
            dQT_dt_pert = f_pert[self._turb_Q_idx]
            H_T_pert = self.compute_turbine_head(state, u_pert, dQT_dt_pert)
            D[0, j] = (H_T_pert - H_T_0) / eps

        return A, B, C, D

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def simulate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray = None,
        y_T_func: Callable = None,
        N_T_func: Callable = None,
        N_P_func: Callable = None,
        **kwargs,
    ):
        """
        Simulate network with time-varying inputs.

        Wraps HydraulicNetwork.simulate() with input mapping.
        """
        if y0 is None:
            y0 = self.get_initial_state()

        # Build external_inputs_func from individual input functions
        defaults = self._input_defaults.copy()

        def external_inputs_func(t):
            result = defaults.copy()
            if y_T_func is not None:
                result["y_T"] = y_T_func(t)
            if N_T_func is not None:
                result["N_T"] = N_T_func(t)
            if N_P_func is not None:
                result["N_P"] = N_P_func(t)
            return result

        return self.network.simulate(
            t_span=t_span,
            y0=y0,
            external_inputs_func=external_inputs_func,
            **kwargs,
        )

    def compute_H_T_from_timeseries(
        self,
        t: np.ndarray,
        Q_T: np.ndarray,
        y_T_func: Callable,
        N_T_func: Callable,
    ) -> np.ndarray:
        """
        Compute turbine head H_T from flow time series.

        Includes dynamic term L × dQ/dt via numerical differentiation.
        """
        n = len(t)
        H_T = np.zeros(n)

        # Numerical dQ/dt
        dQ_dt = np.zeros(n)
        dt = np.diff(t)
        dQ_dt[1:-1] = (Q_T[2:] - Q_T[:-2]) / (t[2:] - t[:-2])
        if n > 1:
            dQ_dt[0] = (Q_T[1] - Q_T[0]) / dt[0] if dt[0] > 0 else 0
            dQ_dt[-1] = (Q_T[-1] - Q_T[-2]) / dt[-1] if dt[-1] > 0 else 0

        for i in range(n):
            y_T = y_T_func(t[i])
            N_T = N_T_func(t[i])
            H_char = self.turbine_char.compute_H(y_T, N_T, Q_T[i])
            H_T[i] = H_char + self._turbine_L * dQ_dt[i]

        return H_T

    def extract_results(self, result) -> Dict[str, np.ndarray]:
        """Delegate to network.extract_results()."""
        return self.network.extract_results(result)

    def print_summary(self):
        """Print network summary."""
        self.network.print_summary()
