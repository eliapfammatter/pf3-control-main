"""
Reduced-Order Nonlinear Model for PF3 Hydraulic Network

This module provides a reduced-order model (ROM) suitable for MPC control.
The reduction is based on singular perturbation theory:
- Fast dynamics (distributed pipe acoustics, f > 2 Hz) → quasi-steady-state
- Slow dynamics (aggregate flows, f < 2 Hz) → kept as states

Eigenvalue Analysis Results (from analyze_model_order.py):
==========================================================
- Full model: 140 states (88 Q + 52 H)
- Slow modes (f < 2 Hz): 38 modes
- Fast modes (f >= 2 Hz): 102 modes
- Dominant poles: λ ≈ -1.65, -1.47 (τ ≈ 0.6-0.7s)

Reduction: 140 → 3 states = 47x reduction

Physical Topology (CLOSED LOOP):
================================

              ┌──────────────────────────────────────────────────────┐
              │                                                      │
              ↓                                                      │
           STANK                                              TURBINE+CONE+ELBOW
              │                                                      ↑
              │ L1-L4 (penstock)                                     │
              ↓                                                      │
           NODE1                                              L19-L18-...-L12
              │                                               (draft tube)
        ┌─────┴─────┐                                                ↑
        │           │                                                │
      PUMP1       PUMP2                                              │
        │           │                                                │
        └─────┬─────┘                                                │
              ↓                                                      │
           NODE2 ────────────────────────────────────────────────────┘

Flow path: STANK → penstock → NODE1 → pumps (parallel) → NODE2 → draft tube → turbine → STANK

Mass balance at junctions (no storage):
- NODE1: Q_pen = Q_p1 + Q_p2
- NODE2: Q_p1 + Q_p2 = Q_turb
- Therefore: Q_pen = Q_turb = Q_p1 + Q_p2 (closed loop constraint)

Reduced States (3-state model):
- Q_loop: Total loop flow [m³/s] (= Q_pen = Q_turb = Q_p1 + Q_p2)
- Q_p1: Flow through pump 1 [m³/s]
- Q_p2: Flow through pump 2 [m³/s]

Note: Q_p1 + Q_p2 = Q_loop is enforced, so only 2 of 3 are independent.
We track all 3 for numerical convenience and output extraction.

The tank level H_tank is assumed quasi-constant (very slow dynamics).
Junction heads H_NODE1, H_NODE2 are computed algebraically from KVL.

Inputs: u = [y_T, N_T, N_P]
Output: y = H_T (turbine head)
"""

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from src.characteristics import Characteristics


class PF3ReducedModel:
    """
    Reduced-order nonlinear model for PF3 hydraulic network (4-branch MNA formulation).

    State vector: x = [Q_T, Q_d]  (2 states)
        Q_T = Q_1 + Q_2  (total flow = turbine/penstock flow)
        Q_d = Q_1 - Q_2  (difference flow = pump imbalance)

    Derived:
        Q_1 = (Q_T + Q_d) / 2  (pump 1 flow)
        Q_2 = (Q_T - Q_d) / 2  (pump 2 flow)

    Inputs: u = [y_T, N_T, N_P]
    Output: y = H_T (turbine head)

    Network topology (4 branches, 3 nodes):
        Branch 1: Pump 1    (Node 1 → Node 2), H_src = +H_P1
        Branch 2: Pump 2    (Node 1 → Node 2), H_src = +H_P2
        Branch 3: Turbine   (Node 2 → Node 3), H_src = -H_T
        Branch 4: Penstock  (Node 3 → Node 1), H_src = 0

        Node 1: Low-pressure manifold (junction, C=0)
        Node 2: High-pressure manifold (junction, C=0)
        Node 3: Surge tank (reference, H_3 = 0)

    See docs/pf3-control-rom.md for full derivation.
    """

    def __init__(
        self,
        turbine_char: Characteristics,
        pump_char: Characteristics,
        data_dir: Path = None,
    ):
        """
        Initialize reduced model.

        Parameters
        ----------
        turbine_char : Characteristics
            Turbine characteristic model
        pump_char : Characteristics
            Pump characteristic model (used for both pumps)
        data_dir : Path, optional
            Directory with SIMSEN DAT files for parameter computation
        """
        from src.hydraulic_elements import DiscreteLoss, Pipe, PumpTurbine
        from src.pf3_lumped_network import CombinedPipeParams

        self.turbine = turbine_char
        self.pump = pump_char

        if data_dir is None:
            data_dir = Path("data/pf3/passive_elements")

        # =========================================================================
        # Load elements from DAT files (same as CasADi version)
        # =========================================================================

        def load_pipe(name):
            return Pipe.from_dat(data_dir / f"{name}.DAT")

        def load_dloss(name):
            return DiscreteLoss.from_dat(data_dir / f"{name}.DAT")

        # Branch 1: Pump 1 path (Node 1 → Node 2)
        L5, LP1 = load_pipe("L5"), load_pipe("LP1")
        E4, E5 = load_dloss("ELBOW4"), load_dloss("ELBOW5")

        # Branch 2: Pump 2 path (Node 1 → Node 2)
        L6, L7, LP2, L10, L11 = (
            load_pipe("L6"),
            load_pipe("L7"),
            load_pipe("LP2"),
            load_pipe("L10"),
            load_pipe("L11"),
        )
        E7, E8, E9, E10, E11 = (
            load_dloss("ELBOW7"),
            load_dloss("ELBOW8"),
            load_dloss("ELBOW9"),
            load_dloss("ELBOW10"),
            load_dloss("ELBOW11"),
        )

        # Branch 3: Turbine path (Node 2 → Node 3)
        L12 = load_pipe("L12")
        L13, L14, L15 = load_pipe("L13"), load_pipe("L14"), load_pipe("L15")
        L16, L17, L18 = load_pipe("L16"), load_pipe("L17"), load_pipe("L18")
        L19_1, L19_2 = load_pipe("L19_1"), load_pipe("L19_2")
        CONE, ELBOW_pipe = load_pipe("CONE"), load_pipe("ELBOW")
        E12, E13 = load_dloss("ELBOW12"), load_dloss("ELBOW13")
        E14, E15 = load_dloss("ELBOW14"), load_dloss("ELBOW15")

        # Branch 4: Penstock (Node 3 → Node 1)
        L1, L2, L3, L4 = (
            load_pipe("L1"),
            load_pipe("L2"),
            load_pipe("L3"),
            load_pipe("L4"),
        )
        E1, E2, E3 = load_dloss("ELBOW1"), load_dloss("ELBOW2"), load_dloss("ELBOW3")

        # Turbine element (for L_turb_machine)
        fturb = PumpTurbine.from_dat(data_dir / "FTURB1.DAT")
        self.L_turb_machine = fturb.hydraulic_L()

        # =========================================================================
        # Combined branch parameters (using CombinedPipeParams like CasADi)
        # =========================================================================

        self.branch1_params = CombinedPipeParams.from_elements(
            [L5, LP1], [E4, E5], Q0=LP1.Q0
        )
        self.branch2_params = CombinedPipeParams.from_elements(
            [L6, L7, LP2, L10, L11], [E7, E8, E9, E10, E11], Q0=LP2.Q0
        )
        self.branch3_params = CombinedPipeParams.from_elements(
            [L12, L13, L14, L15, L16, L17, L18, L19_1, L19_2, CONE, ELBOW_pipe],
            [E12, E13, E14, E15],
            Q0=L12.Q0,
        )
        self.branch4_params = CombinedPipeParams.from_elements(
            [L1, L2, L3, L4], [E1, E2, E3], Q0=L1.Q0
        )

        # =========================================================================
        # Branch inductances and resistances
        # =========================================================================

        self.L_1 = self.branch1_params.L_total
        self.L_2 = self.branch2_params.L_total
        self.L_3 = self.branch3_params.L_total + self.L_turb_machine
        self.L_4 = self.branch4_params.L_total

        self.R_1 = self.branch1_params.R_total
        self.R_2 = self.branch2_params.R_total
        self.R_3 = self.branch3_params.R_total
        self.R_4 = self.branch4_params.R_total

        # =========================================================================
        # Admittance parameters (see docs/pf3-control-rom.md)
        # =========================================================================

        self.alpha = 1.0 / self.L_1 + 1.0 / self.L_2  # Pump admittance sum
        self.beta = 1.0 / self.L_3  # Turbine admittance
        self.gamma = 1.0 / self.L_4  # Penstock admittance
        self.delta = 1.0 / self.L_1 - 1.0 / self.L_2  # Pump admittance difference

        # Determinant of admittance matrix
        self.det_M = self.alpha * (self.beta + self.gamma) + self.beta * self.gamma

        # Guard against singular admittance matrix
        if abs(self.det_M) < 1e-12:
            raise ValueError(
                f"Admittance matrix nearly singular: det(M) = {self.det_M:.2e}. "
                "Check branch inductances."
            )

        # State names
        self.state_names = ["Q_T", "Q_d"]
        self.n_states = 2

        # Input names
        self.input_names = ["y_T", "N_T", "N_P"]
        self.n_inputs = 3

    def ode_rhs(
        self,
        t: float,
        state: np.ndarray,
        inputs: Dict[str, float],
    ) -> np.ndarray:
        """
        Compute state derivatives using MNA-derived equations.

        State: [Q_T, Q_d]
            Q_T = Q_1 + Q_2 (total flow)
            Q_d = Q_1 - Q_2 (difference flow)

        Inputs: {y_T, N_T, N_P}

        Returns: [dQ_T/dt, dQ_d/dt]
        """
        # Unpack state
        Q_T, Q_d = state

        # Recover individual pump flows
        Q_1 = (Q_T + Q_d) / 2.0
        Q_2 = (Q_T - Q_d) / 2.0

        # Unpack inputs
        y_T = inputs.get("y_T", 0.47059)
        N_T = inputs.get("N_T", 369.3346)
        N_P = inputs.get("N_P", -313.2579)

        # =========================================================================
        # Machine heads
        # =========================================================================

        # Turbine: H_T (energy extracted from flow)
        H_T = self.turbine.compute_H(y_T, N_T, Q_T)

        # Pumps: H_P (energy added to flow)
        # Q is negated because pump reference has Q < 0 for normal operation
        H_P1 = self.pump.compute_H(1.0, N_P, -Q_1)
        H_P2 = self.pump.compute_H(1.0, N_P, -Q_2)

        # =========================================================================
        # Friction terms: f_i = R_i × |Q_i| × Q_i
        # =========================================================================

        f_1 = self.R_1 * abs(Q_1) * Q_1
        f_2 = self.R_2 * abs(Q_2) * Q_2
        f_3 = self.R_3 * abs(Q_T) * Q_T  # Q_3 = Q_T
        f_4 = self.R_4 * abs(Q_T) * Q_T  # Q_4 = Q_T

        # =========================================================================
        # Admittance equation RHS (see docs/pf3-control-rom.md)
        # =========================================================================

        b_1 = (f_1 - H_P1) / self.L_1 + (f_2 - H_P2) / self.L_2 - f_4 / self.L_4
        b_2 = -(f_1 - H_P1) / self.L_1 - (f_2 - H_P2) / self.L_2 + (f_3 + H_T) / self.L_3

        # Pressure difference: H_1 - H_2
        dH = (self.beta * b_1 - self.gamma * b_2) / self.det_M

        # =========================================================================
        # State equations
        # =========================================================================

        # Friction sums/differences
        Sigma_f = f_1 / self.L_1 + f_2 / self.L_2
        Delta_f = f_1 / self.L_1 - f_2 / self.L_2

        # Pump head sums/differences
        Sigma_H = H_P1 / self.L_1 + H_P2 / self.L_2
        Delta_H = H_P1 / self.L_1 - H_P2 / self.L_2

        # State derivatives (from MNA derivation)
        dQ_T = self.alpha * dH - Sigma_f + Sigma_H
        dQ_d = self.delta * dH - Delta_f + Delta_H

        return np.array([dQ_T, dQ_d])

    def compute_turbine_head(
        self,
        state: np.ndarray,
        inputs: Dict[str, float],
        dQ_dt: float = 0.0,
    ) -> float:
        """
        Compute turbine head H_T including dynamic term.

        H_total = H_char + L_turb_machine × dQ/dt

        This matches SIMSEN's FTURB-H output which includes the inertial term.

        Parameters
        ----------
        state : array
            State vector [Q_T, Q_d]
        inputs : dict
            Input dict with y_T, N_T
        dQ_dt : float
            Time derivative of turbine flow [m³/s²]

        Returns
        -------
        H_T : float
            Total turbine head [m]
        """
        Q_T = state[0]  # Q_T is the turbine flow

        y_T = inputs.get("y_T", 0.47059)
        N_T = inputs.get("N_T", 369.3346)

        H_char = self.turbine.compute_H(y_T, N_T, Q_T)
        return H_char + self.L_turb_machine * dQ_dt

    def compute_outputs(
        self,
        state: np.ndarray,
        inputs: Dict[str, float],
        dQ_dt: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute all outputs of interest.

        Parameters
        ----------
        state : array
            State vector [Q_T, Q_d]
        inputs : dict
            Input dict with y_T, N_T, N_P, etc.
        dQ_dt : float
            Time derivative of turbine flow [m³/s²] for H_total computation
        """
        Q_T, Q_d = state

        # Recover individual pump flows
        Q_1 = (Q_T + Q_d) / 2.0
        Q_2 = (Q_T - Q_d) / 2.0

        y_T = inputs.get("y_T", 0.47059)
        N_T = inputs.get("N_T", 369.3346)
        N_P = inputs.get("N_P", -313.2579)

        # Machine heads (H_T includes dynamic term)
        H_char = self.turbine.compute_H(y_T, N_T, Q_T)
        H_T = H_char + self.L_turb_machine * dQ_dt
        H_P1 = self.pump.compute_H(1.0, N_P, -Q_1)
        H_P2 = self.pump.compute_H(1.0, N_P, -Q_2)

        return {
            "H_T": H_T,
            "Q_T": Q_T,
            "Q_d": Q_d,
            "Q_1": Q_1,
            "Q_2": Q_2,
            "H_P1": H_P1,
            "H_P2": H_P2,
        }

    def compute_H_T_from_timeseries(
        self,
        t: np.ndarray,
        Q_T: np.ndarray,
        y_T_func,
        N_T_func,
    ) -> np.ndarray:
        """
        Compute turbine head H_T from flow time series.

        Includes dynamic term L × dQ/dt computed via numerical differentiation.

        Parameters
        ----------
        t : array
            Time points [s]
        Q_T : array
            Turbine flow at each time point [m³/s]
        y_T_func : callable
            Guide vane opening as function of time
        N_T_func : callable
            Turbine speed as function of time [rpm]

        Returns
        -------
        H_T : array
            Total turbine head [m] at each time point
        """
        n = len(t)
        H_T = np.zeros(n)

        # Compute dQ/dt via numerical differentiation
        dQ_dt = np.zeros(n)
        dt = np.diff(t)
        # Central differences for interior points
        dQ_dt[1:-1] = (Q_T[2:] - Q_T[:-2]) / (t[2:] - t[:-2])
        # Forward/backward difference for endpoints
        if n > 1:
            dQ_dt[0] = (Q_T[1] - Q_T[0]) / dt[0] if dt[0] > 0 else 0
            dQ_dt[-1] = (Q_T[-1] - Q_T[-2]) / dt[-1] if dt[-1] > 0 else 0

        # Compute H_T at each time point
        for i in range(n):
            y_T = y_T_func(t[i])
            N_T = N_T_func(t[i])
            H_char = self.turbine.compute_H(y_T, N_T, Q_T[i])
            H_T[i] = H_char + self.L_turb_machine * dQ_dt[i]

        return H_T

    def estimate_Q_from_H(
        self,
        H_P: float,
        N_P: float,
        Q_guess: float = 0.1,
    ) -> float:
        """
        Estimate pump flow Q from measured pump head H_P.

        Inverts the relationship: H_P = pump.compute_H(y=1.0, N=N_P, Q=-Q)

        Parameters
        ----------
        H_P : float
            Measured pump head [m]
        N_P : float
            Pump speed [rpm]
        Q_guess : float
            Initial guess for Q [m³/s]

        Returns
        -------
        float
            Estimated pump flow Q [m³/s] (network convention, positive)
        """
        from scipy.optimize import brentq

        def residual(Q):
            # Pump expects Q < 0 for normal operation
            H_computed = self.pump.compute_H(1.0, N_P, -Q)
            return H_computed - H_P

        # Use bounded search - Q is typically 0.05 to 0.15 m³/s
        try:
            Q_est = brentq(residual, 0.01, 0.30)
        except ValueError:
            # Fallback to fsolve if brentq fails
            Q_est, _, ier, _ = fsolve(residual, Q_guess, full_output=True)
            Q_est = float(Q_est)

        return Q_est

    def estimate_state_from_measurements(
        self,
        H_P1: float,
        H_P2: float,
        N_P: float,
    ) -> np.ndarray:
        """
        Estimate state [Q_T, Q_d] from measured pump heads.

        Parameters
        ----------
        H_P1 : float
            Measured head at pump 1 [m]
        H_P2 : float
            Measured head at pump 2 [m]
        N_P : float
            Pump speed [rpm]

        Returns
        -------
        np.ndarray
            Estimated state [Q_T, Q_d]
        """
        Q_1 = self.estimate_Q_from_H(H_P1, N_P, Q_guess=0.10)
        Q_2 = self.estimate_Q_from_H(H_P2, N_P, Q_guess=0.09)
        Q_T = Q_1 + Q_2
        Q_d = Q_1 - Q_2
        return np.array([Q_T, Q_d])

    def get_steady_state(
        self,
        y_T: float = 0.47059,
        N_T: float = 369.3346,
        N_P: float = -313.2579,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute steady-state for given inputs.

        At steady state: dQ_T/dt = 0, dQ_d/dt = 0

        Returns
        -------
        x_ss : np.ndarray
            Steady-state [Q_T, Q_d]
        """
        if x0 is None:
            # Use initial flows from DAT files
            Q0_1 = self.branch1_params.Q0
            Q0_2 = self.branch2_params.Q0
            Q_T_init = Q0_1 + Q0_2
            Q_d_init = Q0_1 - Q0_2
            x0 = np.array([Q_T_init, Q_d_init])

        inputs = {"y_T": y_T, "N_T": N_T, "N_P": N_P}

        def residual(x):
            return self.ode_rhs(0, x, inputs)

        x_ss, info, ier, msg = fsolve(residual, x0, full_output=True)

        if ier != 1:
            print(f"Warning: Steady-state solver did not converge: {msg}")

        return x_ss

    def get_initial_state(self) -> np.ndarray:
        """
        Get initial state from DAT file values.

        Returns [Q_T, Q_d] computed from branch initial flows.
        """
        Q0_1 = self.branch1_params.Q0
        Q0_2 = self.branch2_params.Q0
        Q_T = Q0_1 + Q0_2
        Q_d = Q0_1 - Q0_2
        return np.array([Q_T, Q_d])

    def simulate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray = None,
        t_eval: np.ndarray = None,
        y_T_func: Callable = None,
        N_T_func: Callable = None,
        N_P_func: Callable = None,
        **solve_ivp_kwargs,
    ):
        """
        Simulate the reduced model.

        Parameters match create_pf3_distributed() for compatibility.
        """
        if y0 is None:
            y0 = self.get_steady_state()

        y_T_func = y_T_func or (lambda t: 0.47059)
        N_T_func = N_T_func or (lambda t: 369.3346)
        N_P_func = N_P_func or (lambda t: -313.2579)

        def rhs(t, state):
            inputs = {
                "y_T": y_T_func(t),
                "N_T": N_T_func(t),
                "N_P": N_P_func(t),
                "y_P1": 1.0,
                "y_P2": 1.0,
            }
            return self.ode_rhs(t, state, inputs)

        solve_ivp_kwargs.setdefault("method", "RK45")

        return solve_ivp(rhs, t_span, y0, t_eval=t_eval, **solve_ivp_kwargs)

    def linearize(
        self,
        state: np.ndarray,
        inputs: Dict[str, float],
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the model at a given operating point.

        State: [Q_T, Q_d]
        Inputs: [y_T, N_T, N_P]
        Output: H_T

        Returns (A, B, C, D) where:
            dx/dt = A @ dx + B @ du
            dy = C @ dx + D @ du

        The D matrix captures direct feedthrough from inputs to output,
        which is important for H_T since it depends directly on y_T and N_T
        through the turbine characteristic (not just through the states).
        """
        n_x = self.n_states
        n_u = self.n_inputs

        # A matrix: d(f)/d(x)
        A = np.zeros((n_x, n_x))
        f0 = self.ode_rhs(0, state, inputs)

        for j in range(n_x):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = self.ode_rhs(0, state_plus, inputs)
            A[:, j] = (f_plus - f0) / eps

        # B matrix: d(f)/d(u)
        B = np.zeros((n_x, n_u))
        input_keys = ["y_T", "N_T", "N_P"]

        for j, key in enumerate(input_keys):
            inputs_plus = inputs.copy()
            inputs_plus[key] = inputs[key] + eps
            f_plus = self.ode_rhs(0, state, inputs_plus)
            B[:, j] = (f_plus - f0) / eps

        # C matrix: d(H_T)/d(x)
        # H_T only depends on Q_T (first state), not Q_d
        C = np.zeros((1, n_x))
        y0 = self.compute_turbine_head(state, inputs)

        for j in range(n_x):
            state_plus = state.copy()
            state_plus[j] += eps
            y_plus = self.compute_turbine_head(state_plus, inputs)
            C[0, j] = (y_plus - y0) / eps

        # D matrix: d(H_T)/d(u) - direct feedthrough
        # H_T depends directly on y_T and N_T through turbine characteristic
        D = np.zeros((1, n_u))
        for j, key in enumerate(input_keys):
            inputs_plus = inputs.copy()
            inputs_plus[key] = inputs[key] + eps
            y_plus = self.compute_turbine_head(state, inputs_plus)
            D[0, j] = (y_plus - y0) / eps

        return A, B, C, D

    def print_summary(self):
        """Print model summary."""
        print("=" * 60)
        print("PF3 Reduced-Order Model (4-branch MNA formulation)")
        print("=" * 60)

        print(f"\nStates ({self.n_states}):")
        for i, name in enumerate(self.state_names):
            print(f"  x[{i}]: {name}")
        print(f"  Derived: Q_1 = (Q_T + Q_d)/2, Q_2 = (Q_T - Q_d)/2")

        print(f"\nInputs ({self.n_inputs}):")
        for i, name in enumerate(self.input_names):
            print(f"  u[{i}]: {name}")

        print(f"\nBranch Parameters:")
        print(f"  Branch 1 (Pump 1): L_1 = {self.L_1:.4f} s²/m², R_1 = {self.R_1:.4f} s²/m⁵")
        print(f"  Branch 2 (Pump 2): L_2 = {self.L_2:.4f} s²/m², R_2 = {self.R_2:.4f} s²/m⁵")
        print(f"  Branch 3 (Turbine): L_3 = {self.L_3:.4f} s²/m², R_3 = {self.R_3:.4f} s²/m⁵")
        print(f"  Branch 4 (Penstock): L_4 = {self.L_4:.4f} s²/m², R_4 = {self.R_4:.4f} s²/m⁵")
        print(f"  Turbine machine L: {self.L_turb_machine:.4f} s²/m²")

        print(f"\nAdmittance Parameters:")
        print(f"  α = {self.alpha:.6f} (pump sum)")
        print(f"  β = {self.beta:.6f} (turbine)")
        print(f"  γ = {self.gamma:.6f} (penstock)")
        print(f"  δ = {self.delta:.6f} (pump diff)")
        print(f"  det(M) = {self.det_M:.6f}")

        print(f"\nInitial Flows (from DAT):")
        print(f"  Q0_1 = {self.branch1_params.Q0:.6f} m³/s")
        print(f"  Q0_2 = {self.branch2_params.Q0:.6f} m³/s")

        print(f"\nReduction factor: 140 → {self.n_states} = {140//self.n_states}x")
        print("=" * 60)


# =============================================================================
# Factory function
# =============================================================================


def build_pf3_reduced_model(
    turbine: Characteristics = None,
    pump: Characteristics = None,
    data_dir: Path = Path("data/pf3/passive_elements"),
) -> PF3ReducedModel:
    """
    Create a reduced-order model for PF3.

    Uses 4-branch MNA formulation with states [Q_T, Q_d].

    Parameters
    ----------
    turbine : Characteristics, optional
        Turbine characteristic model. If None, loaded from default path.
    pump : Characteristics, optional
        Pump characteristic model. If None, loaded from default path.
    data_dir : Path
        Directory with SIMSEN DAT files

    Returns
    -------
    PF3ReducedModel
        The reduced-order model
    """
    if turbine is None:
        turbine = Characteristics(
            d_ref=0.3477,
            h_n=5.0,
            q_n=0.1959,
            t_n=224.057,
            n_n=369.3346,
            char_file=Path("data/pf3/missing_files/STORE_4_quadrant_characteristic.txt"),
            interp_method="structured",
        )

    if pump is None:
        pump = Characteristics(
            d_ref=0.535,
            h_n=50.0,
            q_n=1.30,
            t_n=1280.3,
            n_n=790.0,
            char_file=Path("data/pf3/missing_files/PF3_FP_D_535_ext.txt"),
            interp_method="structured",
        )

    return PF3ReducedModel(
        turbine_char=turbine,
        pump_char=pump,
        data_dir=data_dir,
    )


# =============================================================================
