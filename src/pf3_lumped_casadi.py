"""
PF3 Control Reduced-Order Model (CasADi implementation)

2-state nonlinear ROM for MPC control of turbine head.
See pf3-control-rom.md for derivation.

States: x = [Q_T, Q_d]
    Q_T = Q_1 + Q_2  (total flow = turbine/penstock flow)
    Q_d = Q_1 - Q_2  (difference flow = pump imbalance)

Derived:
    Q_1 = (Q_T + Q_d) / 2  (pump 1 flow)
    Q_2 = (Q_T - Q_d) / 2  (pump 2 flow)

Inputs: u = [y_T, N_T, N_P]
    y_T: turbine guide vane opening [-]
    N_T: turbine rotational speed [rpm]
    N_P: pump rotational speed [rpm]

Output: H_T (turbine head)

Network topology (4 branches, 3 nodes):
    Branch 1: Pump 1    (Node 1 → Node 2), H_src = +H_P1  (pump adds head)
    Branch 2: Pump 2    (Node 1 → Node 2), H_src = +H_P2  (pump adds head)
    Branch 3: Turbine   (Node 2 → Node 3), H_src = -H_T   (turbine extracts head)
    Branch 4: Penstock  (Node 3 → Node 1), H_src = 0      (passive pipe)

    Node 1: Low-pressure manifold (junction, C=0)
    Node 2: High-pressure manifold (junction, C=0)
    Node 3: Surge tank (reference, H_3 = 0)
"""

from pathlib import Path

import casadi as ca
import numpy as np

from .hydraulic_elements import DiscreteLoss, Pipe, Tank
from .hydraulic_elements.pump_turbine import PumpTurbine
from .pf3_lumped_network import CombinedPipeParams
from .triinterp import TriInterpCasadi


class FrancisTurbineCasadi:
    """CasADi-compatible turbomachine characteristic interpolation."""

    def __init__(self, turbine: PumpTurbine):
        self.turbine = turbine
        self.interp_wh = TriInterpCasadi(turbine.characteristic.interp_wh)
        self.interp_wb = TriInterpCasadi(turbine.characteristic.interp_wb)

    @classmethod
    def from_dat(cls, file_path: Path, interp_method: str = "structured"):
        # CasADi version requires "structured" for triangulation data
        turbine = PumpTurbine.from_dat(file_path, interp_method=interp_method)
        return cls(turbine)

    def compute_H_char(self, y, N, Q):
        """
        Compute characteristic head H given guide vane opening y, speed N, and discharge Q.

        Uses Suter transform:
            θ = arctan2(υ, α) where α = N/N_n, υ = Q/Q_n
            W_H = interp_wh(y, θ)
            H = W_H × H_n × (α² + υ²)
        """
        alpha = N / self.turbine.characteristic.n_n
        upsilon = Q / self.turbine.characteristic.q_n
        theta = ca.atan2(upsilon, alpha)

        # Call callback (Numba under the hood - fast O(log n) lookup)
        wh = self.interp_wh(y, theta)

        return wh * self.turbine.characteristic.h_n * (alpha**2 + upsilon**2)

    def compute_H(self, y, N, Q, dQ_dt):
        """Compute total head including dynamic term: H = H_char + L × dQ/dt."""
        H_char = self.compute_H_char(y, N, Q)
        L = self.turbine.hydraulic_L()
        return H_char + L * dQ_dt

    def compute_T(self, y, N, Q):
        """
        Compute torque T given guide vane opening y, speed N, and discharge Q.

        Uses Suter transform:
            θ = arctan2(υ, α) where α = N/N_n, υ = Q/Q_n
            W_B = interp_wb(y, θ)
            T = W_B × T_n × (α² + υ²)

        Args:
            y: Guide vane opening [-]
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]

        Returns:
            T: Torque [Nm]
        """
        alpha = N / self.turbine.characteristic.n_n
        upsilon = Q / self.turbine.characteristic.q_n
        theta = ca.atan2(upsilon, alpha)

        # Call callback for W_B (torque coefficient)
        wb = self.interp_wb(y, theta)

        alpha_sq_plus_upsilon_sq = alpha**2 + upsilon**2
        return wb * self.turbine.characteristic.t_n * alpha_sq_plus_upsilon_sq


class PF3LumpedCasadi:
    """
    2-state reduced-order model for PF3 pump-turbine system.

    Uses transformed coordinates [Q_T, Q_d] that separate total loop flow
    from pump imbalance. See docs/pf3-control-rom.md for derivation.

    Attributes:
        L_1, L_2, L_3, L_4: Branch inductances [s²/m²]
        R_1, R_2, R_3, R_4: Branch resistances [s²/m⁵]
        alpha, beta, gamma, delta: Admittance parameters
        det_M: Determinant of admittance matrix
    """

    # Number of states, control inputs, and TVPs
    n_x: int = 2  # [Q_T, Q_d]
    n_u: int = 1  # [N_P] (control input)
    n_tvp: int = 2  # [y_T, N_T] (time-varying parameters)

    def __init__(self, data_dir: Path):
        # =========================================================================
        # Load elements from DAT files
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

        # Branch 3: Penstock (Node 2 → Node 3) - includes turbine path
        L12 = load_pipe("L12")
        L13, L14, L15 = load_pipe("L13"), load_pipe("L14"), load_pipe("L15")
        L16, L17, L18 = load_pipe("L16"), load_pipe("L17"), load_pipe("L18")
        L19_1, L19_2 = load_pipe("L19_1"), load_pipe("L19_2")
        CONE, ELBOW_pipe = load_pipe("CONE"), load_pipe("ELBOW")
        E12, E13 = load_dloss("ELBOW12"), load_dloss("ELBOW13")
        E14, E15 = load_dloss("ELBOW14"), load_dloss("ELBOW15")

        # Branch 4: Tailwater (Node 3 → Node 1)
        L1, L2, L3, L4 = (
            load_pipe("L1"),
            load_pipe("L2"),
            load_pipe("L3"),
            load_pipe("L4"),
        )
        E1, E2, E3 = load_dloss("ELBOW1"), load_dloss("ELBOW2"), load_dloss("ELBOW3")

        # Combined branch parameters
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

        self.tank = Tank.from_dat(data_dir / "STANK.DAT")

        # Machine characteristics (CasADi-compatible)
        self.fturb = FrancisTurbineCasadi.from_dat(data_dir / "FTURB1.DAT")
        self.pump1 = FrancisTurbineCasadi.from_dat(data_dir / "PUMP1.DAT")
        self.pump2 = FrancisTurbineCasadi.from_dat(data_dir / "PUMP2.DAT")

        # =========================================================================
        # Precompute branch parameters
        # =========================================================================

        # Branch inductances
        self.L_1 = self.branch1_params.L_total
        self.L_2 = self.branch2_params.L_total
        self.L_3 = self.branch3_params.L_total + self.fturb.turbine.hydraulic_L()
        self.L_4 = self.branch4_params.L_total

        # Branch resistances
        self.R_1 = self.branch1_params.R_total
        self.R_2 = self.branch2_params.R_total
        self.R_3 = self.branch3_params.R_total
        self.R_4 = self.branch4_params.R_total

        # =========================================================================
        # Admittance parameters (see docs/pf3-control-rom.md)
        # =========================================================================

        self.alpha = 1.0 / self.L_1 + 1.0 / self.L_2  # Pump admittance sum
        self.beta = 1.0 / self.L_3  # Penstock admittance (includes turbine L)
        self.gamma = 1.0 / self.L_4  # Tailwater admittance
        self.delta = 1.0 / self.L_1 - 1.0 / self.L_2  # Pump admittance difference

        # Determinant of admittance matrix
        self.det_M = self.alpha * (self.beta + self.gamma) + self.beta * self.gamma

        # Guard against singular admittance matrix
        if abs(self.det_M) < 1e-12:
            raise ValueError(
                f"Admittance matrix nearly singular: det(M) = {self.det_M:.2e}. "
                "Check branch inductances."
            )

    def get_branch_params(self):
        """Return branch parameters for reference."""
        return {
            "L_1": self.L_1,
            "L_2": self.L_2,
            "L_3": self.L_3,
            "L_4": self.L_4,
            "R_1": self.R_1,
            "R_2": self.R_2,
            "R_3": self.R_3,
            "R_4": self.R_4,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "det_M": self.det_M,
        }

    def solve_dH(self, Q_1, Q_2, Q_T, H_P1, H_P2, H_T):
        """
        Compute pressure difference dH = H_1 - H_2 directly.

        Simplified formula (only need dH, not individual heads):
            dH = (β * b_1 - γ * b_2) / det(M)

        This is more numerically stable than computing H_1, H_2 separately.
        """
        # Friction terms
        f_1 = self.R_1 * Q_1 * ca.fabs(Q_1)
        f_2 = self.R_2 * Q_2 * ca.fabs(Q_2)
        f_3 = self.R_3 * Q_T * ca.fabs(Q_T)
        f_4 = self.R_4 * Q_T * ca.fabs(Q_T)

        # RHS terms (see docs/pf3-control-rom.md for derivation)
        b_1 = (f_1 - H_P1) / self.L_1 + (f_2 - H_P2) / self.L_2 - f_4 / self.L_4
        b_2 = (
            -(f_1 - H_P1) / self.L_1 - (f_2 - H_P2) / self.L_2 + (f_3 + H_T) / self.L_3
        )

        # Direct dH formula
        dH = (self.beta * b_1 - self.gamma * b_2) / self.det_M
        return dH

    def rhs(self, x, u):
        """
        Compute state derivatives for the 2-state ROM.

        State equations (from docs/pf3-control-rom.md):
            dQ_T/dt = α(H_1 - H_2) - Σ_f + Σ_H
            dQ_d/dt = δ(H_1 - H_2) - Δ_f + Δ_H

        where:
            Σ_f = f_1/L_1 + f_2/L_2  (friction sum)
            Δ_f = f_1/L_1 - f_2/L_2  (friction difference)
            Σ_H = H_P1/L_1 + H_P2/L_2  (pump head sum)
            Δ_H = H_P1/L_1 - H_P2/L_2  (pump head difference)

        Args:
            x: State vector [Q_T, Q_d]
            u: Input vector [y_T, N_T, N_P]

        Returns:
            State derivatives [dQ_T/dt, dQ_d/dt] as CasADi vertcat
        """
        Q_T, Q_d = x[0], x[1]
        y_T, N_T, N_P = u[0], u[1], u[2]

        # Recover individual pump flows
        Q_1 = (Q_T + Q_d) / 2  # Pump 1 flow
        Q_2 = (Q_T - Q_d) / 2  # Pump 2 flow

        # Machine heads (characteristic only, no dynamic term)
        # Pumps: negate Q for pump convention (positive Q in network = reverse operation)
        H_P1 = self.pump1.compute_H_char(1, N_P, -Q_1)
        H_P2 = self.pump2.compute_H_char(1, N_P, -Q_2)
        H_T = self.fturb.compute_H_char(y_T, N_T, Q_T)

        # Solve for pressure difference (simpler than full junction heads)
        dH = self.solve_dH(Q_1, Q_2, Q_T, H_P1, H_P2, H_T)

        # Friction terms for pump branches
        f_1 = self.R_1 * Q_1 * ca.fabs(Q_1)
        f_2 = self.R_2 * Q_2 * ca.fabs(Q_2)

        # Friction sums/differences
        Sigma_f = f_1 / self.L_1 + f_2 / self.L_2
        Delta_f = f_1 / self.L_1 - f_2 / self.L_2

        # Pump head sums/differences
        Sigma_H = H_P1 / self.L_1 + H_P2 / self.L_2
        Delta_H = H_P1 / self.L_1 - H_P2 / self.L_2

        # State equations (pump momentum, adding/subtracting)
        dQ_T = self.alpha * dH - Sigma_f + Sigma_H
        dQ_d = self.delta * dH - Delta_f + Delta_H

        return ca.vertcat(dQ_T, dQ_d)

    def compute_H_T(self, Q_T, y_T, N_T, dQ_T=0.0):
        """
        Compute turbine head output including dynamic term.

        Args:
            Q_T: Turbine flow [m³/s]
            y_T: Guide vane opening [-]
            N_T: Turbine speed [rpm]
            dQ_T: Flow derivative [m³/s²] (optional, for dynamic head)

        Returns:
            H_T: Turbine head [m]
        """
        return self.fturb.compute_H(y_T, N_T, Q_T, dQ_T)

    def build_linearization_functions(self):
        """
        Build CasADi functions for Jacobians (linearization).

        Creates functions:
            A_func(x, u_ctrl, tvp) -> ∂f/∂x  (2x2)
            B_func(x, u_ctrl, tvp) -> ∂f/∂u  (2x1, only N_P)
            C_func(x, u_ctrl, tvp) -> ∂h/∂x  (1x2, for H_T output)
            D_func(x, u_ctrl, tvp) -> ∂h/∂u  (1x1, for H_T output)

        where:
            x = [Q_T, Q_d]
            u_ctrl = [N_P]  (control input only)
            tvp = [y_T, N_T]  (time-varying parameters)
        """
        # Symbolic variables
        x = ca.SX.sym("x", 2)  # [Q_T, Q_d]
        u_ctrl = ca.SX.sym("u_ctrl", 1)  # [N_P] - control only
        tvp = ca.SX.sym("tvp", 2)  # [y_T, N_T]

        # Full input vector for rhs
        u_full = ca.vertcat(tvp[0], tvp[1], u_ctrl[0])  # [y_T, N_T, N_P]

        # State dynamics f(x, u)
        f = self.rhs(x, u_full)

        # Output h(x, u) = H_T (characteristic head, no dynamic term for simplicity)
        Q_T = x[0]
        y_T, N_T = tvp[0], tvp[1]
        h = self.fturb.compute_H_char(y_T, N_T, Q_T)

        # Jacobians
        A = ca.jacobian(f, x)  # 2x2
        B = ca.jacobian(f, u_ctrl)  # 2x1
        C = ca.jacobian(h, x)  # 1x2
        D = ca.jacobian(h, u_ctrl)  # 1x1

        # Compile functions
        self._A_func = ca.Function("A", [x, u_ctrl, tvp], [A])
        self._B_func = ca.Function("B", [x, u_ctrl, tvp], [B])
        self._C_func = ca.Function("C", [x, u_ctrl, tvp], [C])
        self._D_func = ca.Function("D", [x, u_ctrl, tvp], [D])
        self._f_func = ca.Function("f", [x, u_ctrl, tvp], [f])
        self._h_func = ca.Function("h", [x, u_ctrl, tvp], [h])

        self._linearization_built = True

    def linearize_at(self, x0, u0, tvp0):
        """
        Linearize the model at operating point (x0, u0, tvp0).

        Returns continuous-time linearized model:
            ẋ ≈ A·x + B·u + c
            y ≈ C·x + D·u + d

        where c and d are affine offsets to match the nonlinear model at the
        linearization point.

        Args:
            x0: State [Q_T, Q_d] (2,)
            u0: Control input [N_P] (1,)
            tvp0: Time-varying params [y_T, N_T] (2,)

        Returns:
            dict with keys: A, B, C, D, c, d, f0, h0
        """
        if not hasattr(self, "_linearization_built") or not self._linearization_built:
            self.build_linearization_functions()

        x0 = np.asarray(x0).flatten()
        u0 = np.asarray(u0).flatten()
        tvp0 = np.asarray(tvp0).flatten()

        # Evaluate Jacobians
        A = np.asarray(self._A_func(x0, u0, tvp0))
        B = np.asarray(self._B_func(x0, u0, tvp0))
        C = np.asarray(self._C_func(x0, u0, tvp0))
        D = np.asarray(self._D_func(x0, u0, tvp0))

        # Evaluate nonlinear functions at operating point
        f0 = np.asarray(self._f_func(x0, u0, tvp0)).flatten()
        h0 = np.asarray(self._h_func(x0, u0, tvp0)).flatten()

        # Affine offsets: c = f(x0,u0) - A·x0 - B·u0
        c = f0 - A @ x0 - B @ u0
        d = h0 - C @ x0 - D @ u0

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "c": c,
            "d": d,
            "f0": f0,
            "h0": h0,
        }

    def discretize(self, A, B, c, dt, method="euler"):
        """
        Discretize continuous-time affine model to discrete-time.

        Continuous: ẋ = A·x + B·u + c
        Discrete:   x_{k+1} = A_d·x_k + B_d·u_k + c_d

        Args:
            A: State matrix (n_x, n_x)
            B: Input matrix (n_x, n_u)
            c: Affine offset (n_x,)
            dt: Time step
            method: "euler" or "exact"

        Returns:
            A_d, B_d, c_d
        """
        n_x = A.shape[0]
        I = np.eye(n_x)

        if method == "euler":
            A_d = I + A * dt
            B_d = B * dt
            c_d = c * dt
        elif method == "exact":
            # Exact discretization using matrix exponential
            # Build augmented system for affine term
            from scipy.linalg import expm

            n_u = B.shape[1]
            # Augmented matrix [A, B, c; 0, 0, 0; 0, 0, 0]
            M = np.zeros((n_x + n_u + 1, n_x + n_u + 1))
            M[:n_x, :n_x] = A
            M[:n_x, n_x : n_x + n_u] = B
            M[:n_x, -1] = c

            eM = expm(M * dt)
            A_d = eM[:n_x, :n_x]
            B_d = eM[:n_x, n_x : n_x + n_u]
            c_d = eM[:n_x, -1]
        else:
            raise ValueError(f"Unknown discretization method: {method}")

        return A_d, B_d, c_d

    def get_initial_state(self):
        """
        Return initial state [Q_T, Q_d] from DAT file initial conditions.

        Q_T = Q_1 + Q_2 (total flow)
        Q_d = Q_1 - Q_2 (difference flow)
        """
        Q_1 = self.branch1_params.Q0  # Pump 1 initial flow
        Q_2 = self.branch2_params.Q0  # Pump 2 initial flow
        Q_T = Q_1 + Q_2
        Q_d = Q_1 - Q_2
        return np.array([Q_T, Q_d])

    def get_initial_pump_flows(self):
        """Return initial individual pump flows [Q_1, Q_2] from DAT file initial conditions."""
        return np.array([self.branch1_params.Q0, self.branch2_params.Q0])

    def print_summary(self):
        """Print model parameters."""
        print("PF3 Control ROM (2-state)")
        print("=" * 50)
        print(f"\nStates: [Q_T, Q_d]")
        print(f"  Q_T = Q_1 + Q_2 (total/turbine flow)")
        print(f"  Q_d = Q_1 - Q_2 (pump imbalance)")
        print(f"\nBranch inductances [s²/m²]:")
        print(f"  L_1 (pump 1):    {self.L_1:.4f}")
        print(f"  L_2 (pump 2):    {self.L_2:.4f}")
        print(f"  L_3 (penstock):  {self.L_3:.4f}")
        print(f"  L_4 (tailwater): {self.L_4:.4f}")
        print(f"\nBranch resistances [s²/m⁵]:")
        print(f"  R_1 (pump 1):    {self.R_1:.4f}")
        print(f"  R_2 (pump 2):    {self.R_2:.4f}")
        print(f"  R_3 (penstock):  {self.R_3:.4f}")
        print(f"  R_4 (tailwater): {self.R_4:.4f}")
        print(f"\nAdmittance parameters:")
        print(f"  α = 1/L_1 + 1/L_2 = {self.alpha:.6f}")
        print(f"  β = 1/L_3         = {self.beta:.6f}")
        print(f"  γ = 1/L_4         = {self.gamma:.6f}")
        print(f"  δ = 1/L_1 - 1/L_2 = {self.delta:.6f}")
        print(f"  det(M) = α(β+γ) + βγ = {self.det_M:.6f}")
        print(f"\nInitial state:")
        x0 = self.get_initial_state()
        Q0 = self.get_initial_pump_flows()
        print(f"  Q_1 = {Q0[0]:.6f} m³/s")
        print(f"  Q_2 = {Q0[1]:.6f} m³/s")
        print(f"  Q_T = {x0[0]:.6f} m³/s")
        print(f"  Q_d = {x0[1]:.6f} m³/s")


class PF3Simulator:
    """
    Fast RK4 simulator using compiled CasADi expressions.

    Uses the 2-state model: [Q_T, Q_d].
    Derived flows: Q_1 = (Q_T + Q_d)/2, Q_2 = (Q_T - Q_d)/2
    """

    def __init__(self, model: PF3LumpedCasadi):
        self.model = model
        self._mapaccum_cache = {}  # Cache mapaccum functions by n_steps

        # CasADi symbolic variables (2-state model)
        x = ca.SX.sym("x", 2)  # [Q_T, Q_d]
        u = ca.SX.sym("u", 3)  # [y_T, N_T, N_P]

        # Build RHS from model
        xdot = model.rhs(x, u)

        # Compile RHS function
        self.rhs_func = ca.Function("rhs", [x, u], [xdot])

        # Build RK4 as single compiled function
        dt = ca.SX.sym("dt")
        k1 = self.rhs_func(x, u)
        k2 = self.rhs_func(x + 0.5 * dt * k1, u)
        k3 = self.rhs_func(x + 0.5 * dt * k2, u)
        k4 = self.rhs_func(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # RK4 step function: (x, u, dt) -> x_next
        self.rk4_step = ca.Function("rk4_step", [x, u, dt], [x_next])

        # For mapaccum: step function with fixed dt baked in
        self._x_sym = x
        self._u_sym = u
        self._x_next_sym = x_next
        self._dt_sym = dt

        # Build H_T output function (characteristic only)
        Q_T_sym = x[0]
        y_T_sym, N_T_sym = u[0], u[1]
        H_T_sym = model.fturb.compute_H_char(y_T_sym, N_T_sym, Q_T_sym)
        self.H_T_func = ca.Function("H_T", [x, u], [H_T_sym])

    def _get_mapaccum(self, n_steps: int, dt: float) -> ca.Function:
        """Get or create mapaccum function for given n_steps and dt."""
        key = (n_steps, dt)
        if key not in self._mapaccum_cache:
            # Create step function with fixed dt
            x_next_fixed = ca.substitute(self._x_next_sym, self._dt_sym, dt)
            step_fixed = ca.Function("step", [self._x_sym, self._u_sym], [x_next_fixed])
            # Create mapaccum: loops internally in C
            self._mapaccum_cache[key] = step_fixed.mapaccum("sim", n_steps)
        return self._mapaccum_cache[key]

    def simulate(self, x0, t_span, dt, u_func):
        """
        Run RK4 simulation.

        Args:
            x0: Initial state [Q_T, Q_d]
            t_span: (t_start, t_end)
            dt: Time step
            u_func: Callable u_func(t) -> [y_T, N_T, N_P]

        Returns:
            dict with 't', 'x', state trajectories, derived flows, and H_T
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        # Time array
        t_hist = np.linspace(t_start, t_end, n_steps + 1)

        # Pre-compute all inputs (required for mapaccum)
        u_all = np.zeros((3, n_steps))
        for i in range(n_steps):
            t = t_start + i * dt
            u_all[:, i] = u_func(t)

        x0_arr = np.asarray(x0).flatten()

        # Single CasADi call for entire trajectory (mapaccum)
        F_sim = self._get_mapaccum(n_steps, dt)
        x_result = F_sim(x0_arr, u_all)
        # Result is 2 x n_steps matrix (each column is state after step i)
        x_hist = np.zeros((n_steps + 1, 2))
        x_hist[0] = x0_arr
        x_hist[1:] = np.asarray(x_result).T

        # Compute derivatives at each point (vectorized via map)
        rhs_map = self.rhs_func.map(n_steps + 1)
        u_all_ext = np.column_stack([u_all, u_all[:, -1]])  # Extend for last point
        dx_hist = np.asarray(rhs_map(x_hist.T, u_all_ext)).T

        # Extract state components
        Q_T = x_hist[:, 0]
        Q_d = x_hist[:, 1]
        dQ_T = dx_hist[:, 0]
        dQ_d = dx_hist[:, 1]

        # Compute derived flows
        Q_1 = (Q_T + Q_d) / 2
        Q_2 = (Q_T - Q_d) / 2
        dQ_1 = (dQ_T + dQ_d) / 2
        dQ_2 = (dQ_T - dQ_d) / 2

        # Compute H_T (vectorized)
        H_T_map = self.H_T_func.map(n_steps + 1)
        H_T = np.asarray(H_T_map(x_hist.T, u_all_ext)).flatten()

        return {
            "t": t_hist,
            "x": x_hist,
            "x_dot": dx_hist,
            # State variables
            "Q_T": Q_T,
            "Q_d": Q_d,
            "dQ_T": dQ_T,
            "dQ_d": dQ_d,
            # Derived flows (individual pumps)
            "Q_1": Q_1,
            "Q_2": Q_2,
            "dQ_1": dQ_1,
            "dQ_2": dQ_2,
            # Output
            "H_T": H_T,
        }
