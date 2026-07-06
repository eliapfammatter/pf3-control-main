"""
Nonlinear MPC Controller using PF3LumpedCasadi Model.

States: [Q_T, Q_d]
    Q_T = Q_1 + Q_2 (total flow)
    Q_d = Q_1 - Q_2 (pump imbalance)
Control: N_P (pump speed)
Output: H_T (turbine head)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import casadi as ca
import do_mpc
import numpy as np

from src.controllers import Controller, ControllerState
from src.helpers import FPoints
from src.pf3_lumped_casadi import PF3LumpedCasadi


@dataclass
class MPCParams:
    """Parameters for MPC controllers (NMPC and LTV-MPC).

    Structural parameters (N_p, dt, bounds, P_max, R_du) are baked into the
    compiled solver. Changing them triggers recompilation.

    Tuning parameters (Q_H_T, Q_terminal) can be changed at runtime
    without recompilation.
    """

    # Structural parameters (require recompilation if changed)
    dt: float = 0.1
    N_p: int = 20
    u_min: float = -400.0
    u_max: float = -200.0
    P_max: float = 300000.0  # Maximum power per pump [W] = 300 kW
    R_du: float = 0.01  # Control rate penalty

    # Tuning parameters (can be changed at runtime)
    Q_H_T: float = 10.0
    Q_terminal: float = 100.0

    # Solver options (JIT disabled since we use compile_nlp)
    nlpsol_opts: dict = field(
        default_factory=lambda: {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.hessian_approximation": "limited-memory",
        }
    )

    def get_cache_key(self) -> str:
        """Return cache key based on structural parameters.

        Uses only alphanumeric characters and underscores for valid C identifier.
        """

        def sanitize(val: float) -> str:
            """Convert float to valid C identifier substring."""
            s = f"{val:g}"  # Compact representation
            s = s.replace(".", "p")  # decimal point -> 'p'
            s = s.replace("-", "m")  # minus -> 'm'
            s = s.replace("+", "")  # remove plus
            return s

        return (
            f"nlp_Np{self.N_p}_dt{sanitize(self.dt)}"
            f"_umin{sanitize(self.u_min)}_umax{sanitize(self.u_max)}"
            f"_Pmax{sanitize(self.P_max)}_Rdu{sanitize(self.R_du)}"
        )


class NMPCCasadi(Controller):
    """Nonlinear MPC using PF3LumpedCasadi 2-state model."""

    def __init__(
        self,
        H_ref: FPoints,
        model: PF3LumpedCasadi,
        fpoints_y_T: FPoints,
        fpoints_N_T: FPoints,
        params: Optional[MPCParams] = None,
        u_nominal: float = -313.0,
    ):
        self._fpoints_H_ref = H_ref
        self._fpoints_y_T = fpoints_y_T
        self._fpoints_N_T = fpoints_N_T
        self.model = model
        self.params = params or MPCParams()
        self.u_nominal = u_nominal

        self._u_prev = u_nominal
        self._x_est = model.get_initial_state()
        self._last_prediction: Optional[np.ndarray] = None
        self._current_state: Optional[ControllerState] = None
        self._H_T_model_at_x0: Optional[float] = None

        self._model = None
        self._mpc = None

    def build_mpc(self) -> None:
        """Build do-mpc model and controller.

        Compiles the NLP to a .so file for fast loading on subsequent runs.
        Tuning parameters (Q_H_T, Q_terminal, R_du) are symbolic and can be
        changed at runtime without recompilation.
        """
        model = do_mpc.model.Model("continuous")

        # States: [Q_T, Q_d]
        Q_T = model.set_variable(var_type="_x", var_name="Q_T")
        Q_d = model.set_variable(var_type="_x", var_name="Q_d")

        # Control
        N_P = model.set_variable(var_type="_u", var_name="N_P")

        # TVPs (turbine inputs)
        y_T = model.set_variable(var_type="_tvp", var_name="y_T")
        N_T = model.set_variable(var_type="_tvp", var_name="N_T")
        H_ref = model.set_variable(var_type="_tvp", var_name="H_ref")

        # Tuning parameters (can be changed at runtime without recompilation)
        Q_H_T_p = model.set_variable(var_type="_p", var_name="Q_H_T")
        Q_terminal_p = model.set_variable(var_type="_p", var_name="Q_terminal")

        # Dynamics from CasADi model
        xdot = self.model.rhs([Q_T, Q_d], [y_T, N_T, N_P])
        model.set_rhs("Q_T", xdot[0])
        model.set_rhs("Q_d", xdot[1])

        # Turbine head expressions
        dQ_T_dt = xdot[0]
        # Full head (with dynamic term) for stage cost
        H_T = self.model.fturb.compute_H(y_T, N_T, Q_T, dQ_T_dt)
        # Characteristic head for terminal cost (steady state: dQ/dt = 0)
        H_T_ss = self.model.fturb.compute_H_char(y_T, N_T, Q_T)
        model.set_expression("H_T", H_T)
        model.set_expression("H_T_ss", H_T_ss)

        # Power expressions
        # Recover individual pump flows
        Q_1 = (Q_T + Q_d) / 2  # Pump 1 flow
        Q_2 = (Q_T - Q_d) / 2  # Pump 2 flow

        # Compute pump torques (y=1 for pumps, negate Q for pump convention)
        T_P1 = self.model.pump1.compute_T(1, N_P, -Q_1)
        T_P2 = self.model.pump2.compute_T(1, N_P, -Q_2)

        # Compute power: P = T × ω where ω = 2π N / 60
        omega = 2 * ca.pi * N_P / 60  # [rad/s]
        P_P1 = T_P1 * omega  # [W]
        P_P2 = T_P2 * omega  # [W]

        model.set_expression("P_P1", P_P1)
        model.set_expression("P_P2", P_P2)
        model.set_expression("T_P1", T_P1)
        model.set_expression("T_P2", T_P2)

        model.setup()

        # MPC controller
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.params.N_p,
            t_step=self.params.dt,
            state_discretization="collocation",
            collocation_type="radau",
            collocation_deg=2,
            collocation_ni=2,
            store_full_solution=True,
            nlpsol_opts=self.params.nlpsol_opts,
        )

        # TVP function
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(self.params.N_p + 1):
                t_k = t_now + k * self.params.dt
                tvp_template["_tvp", k, "y_T"] = float(self._fpoints_y_T(t_k))
                tvp_template["_tvp", k, "N_T"] = float(self._fpoints_N_T(t_k))
                tvp_template["_tvp", k, "H_ref"] = float(self._fpoints_H_ref(t_k))
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        # Parameter function (tuning parameters)
        p_template = mpc.get_p_template(1)  # 1 scenario

        def p_fun(t_now):
            p_template["_p", :, "Q_H_T"] = self.params.Q_H_T
            p_template["_p", :, "Q_terminal"] = self.params.Q_terminal
            return p_template

        mpc.set_p_fun(p_fun)

        # Cost: track H_ref (using symbolic parameters for weights)
        H_T_expr = model.aux["H_T"]
        H_T_ss_expr = model.aux["H_T_ss"]
        H_ref_tvp = model.tvp["H_ref"]

        lterm = Q_H_T_p * (H_T_expr - H_ref_tvp) ** 2
        mterm = Q_terminal_p * (H_T_ss_expr - H_ref_tvp) ** 2
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(N_P=self.params.R_du)

        # Bounds (fixed, part of cache key)
        mpc.bounds["lower", "_u", "N_P"] = self.params.u_min
        mpc.bounds["upper", "_u", "N_P"] = self.params.u_max

        # Power constraints (fixed, part of cache key)
        P_P1_expr = model.aux["P_P1"]
        P_P2_expr = model.aux["P_P2"]
        mpc.set_nl_cons("power_pump1", -P_P1_expr, ub=self.params.P_max)
        mpc.set_nl_cons("power_pump2", -P_P2_expr, ub=self.params.P_max)

        mpc.setup()

        # Compile NLP to .so file (or load existing)
        # Note: Use simple filenames (no path) to avoid CasADi check_name assertion failure
        cache_key = self.params.get_cache_key()
        cname = f"{cache_key}.c"
        libname = f"{cache_key}.so"
        compiled_path = Path(libname)

        if compiled_path.exists():
            print(f"[NMPC] Loading compiled solver: {compiled_path}")
        else:
            print(f"[NMPC] Compiling solver to: {compiled_path}")
            print(
                f"       (N_p={self.params.N_p}, dt={self.params.dt}, "
                f"u_min={self.params.u_min}, u_max={self.params.u_max}, "
                f"P_max={self.params.P_max}, R_du={self.params.R_du})"
            )

        # Monkey-patch do_mpc bug: nlpsol not imported in optimizer module
        # This is needed to be able to compile the NMPC.
        import do_mpc.optimizer as _opt

        if not hasattr(_opt, "nlpsol"):
            _opt.nlpsol = ca.nlpsol

        # Custom compiler command to link triinterp library
        # Note: $ORIGIN must be escaped to prevent shell expansion
        compiler_cmd = (
            f"gcc -fPIC -shared -O3 {cname} -o {libname} "
            f"-L./src/triinterp -ltriinterp '-Wl,-rpath,$ORIGIN/src/triinterp'"
        )

        mpc.compile_nlp(
            overwrite=False,
            cname=cname,
            libname=libname,
            compiler_command=compiler_cmd,
        )

        self._model = model
        self._mpc = mpc

    def reset(self) -> None:
        if self._model == None or self._mpc == None:
            self.build_mpc()
        self._u_prev = self.u_nominal
        self._x_est = self.model.get_initial_state()
        self._last_prediction = None  # Also signals need for set_initial_guess()
        self._H_T_model_at_x0 = None
        self._last_power_P1 = None
        self._last_power_P2 = None

    @property
    def name(self) -> str:
        return f"NMPC-CasADi (2-state, N_p={self.params.N_p})"

    def update_fpoints(self, fpoints_H_ref, fpoints_y_T, fpoints_N_T):
        self._fpoints_H_ref = fpoints_H_ref
        self._fpoints_y_T = fpoints_y_T
        self._fpoints_N_T = fpoints_N_T

    def compute_pump_speed(self, time: float, state: ControllerState) -> float:
        if self._model == None or self._mpc == None:
            self.build_mpc()

        self._current_state = state

        # State estimation (oracle: use FMU Q values directly)
        # Convert from individual pump flows to [Q_T, Q_d]
        if state.Q_P1 is not None and state.Q_P2 is not None and state.Q_T is not None:
            # FMU convention: negative Q for pumps, convert to network convention
            Q_T = state.Q_T
            Q_d = state.Q_T + 2 * state.Q_P2
        else:
            raise ValueError("the flows must be given!")

        self._x_est = np.array([Q_T, Q_d])

        # =====================================================================
        # DIAGNOSTIC: Compare FMU vs MPC model turbine head
        # =====================================================================
        # FMU values
        H_T_fmu = state.H_T
        Q_T_fmu = state.Q_T
        N_T_fmu = state.N_T
        y_T_fmu = state.y_T

        # MPC internal values (Q from state estimation, N/y from TVPs)
        Q_T_mpc = Q_T  # From state estimation above
        N_T_mpc = float(self._fpoints_N_T(time))
        y_T_mpc = float(self._fpoints_y_T(time))

        # MPC model prediction (with correct dQ_dt from dynamics)
        # Use FMU's actual N_P for fair comparison (not self._u_prev)
        N_P_fmu = state.N_P
        x_mpc = [Q_T_mpc, Q_d]
        u_mpc = [y_T_fmu, N_T_fmu, N_P_fmu]  # Use FMU values for all inputs
        xdot = self.model.rhs(x_mpc, u_mpc)
        dQ_T_mpc = float(xdot[0])
        H_T_model = float(
            self.model.fturb.compute_H(y_T_fmu, N_T_fmu, Q_T_mpc, dQ_T_mpc)
        )

        # Solve MPC
        x0 = self._x_est.reshape(-1, 1)
        self._mpc.x0 = x0

        # Initialize warm-start on first step (after reset or init)
        if self._last_prediction is None:
            self._mpc.u0 = np.array([[self._u_prev]])
            self._mpc.set_initial_guess()

        try:
            u0 = self._mpc.make_step(x0)
            N_P = float(np.asarray(u0).flatten()[0])

            # No post-processing needed - constraints handled in MPC formulation

            # Store prediction
            try:
                pred = self._mpc.data.prediction(("_aux", "H_T"))
                self._last_prediction = pred[0, :, 0]
                # Store model's H_T at current state (for mismatch diagnostics)
                self._H_T_model_at_x0 = float(self._last_prediction[0])

                # Store power predictions
                pred_P1 = self._mpc.data.prediction(("_aux", "P_P1"))
                pred_P2 = self._mpc.data.prediction(("_aux", "P_P2"))
                self._last_power_P1 = pred_P1[0, :, 0]
                self._last_power_P2 = pred_P2[0, :, 0]
            except Exception:
                self._H_T_model_at_x0 = None
                self._last_power_P1 = None
                self._last_power_P2 = None

        except Exception as e:
            print(f"MPC failed: {e}")
            N_P = self._u_prev

        self._u_prev = N_P
        return N_P

    @property
    def predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return MPC's internal H_T prediction (including model's estimate at t=0)."""
        return self._last_prediction

    @property
    def predicted_power_P1(self) -> Optional[np.ndarray]:
        """Return MPC's predicted pump 1 power trajectory [W]."""
        return self._last_power_P1

    @property
    def predicted_power_P2(self) -> Optional[np.ndarray]:
        """Return MPC's predicted pump 2 power trajectory [W]."""
        return self._last_power_P2
