"""
Linear Time-Varying MPC Controller using do-mpc's LinearModel.

Linearizes the nonlinear model at each control step using our CasADi-based
linearization, then builds a do-mpc LinearModel and MPC for solving.

States: [Q_T, Q_d]
Control: N_P (pump speed)
Output: H_T (turbine head)
"""

from typing import Optional

import do_mpc
import numpy as np

from src.controllers import Controller, ControllerState
from src.helpers import FPoints
from src.nmpc_casadi import MPCParams
from src.pf3_lumped_casadi import PF3LumpedCasadi


class LTVMPCDoMPC(Controller):
    """
    Linear Time-Varying MPC using do-mpc's LinearModel.

    At each step:
    1. Linearize nonlinear model around current state (using CasADi Jacobians)
    2. Create do-mpc LinearModel with numeric A, B, C matrices
    3. Build MPC and solve
    4. Apply first control action
    """

    def __init__(
        self,
        H_ref: float,
        model: PF3LumpedCasadi,
        fpoints_y_T: FPoints,
        fpoints_N_T: FPoints,
        params: Optional[MPCParams] = None,
        u_nominal: float = -313.0,
    ):
        self.H_ref = H_ref
        self.model = model
        self._fpoints_y_T = fpoints_y_T
        self._fpoints_N_T = fpoints_N_T
        self.params = params or MPCParams()
        self.u_nominal = u_nominal

        self._u_prev = u_nominal
        self._x_est = model.get_initial_state()
        self._last_prediction: Optional[np.ndarray] = None

        # Build linearization functions (CasADi Jacobians)
        self.model.build_linearization_functions()

    def _build_linear_model(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray
    ) -> do_mpc.model.LinearModel:
        """Build do-mpc LinearModel from numeric matrices."""
        linear_model = do_mpc.model.LinearModel("continuous")

        # Define deviation variables
        linear_model.set_variable("_x", "delQ_T")
        linear_model.set_variable("_x", "delQ_d")
        linear_model.set_variable("_u", "delN_P")

        # Setup with matrices (dynamics: dx/dt = A*x + B*u, output: y = C*x)
        linear_model.setup(A=A, B=B, C=C)

        return linear_model

    def _build_mpc(
        self,
        linear_model: do_mpc.model.LinearModel,
        C: np.ndarray,
        H_T_ss: float,
        N_P_ss: float,
    ) -> do_mpc.controller.MPC:
        """Build MPC controller from linearized model."""
        import casadi as ca

        mpc = do_mpc.controller.MPC(linear_model)

        mpc.set_param(
            n_horizon=self.params.N_p,
            t_step=self.params.dt,
            store_full_solution=True,
            nlpsol_opts=self.params.nlpsol_opts,
        )

        # Cost: track H_ref
        # H_T = H_T_ss + C @ del_x, so deviation from H_ref is:
        # (H_T_ss + del_H_T) - H_ref = del_H_T + (H_T_ss - H_ref)
        # We want to minimize |H_T - H_ref|^2 = |del_H_T + offset|^2
        offset = H_T_ss - self.H_ref

        # Compute output expression from state: del_H_T = C @ x
        C_dm = ca.DM(C)
        del_H_T = C_dm @ linear_model.x.cat  # Output = C @ x (scalar)

        lterm = self.params.Q_H_T * (del_H_T + offset) ** 2
        mterm = self.params.Q_terminal * (del_H_T + offset) ** 2
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(delN_P=self.params.R_du)

        # Bounds on control deviation
        mpc.bounds["lower", "_u", "delN_P"] = self.params.u_min - N_P_ss
        mpc.bounds["upper", "_u", "delN_P"] = self.params.u_max - N_P_ss

        mpc.setup()
        return mpc

    def reset(self) -> None:
        self._u_prev = self.u_nominal
        self._x_est = self.model.get_initial_state()
        self._last_prediction = None

    @property
    def name(self) -> str:
        return f"LTV-MPC-DoMPC (N_p={self.params.N_p}, Q_T_P={self.params.Q_H_T})"

    def compute_pump_speed(self, time: float, state: ControllerState) -> float:
        """Compute optimal pump speed using LTV-MPC."""
        # State estimation (oracle: use FMU Q values directly)
        if state.Q_P1 is not None and state.Q_P2 is not None and state.Q_T is not None:
            Q_T = state.Q_T
            Q_d = state.Q_T + 2 * state.Q_P2
        else:
            raise ValueError("Flow measurements required!")

        self._x_est = np.array([Q_T, Q_d])

        # Current operating point for linearization
        x_ss = np.array([Q_T, Q_d])
        u_ss = np.array([state.N_P])
        tvp_ss = np.array([state.y_T, state.N_T])

        # Linearize nonlinear model at current state
        lin = self.model.linearize_at(x_ss, u_ss, tvp_ss)
        A, B, C = lin["A"], lin["B"], lin["C"]
        H_T_ss = float(lin["h0"][0])  # Output at linearization point

        # Build LinearModel and MPC
        try:
            linear_model = self._build_linear_model(A, B, C)
            mpc = self._build_mpc(linear_model, C, H_T_ss, state.N_P)

            # Initial state (deviation from linearization point = 0)
            x0_dev = np.array([[0.0], [0.0]])
            mpc.x0 = x0_dev
            mpc.u0 = np.array([[0.0]])
            mpc.set_initial_guess()

            # Solve
            u0_dev = mpc.make_step(x0_dev)
            delta_N_P = float(np.asarray(u0_dev).flatten()[0])

            # Convert deviation to absolute value
            N_P = state.N_P + delta_N_P

            # Extract H_T prediction trajectory
            # del_x predictions: shape (n_states, n_horizon+1, n_scenarios)
            try:
                del_x_pred = np.hstack(
                    [
                        mpc.data.prediction(("_x", "delQ_T")),
                        mpc.data.prediction(("_x", "delQ_d")),
                    ]
                ).T  # Shape: (2, N_p+1)
                # H_T = H_T_ss + C @ del_x
                del_H_T_pred = (C @ del_x_pred).flatten()
                self._last_prediction = H_T_ss + del_H_T_pred
            except Exception:
                self._last_prediction = None

        except Exception as e:
            print(f"LTV-MPC solve failed: {e}")
            N_P = self._u_prev
            self._last_prediction = None

        # Clamp to bounds
        N_P = np.clip(N_P, self.params.u_min, self.params.u_max)

        self._u_prev = N_P
        return N_P

    @property
    def predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return predicted H_T trajectory (not implemented for LTV)."""
        return self._last_prediction
