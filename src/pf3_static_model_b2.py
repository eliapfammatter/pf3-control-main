"""
PF3 Static Gain Model per Appendix B.

This implements the H-based (forward characteristic) formulation:
- Characteristic: H = Γ(y, N, Q) — head as function of flow (single-valued)
- Linearization: Γ^H_Q = ∂H/∂Q, Γ^H_N = ∂H/∂N, etc.
- Static gain: K_P = -C @ A^{-1} @ B

Same interface as PF3System so it can be used with existing PI controller.
"""

from pathlib import Path

import numpy as np

from .characteristics import Characteristics
from .hydraulic_elements import DiscreteLoss, Pipe


class PF3StaticModelB2:
    """
    Static gain model per Appendix B (H-based forward characteristic formulation).

    Computes K_P = δH_T / δN_P using forward characteristics H = Γ(y, N, Q).

    The model uses:
    - H = Γ(y, N, Q) characteristic linearization (forward, single-valued)
    - States: x = [δH_T, δH_P1, δH_P2]^T
    - Control input: u = δN_P
    - Disturbances: d = [δy_T, δN_T]^T
    - Output: y = δH_T

    Key advantage: Forward characteristic avoids root-finding and is always
    single-valued. Mass balance Q_T = Q_1 + Q_2 is enforced by construction.

    Parameters
    ----------
    turbine : Characteristics
        Turbine characteristic model
    pump : Characteristics
        Pump characteristic model (same for P1 and P2)
    hydraulic_elements_dir : Path
        Directory containing DAT files for hydraulic elements
    eps_rel : float
        Relative epsilon for numerical differentiation (default: 2e-2)
    """

    def __init__(
        self,
        turbine: Characteristics,
        pump: Characteristics,
        hydraulic_elements_dir: Path,
        eps_rel: float = 2e-2,
    ):
        self.turbine = turbine
        self.pump = pump
        self.eps_rel = eps_rel

        # Load resistances following B2's branch numbering:
        # R_1 = pump 1 branch resistance
        # R_2 = pump 2 branch resistance
        # R_34 = turbine + penstock branch resistance (R_3 + R_4)
        self.R_1 = self._load_pump1_resistance(hydraulic_elements_dir)
        self.R_2 = self._load_pump2_resistance(hydraulic_elements_dir)
        self.R_34 = self._load_turbine_resistance(hydraulic_elements_dir)

    @staticmethod
    def _load_pump1_resistance(data_path: Path) -> float:
        """Load pump 1 branch resistance (B2's R_1)."""
        elements = [
            DiscreteLoss.from_dat(data_path / "ELBOW4.DAT"),
            Pipe.from_dat(data_path / "L5.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW5.DAT"),
            Pipe.from_dat(data_path / "LP1.DAT"),
            # Pump is H_source, not resistance
            Pipe.from_dat(data_path / "L11.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW11.DAT"),
        ]
        return sum(e.hydraulic_R() for e in elements)

    @staticmethod
    def _load_pump2_resistance(data_path: Path) -> float:
        """Load pump 2 branch resistance (B2's R_2)."""
        elements = [
            DiscreteLoss.from_dat(data_path / "ELBOW7.DAT"),
            Pipe.from_dat(data_path / "L6.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW8.DAT"),
            Pipe.from_dat(data_path / "L7.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW9.DAT"),
            Pipe.from_dat(data_path / "LP2.DAT"),
            # Pump is H_source, not resistance
            Pipe.from_dat(data_path / "L10.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW10.DAT"),
        ]
        return sum(e.hydraulic_R() for e in elements)

    @staticmethod
    def _load_turbine_resistance(data_path: Path) -> float:
        """Load turbine + penstock branch resistance (B2's R_34 = R_3 + R_4)."""
        # Penstock path: STANK -> L1 -> L2 -> L3 -> L4 -> NODE1
        penstock = [
            Pipe.from_dat(data_path / "L1.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW1.DAT"),
            Pipe.from_dat(data_path / "L2.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW2.DAT"),
            Pipe.from_dat(data_path / "L3.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW3.DAT"),
            Pipe.from_dat(data_path / "L4.DAT"),
        ]
        # Draft tube path: NODE2 -> L12 -> ... -> L19 -> CONE -> ELBOW -> STANK
        draft_tube = [
            Pipe.from_dat(data_path / "L12.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW12.DAT"),
            Pipe.from_dat(data_path / "L13.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW13.DAT"),
            Pipe.from_dat(data_path / "L14.DAT"),
            Pipe.from_dat(data_path / "L15.DAT"),
            Pipe.from_dat(data_path / "L16.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW14.DAT"),
            Pipe.from_dat(data_path / "L17.DAT"),
            DiscreteLoss.from_dat(data_path / "ELBOW15.DAT"),
            Pipe.from_dat(data_path / "L18.DAT"),
            Pipe.from_dat(data_path / "L19_1.DAT"),
            Pipe.from_dat(data_path / "L19_2.DAT"),
            Pipe.from_dat(data_path / "CONE.DAT"),
            # FTURB is H_source, not resistance
            # ELBOW at surge tank has negligible resistance
        ]
        return sum(e.hydraulic_R() for e in penstock + draft_tube)

    def compute_G_u(
        self,
        y_T: float,
        N_T: float,
        H_T: float,
        N_P: float,
        H_P1: float,
        H_P2: float,
    ) -> dict:
        """
        Compute static gain K_P = δH_T / δN_P per Appendix B.

        Uses H-based (forward) linearization: H = Γ(y, N, Q).

        Parameters
        ----------
        y_T : float
            Turbine guide vane opening [-]
        N_T : float
            Turbine rotational speed [rpm]
        H_T : float
            Turbine head [m]
        N_P : float
            Pump rotational speed [rpm]
        H_P1 : float
            Pump 1 head [m]
        H_P2 : float
            Pump 2 head [m]

        Returns
        -------
        dict
            Dictionary containing:
            - 'G_u': Static gain K_P [m/rpm]
            - Partial derivatives and intermediate coefficients
            - Operating point information
        """
        # Step 1: Compute Q values at operating point
        # Get pump flows from inverse characteristic (pump convention)
        Q_P1, _ = self.pump.QT_from_yNH(1.0, N_P, H_P1)
        Q_P2, _ = self.pump.QT_from_yNH(1.0, N_P, H_P2)

        # Convert to network convention and ENFORCE mass balance
        # Q_1, Q_2 = flow from NODE1 to NODE2 (positive)
        # Q_T = Q_3 = turbine flow = Q_1 + Q_2 (by mass balance)
        Q_1 = -Q_P1  # Network convention
        Q_2 = -Q_P2
        Q_T = Q_1 + Q_2  # Mass balance ENFORCED (not from turbine inverse!)

        # Step 2: Compute H-based partial derivatives using linearize_H
        # linearize_H returns [∂H/∂y, ∂H/∂N, ∂H/∂Q]

        # Turbine: H_T = Γ^{H,T}(y_T, N_T, Q_T)
        jac_T = self.turbine.linearize_H(y_T, N_T, Q_T, suter=True, eps_rel=self.eps_rel)
        Gamma_T_y = jac_T[0]   # ∂H_T/∂y_T
        Gamma_T_N = jac_T[1]   # ∂H_T/∂N_T
        Gamma_T_Q = jac_T[2]   # ∂H_T/∂Q_T

        # Pump 1: H_P1 = Γ^{H,P}(N_P, Q_P1) in pump convention
        # Need to convert derivative to network convention: ∂H/∂Q_1 = -∂H/∂Q_P1
        jac_P1 = self.pump.linearize_H(1.0, N_P, Q_P1, suter=True, eps_rel=self.eps_rel)
        Gamma_P1_N = jac_P1[1]       # ∂H_P1/∂N_P
        Gamma_P1_Q = -jac_P1[2]      # ∂H_P1/∂Q_1 (network) = -∂H_P1/∂Q_P1

        # Pump 2: H_P2 = Γ^{H,P}(N_P, Q_P2)
        jac_P2 = self.pump.linearize_H(1.0, N_P, Q_P2, suter=True, eps_rel=self.eps_rel)
        Gamma_P2_N = jac_P2[1]       # ∂H_P2/∂N_P
        Gamma_P2_Q = -jac_P2[2]      # ∂H_P2/∂Q_2 (network) = -∂H_P2/∂Q_P2

        # Step 3: Build Appendix B coefficients
        # α_T = 1 / Γ^{H,T}_Q
        # α_1 = 1 / Γ^{H,P1}_Q
        # α_2 = 1 / Γ^{H,P2}_Q
        alpha_T = 1.0 / Gamma_T_Q
        alpha_1 = 1.0 / Gamma_P1_Q
        alpha_2 = 1.0 / Gamma_P2_Q

        # β_1 = Γ^{H,P1}_N / Γ^{H,P1}_Q
        # β_2 = Γ^{H,P2}_N / Γ^{H,P2}_Q
        beta_1 = Gamma_P1_N / Gamma_P1_Q
        beta_2 = Gamma_P2_N / Gamma_P2_Q

        # γ_y = Γ^{H,T}_y / Γ^{H,T}_Q
        # γ_N = Γ^{H,T}_N / Γ^{H,T}_Q
        gamma_y = Gamma_T_y / Gamma_T_Q
        gamma_N = Gamma_T_N / Gamma_T_Q

        # r_1 = 2 * R_1 * |Q_1|
        # r_2 = 2 * R_2 * |Q_2|
        # r_34 = 2 * R_34 * |Q_T|
        r_1 = 2.0 * self.R_1 * abs(Q_1)
        r_2 = 2.0 * self.R_2 * abs(Q_2)
        r_34 = 2.0 * self.R_34 * abs(Q_T)

        # Step 4: Build matrices A, B, E per Appendix B
        # States: x = [δH_T, δH_P1, δH_P2]^T
        # 0 = A @ x + B @ u + E @ d
        # y = C @ x

        # Row 1: Mass balance (δQ_1 + δQ_2 - δQ_3 = 0)
        # Row 2: KVL loop 1 (turbine + pump 2)
        # Row 3: KVL loop 2 (turbine + pump 1)
        A = np.array([
            [-alpha_T, alpha_1, alpha_2],
            [1.0 + r_34 * alpha_T, 0.0, -1.0 + r_2 * alpha_2],
            [1.0 + r_34 * alpha_T, -1.0 + r_1 * alpha_1, 0.0],
        ])

        B = np.array([
            [-beta_1 - beta_2],
            [-r_2 * beta_2],
            [-r_1 * beta_1],
        ])

        E = np.array([
            [gamma_y, gamma_N],
            [-r_34 * gamma_y, -r_34 * gamma_N],
            [-r_34 * gamma_y, -r_34 * gamma_N],
        ])

        C = np.array([[1.0, 0.0, 0.0]])

        # Step 5: Compute static gain K_P = -C @ A^{-1} @ B
        try:
            A_inv = np.linalg.inv(A)
            K_P = float(-C @ A_inv @ B)
        except np.linalg.LinAlgError:
            K_P = float('nan')

        return {
            "G_u": K_P,
            # Partial derivatives (H-based, forward characteristic)
            "Gamma_T_y": Gamma_T_y,
            "Gamma_T_N": Gamma_T_N,
            "Gamma_T_Q": Gamma_T_Q,
            "Gamma_P1_N": Gamma_P1_N,
            "Gamma_P1_Q": Gamma_P1_Q,
            "Gamma_P2_N": Gamma_P2_N,
            "Gamma_P2_Q": Gamma_P2_Q,
            # Appendix B coefficients
            "alpha_T": alpha_T,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "gamma_y": gamma_y,
            "gamma_N": gamma_N,
            "r_1": r_1,
            "r_2": r_2,
            "r_34": r_34,
            # Matrices
            "A": A,
            "B": B,
            "E": E,
            "C": C,
            # Operating point (flows with mass balance enforced)
            "Q_T": Q_T,
            "Q_1": Q_1,
            "Q_2": Q_2,
            "operating_point": {
                "y_T": y_T,
                "N_T": N_T,
                "H_T": H_T,
                "N_P": N_P,
                "H_P1": H_P1,
                "H_P2": H_P2,
            },
        }

    def print_analysis(self, result: dict) -> None:
        """Pretty print the static gain analysis."""
        print("=" * 80)
        print("PF3 Static Gain Analysis (Appendix B, H-based forward formulation)")
        print("=" * 80)

        print("\nHydraulic Resistances:")
        print("-" * 80)
        print(f"  R_1 (pump 1 branch)    = {self.R_1:.4f} s²/m⁵")
        print(f"  R_2 (pump 2 branch)    = {self.R_2:.4f} s²/m⁵")
        print(f"  R_34 (turbine branch)  = {self.R_34:.4f} s²/m⁵")

        op = result["operating_point"]
        print("\nOperating Point:")
        print("-" * 80)
        print(f"  y_T  = {op['y_T']:10.4f}")
        print(f"  N_T  = {op['N_T']:10.4f} rpm")
        print(f"  H_T  = {op['H_T']:10.4f} m")
        print(f"  N_P  = {op['N_P']:10.4f} rpm")
        print(f"  H_P1 = {op['H_P1']:10.4f} m")
        print(f"  H_P2 = {op['H_P2']:10.4f} m")

        print("\nComputed Discharges (mass balance enforced: Q_T = Q_1 + Q_2):")
        print("-" * 80)
        print(f"  Q_1  = {result['Q_1']:10.6f} m³/s  (pump 1, network convention)")
        print(f"  Q_2  = {result['Q_2']:10.6f} m³/s  (pump 2, network convention)")
        print(f"  Q_T  = {result['Q_T']:10.6f} m³/s  (turbine, = Q_1 + Q_2)")

        print("\nH-based Partial Derivatives (∂H/∂..., forward characteristic):")
        print("-" * 80)
        print(f"  Γ^T_y  = ∂H_T/∂y   = {result['Gamma_T_y']:10.6f} m")
        print(f"  Γ^T_N  = ∂H_T/∂N_T = {result['Gamma_T_N']:10.6e} m/rpm")
        print(f"  Γ^T_Q  = ∂H_T/∂Q_T = {result['Gamma_T_Q']:10.6f} m/(m³/s)")
        print(f"  Γ^P1_N = ∂H_P1/∂N  = {result['Gamma_P1_N']:10.6e} m/rpm")
        print(f"  Γ^P1_Q = ∂H_P1/∂Q_1= {result['Gamma_P1_Q']:10.6f} m/(m³/s)")
        print(f"  Γ^P2_N = ∂H_P2/∂N  = {result['Gamma_P2_N']:10.6e} m/rpm")
        print(f"  Γ^P2_Q = ∂H_P2/∂Q_2= {result['Gamma_P2_Q']:10.6f} m/(m³/s)")

        print("\nAppendix B Coefficients:")
        print("-" * 80)
        print(f"  α_T  = 1/Γ^T_Q       = {result['alpha_T']:10.6f}")
        print(f"  α_1  = 1/Γ^P1_Q      = {result['alpha_1']:10.6f}")
        print(f"  α_2  = 1/Γ^P2_Q      = {result['alpha_2']:10.6f}")
        print(f"  β_1  = Γ^P1_N/Γ^P1_Q = {result['beta_1']:10.6e}")
        print(f"  β_2  = Γ^P2_N/Γ^P2_Q = {result['beta_2']:10.6e}")
        print(f"  γ_y  = Γ^T_y/Γ^T_Q   = {result['gamma_y']:10.6f}")
        print(f"  γ_N  = Γ^T_N/Γ^T_Q   = {result['gamma_N']:10.6e}")
        print(f"  r_1  = 2R_1|Q_1|     = {result['r_1']:10.6f}")
        print(f"  r_2  = 2R_2|Q_2|     = {result['r_2']:10.6f}")
        print(f"  r_34 = 2R_34|Q_T|    = {result['r_34']:10.6f}")

        print("\n" + "=" * 80)
        print(f"STATIC GAIN:  K_P = {result['G_u']:10.6e} m/rpm")
        print("=" * 80)
        print(
            f"\nInterpretation: A change of 1 rpm in pump speed causes a "
            f"{result['G_u']:.6e} m change in turbine head."
        )
