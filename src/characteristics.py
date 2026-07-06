import math
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.optimize import fsolve

from src.structured_tri_interplator import (
    StructuredTriInterpolator,
    find_triangle_and_interpolate,
)


class Characteristics:
    def __init__(
        self,
        d_ref,
        h_n,
        q_n,
        t_n,
        n_n,
        char_file: Path,
        interp_method: Literal["linear", "clough_tocher", "structured"] = "structured",
    ) -> None:
        """
        Initialize pump-turbine model.

        Args:
            d_ref: Reference diameter [m]
            h_n: Rated head [m]
            q_n: Rated discharge [m³/s]
            t_n: Rated torque [Nm]
            n_n: Rated rotational speed [rpm]
            yn11q11t11_file: Path to characteristic curve file
            interp_method: Interpolation method for Suter transform.
                - "linear": LinearNDInterpolator with Delaunay (C0 continuous, faster)
                - "clough_tocher": CloughTocher2DInterpolator (C1 continuous, smoother)
                - "structured": Triangulation respecting iso-y curves (physically meaningful)
        """
        self.d_ref = d_ref
        self.h_n = h_n
        self.q_n = q_n
        self.t_n = t_n
        self.n_n = n_n
        self.interp_method = interp_method

        self.rho = 1000
        self.g = 9.81

        # Parse characteristic file: read everything inside <List>...</List>
        data_lines = []
        in_list = False
        with open(char_file, "r") as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped == "<List>":
                    in_list = True
                    continue
                if line_stripped == "</List>":
                    break
                if in_list and line_stripped and not line_stripped.startswith(";"):
                    data_lines.append(line_stripped)

        if not data_lines:
            raise ValueError(
                f"No data found in {char_file}. Expected <List>...</List> block."
            )

        data = np.loadtxt(data_lines)

        self.y_data = data[:, 0]
        self.n11_data = data[:, 1]
        self.q11_data = data[:, 2]
        self.t11_data = data[:, 3]

        # Build standard (y, N11) interpolators
        self.yn11q11 = LinearNDInterpolator(
            points=list(zip(self.y_data, self.n11_data)),
            values=self.q11_data,
        )
        self.yn11t11 = LinearNDInterpolator(
            points=list(zip(self.y_data, self.n11_data)), values=self.t11_data
        )

        # Build Suter transform interpolators
        self._build_suter_interpolators()

    @property
    def beta_bar(self) -> float:
        """
        Blade angle at runner outlet [rad], computed from BEP nominal values.

        At BEP with no residual swirl (Cu ≈ 0), the outlet flow angle equals
        the blade angle:
            β̄ = atan(Cm / U)

        where:
            Cm = Q_n / A_bar       (meridional velocity)
            U = ω_n × D_ref/2      (peripheral velocity)
            A_bar = π/4 × D_ref²   (outlet area, assuming circular)
            ω_n = N_n × π/30       (angular velocity)

        This is a geometric property of the runner, constant for all conditions.
        """
        # Outlet area (assuming circular with diameter = d_ref)
        A_bar = math.pi / 4 * self.d_ref**2

        # Meridional velocity at BEP
        Cm_bar = self.q_n / A_bar

        # Angular velocity at BEP [rad/s]
        omega_n = self.n_n * math.pi / 30

        # Peripheral velocity at outlet
        U_bar = omega_n * self.d_ref / 2

        # Blade angle
        return math.atan(Cm_bar / U_bar)

    @property
    def beta_bar_deg(self) -> float:
        """Blade angle at runner outlet [degrees]."""
        return math.degrees(self.beta_bar)

    def swirl_number(self, N: float, Q: float) -> float:
        """
        Compute swirl number S at runner outlet for given speed and flow.

        The swirl number quantifies the ratio of tangential to axial momentum:
            S = U/Cm - cot(β̄)

        where:
            U = ω × D_ref/2      (peripheral velocity)
            Cm = Q / A_bar       (meridional velocity)
            β̄ = blade angle at outlet (fixed geometry)

        Simplified:
            S = (π² × D_ref³ × N) / (240 × Q) - cot(β̄)

        Args:
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]

        Returns:
            S: Swirl number [-]
               S = 0: No swirl (BEP condition at outlet)
               S > 0: Co-rotation swirl (part load)
               S < 0: Counter-rotation swirl (overload)

        Note:
            H is not needed directly - swirl depends on N/Q ratio.
            However, Q = f(y, N, H) from the characteristic.
        """
        if abs(Q) < 1e-12:
            return float("inf") if N > 0 else float("-inf")

        # Outlet area
        A_bar = math.pi / 4 * self.d_ref**2

        # Velocities
        omega = N * math.pi / 30  # [rad/s]
        U = omega * self.d_ref / 2  # peripheral velocity
        Cm = Q / A_bar  # meridional velocity

        # Swirl number
        cot_beta = 1.0 / math.tan(self.beta_bar)
        S = U / Cm - cot_beta

        return S

    def N11(self, N, H):
        return (N * self.d_ref) / np.sqrt(H)

    def N(self, N11, H):
        return N11 * np.sqrt(H) / self.d_ref

    def Q11(self, Q, H):
        return Q / (self.d_ref * self.d_ref * np.sqrt(H))

    def Q(self, Q11, H):
        return Q11 * (self.d_ref**2) * np.sqrt(H)

    def T11(self, T, H):
        return T / (self.d_ref * self.d_ref * self.d_ref * H)

    def T(self, T11, H):
        return T11 * (self.d_ref**3) * (H)

    def N11_from_Q11S(self, Q11, S):
        """N11 for given Q11 and swirl S. From: N11/Q11 = (240/π²)(S + cot(β))"""
        return Q11 * (240 / (math.pi**2)) * (S + 1 / math.tan(self.beta_bar))

    def Q11_from_N11S(self, N11, S):
        """Q11 for given N11 and swirl S. From: N11/Q11 = (240/π²)(S + cot(β))"""
        return N11 / ((240 / (math.pi**2)) * (S + 1 / math.tan(self.beta_bar)))

    def H_from_Q11Q(self, Q11, Q):
        return (Q / (Q11 * self.d_ref**2)) ** 2

    def H_from_N11N(self, N11, N):
        return (N * self.d_ref / N11) ** 2

    def N_from_yHS(self, y, H, S_target):
        """
        Find rotational speed N for given guide vane opening, head, and swirl.

        Solves for N such that the operating point (y, N, H) produces swirl S_target.

        Args:
            y: Guide vane opening [-]
            H: Head [m]
            S_target: Target swirl number [-]

        Returns:
            N: Rotational speed [rpm]
        """
        from scipy.optimize import brentq

        def residual(N):
            Q, _ = self.QT_from_yNH(y, N, H)
            S_calc = self.swirl_number(N, Q)
            return S_calc - S_target

        # Search bounds: use nominal speed as reference
        N_min = self.n_n * 0.1
        N_max = self.n_n * 3.0

        return brentq(residual, N_min, N_max)

    def y_from_NQH(self, N, Q, H):
        """
        Find guide vane opening y for given operating point (N, Q, H).

        Solves for y such that QT_from_yNH(y, N, H) = Q.

        Args:
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]
            H: Head [m]

        Returns:
            y: Guide vane opening [-]
        """
        from scipy.optimize import brentq

        def residual(y):
            Q_calc, _ = self.QT_from_yNH(y, N, H)
            return Q_calc - Q

        # Search in valid y range from characteristic data
        y_min = float(np.min(self.y_data))
        y_max = float(np.max(self.y_data))

        return brentq(residual, y_min, y_max)

    @property
    def _turbine_mask(self):
        """Boolean mask for turbine quadrant (N11 > 0, Q11 > 0, T11 > 0)."""
        return (self.n11_data > 0) & (self.q11_data > 0) & (self.t11_data > 0)

    def _eta_all(self):
        """Compute efficiency for all data points (all quadrants)."""
        return (
            np.pi
            * self.n11_data
            * self.t11_data
            / (30 * self.rho * self.g * self.q11_data)
        )

    def eta_turb(self):
        """Return efficiency values for turbine quadrant only."""
        return self._eta_all()[self._turbine_mask]

    def bep_turb_at_head(self, H):
        """
        Return the [efficiency, N, Q, T, y] at the bep of turbine mode.
        """
        bep11 = self.bep_turb()
        return np.array(
            [
                bep11[0],
                self.N(bep11[1], H),
                self.Q(bep11[2], H),
                self.T(bep11[3], H),
                bep11[4],
            ]
        )

    def bep_turb(self):
        """
        Return the [efficiency, n11, q11, t11, y] at the bep of turbine mode.
        """
        eta = self._eta_all()
        chart = np.vstack(
            (eta, self.n11_data, self.q11_data, self.t11_data, self.y_data)
        )
        chart = chart[:, self._turbine_mask]
        best_idx = np.argmax(chart[0])
        return chart[:, best_idx]

    def _build_suter_interpolators(self):
        """
        Convert characteristic curves to Suter coordinates and build interpolators.

        - Converts (y, N11, Q11, T11) to (y, theta, WB, WH)
        - Suter coordinates are independent of operating head H
        """
        # Compute BEP values in unit parameter space (Nicolet thesis eq. 5.38-5.43)
        self.q_11_n = self.Q11(self.q_n, self.h_n)  # Q11BEP = q_n/(d_ref²√h_n)
        self.n_11_n = self.N11(self.n_n, self.h_n)  # N11BEP = n_n·d_ref/√h_n
        self.t_11_n = self.T11(self.t_n, self.h_n)  # T11BEP = t_n/(d_ref³·h_n)

        # Compute theta from Nicolet equation 5.38:
        # θ = atan2(υ, α) where υ = Q11/Q11BEP, α = N11/N11BEP
        n11_normalized = self.n11_data / self.n_11_n
        q11_normalized = self.q11_data / self.q_11_n

        theta_data = np.arctan2(q11_normalized, n11_normalized)

        # Compute WH and WB from Suter formulas (Nicolet eq. 5.42, 5.43):
        # W_H = 1 / ((Q11/Q11BEP)² + (N11/N11BEP)²)
        # W_B = WH * T11/T11BEP
        denom = np.square(self.q11_data / self.q_11_n) + np.square(
            self.n11_data / self.n_11_n
        )
        valid_mask = denom > 1e-10

        if not np.sum(np.invert(valid_mask)) == 0:
            raise ValueError(
                "this characteristic file has data too close to the 0,0 point"
            )

        wh_data = np.zeros_like(self.t11_data)
        wh_data = 1.0 / denom

        wb_data = np.zeros_like(self.t11_data)
        wb_data = wh_data * self.t11_data / self.t_11_n

        # Store Suter data for debugging/analysis
        self.theta_data = theta_data
        self.wb_data = wb_data
        self.wh_data = wh_data

        # Create interpolators in (y, theta) space
        # Theta is circular (-π to +π wraps around), but the interpolators
        # treat it as linear. To handle boundary cases, duplicate points near ±π to
        # the opposite side so the interpolator sees continuity across the wrap.
        wrap_threshold = np.pi * 0.8  # duplicate points beyond this

        # Find points near +π and -π boundaries
        near_pos_pi = theta_data > wrap_threshold
        near_neg_pi = theta_data < -wrap_threshold

        # Compute efficiency for all points
        eta_data = self._eta_all()

        # Duplicate points: shift theta by ±2π
        y_extended = self.y_data.copy()
        theta_extended = theta_data.copy()
        wh_extended = wh_data.copy()
        wb_extended = wb_data.copy()
        n11_extended = self.n11_data.copy()
        eta_extended = eta_data.copy()

        if np.any(near_pos_pi):
            # Points near +π: also add them at theta - 2π
            y_extended = np.concatenate([y_extended, self.y_data[near_pos_pi]])
            theta_extended = np.concatenate(
                [theta_extended, theta_data[near_pos_pi] - 2 * np.pi]
            )
            wh_extended = np.concatenate([wh_extended, wh_data[near_pos_pi]])
            wb_extended = np.concatenate([wb_extended, wb_data[near_pos_pi]])
            n11_extended = np.concatenate([n11_extended, self.n11_data[near_pos_pi]])
            eta_extended = np.concatenate([eta_extended, eta_data[near_pos_pi]])

        if np.any(near_neg_pi):
            # Points near -π: also add them at theta + 2π
            y_extended = np.concatenate([y_extended, self.y_data[near_neg_pi]])
            theta_extended = np.concatenate(
                [theta_extended, theta_data[near_neg_pi] + 2 * np.pi]
            )
            wh_extended = np.concatenate([wh_extended, wh_data[near_neg_pi]])
            wb_extended = np.concatenate([wb_extended, wb_data[near_neg_pi]])
            n11_extended = np.concatenate([n11_extended, self.n11_data[near_neg_pi]])
            eta_extended = np.concatenate([eta_extended, eta_data[near_neg_pi]])

        points = np.column_stack([y_extended, theta_extended])

        # store extended data
        self.y_data_ext = y_extended
        self.theta_data_ext = theta_extended
        self.wb_data_ext = wb_extended
        self.wh_data_ext = wh_extended
        self.n11_data_ext = n11_extended
        self.eta_data_ext = eta_extended

        if self.interp_method == "clough_tocher":
            self.interp_wb = CloughTocher2DInterpolator(points, wb_extended)
            self.interp_wh = CloughTocher2DInterpolator(points, wh_extended)
            self.interp_n11 = CloughTocher2DInterpolator(points, n11_extended)
            self.interp_eta = CloughTocher2DInterpolator(points, eta_extended)
        elif self.interp_method == "linear":
            self.interp_wb = LinearNDInterpolator(points, wb_extended)
            self.interp_wh = LinearNDInterpolator(points, wh_extended)
            self.interp_n11 = LinearNDInterpolator(points, n11_extended)
            self.interp_eta = LinearNDInterpolator(points, eta_extended)
        else:  # "structured" (default)
            self.interp_wh = StructuredTriInterpolator(points, wh_extended)
            self.interp_wb = StructuredTriInterpolator(points, wb_extended)
            self.interp_n11 = StructuredTriInterpolator(points, n11_extended)
            self.interp_eta = StructuredTriInterpolator(points, eta_extended)

        # Cache numba kernel args for fast scalar lookup (structured only)
        self._use_fast_scalar = self.interp_method == "structured"
        if self._use_fast_scalar:
            interp = self.interp_wh
            self._wh_tri_vertices = interp._tri_vertices
            self._wh_tri_y = interp._tri_y
            self._wh_tri_theta = interp._tri_theta
            self._wh_values = interp._values
            self._wh_strip_starts = interp._strip_starts
            self._wh_y_levels = interp._y_levels

        # Store theta range for bounded searches
        self.theta_min = np.min(theta_data)
        self.theta_max = np.max(theta_data)

    def _find_theta(self, y, alpha, h, theta_init=None):
        r"""
        Find theta such that the system of equations holds:

        $$
        \begin{cases}
        \theta = \tan^{-1}\left(\frac{Q/Q_{BEP}}{N/N_{BEP}}\right) \\[10pt]
        W_H = \frac{H/H_{BEP}}{(Q/Q_{BEP})^2 + (N/N_{BEP})^2} \\[10pt]
        W_H = I_H(y, \theta)
        \end{cases}
        $$

        Eliminating $Q$ and $W_H$, this function solves the equation: (and 1 + tan²x = cos²x)

        $$
        0 = H/H_{BEP} - I_H(y,\theta)\cdot(N/N_{BEP})^2 \cdot \frac{1}{(\cos^2(\theta)}
        $$

        Args:
            y: Guide vane opening
            alpha: Normalized speed alpha = N/N_n
            h: Normalized head h = H/H_n

        Returns:
            theta value which satisfies the current condition.
        """

        def equation(theta_arr):
            theta = theta_arr[0]
            wh = self.interp_wh([[y, theta]])[0]
            cos_theta = np.cos(theta)

            # Handle NaN when (y, theta) is outside interpolation convex hull
            if np.isnan(wh):
                return 1e10  # Large positive value to guide solver away

            # Handle singularity at theta = ±π/2
            if abs(cos_theta) < 1e-10:
                return 1e10

            return h - wh * alpha**2 / cos_theta**2

        if theta_init is None:
            # Initial guess
            n11 = alpha * self.n_n * self.d_ref / np.sqrt(h * self.h_n)
            q11_approx = self.yn11q11([[y, n11]])[0]

            # Convert Q11 to Q, then to upsilon: Q = Q11 * D² * √H
            q_approx = q11_approx * self.d_ref**2 * np.sqrt(h * self.h_n)
            upsilon_approx = q_approx / self.q_n
            theta_init = np.arctan2(upsilon_approx, alpha)

            # Fallback to simple guess if standard method fails
            if np.isnan(theta_init):
                theta_init = np.pi / 4 if alpha > 0 else np.pi / 6

        theta_solution = fsolve(equation, [theta_init])[0]
        return theta_solution

    def QT_from_yNH(self, y, N, H):
        """
        Compute both Q and T using Suter transform.

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            H: Head [m]

        Returns:
            (Q, T): Discharge [m³/s] and Torque [Nm]
        """
        # Compute normalized parameters
        alpha = N / self.n_n
        h = H / self.h_n

        # Find theta that satisfies the Suter equations
        theta = self._find_theta(y, alpha, h)

        # Look up WB(y, theta)
        wb = self.interp_wb([[y, theta]])[0]

        # Compute alpha² + upsilon² and upsilon
        cos_theta = np.cos(theta)
        alpha_sq_plus_upsilon_sq = alpha**2 / cos_theta**2
        upsilon = alpha * np.tan(theta)

        # Compute beta
        beta = wb * alpha_sq_plus_upsilon_sq

        # Convert back to physical units
        Q = upsilon * self.q_n
        T = beta * self.t_n

        return Q, T

    def linearize_Q(self, y, N, H, suter=False, eps_rel=2e-2):
        """
        Compute the linearization (Jacobian) of Q(y,N,H) using numerical differentiation.

        Uses relatively large epsilon to average over multiple data regions, providing
        smoother gradients suitable for control applications.

        Returns the partial derivatives [∂Q/∂y, ∂Q/∂N, ∂Q/∂H] evaluated at (y, N, H).

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            H: Head [m]
            suter: Use Suter transform (default: False)
            eps_rel: Relative epsilon for numerical differentiation (default: 2e-2)

        Returns:
            jacobian: Array [∂Q/∂y, ∂Q/∂N, ∂Q/∂H] in units [m³/s, m³/s/rpm, m³/s/m]
        """
        # Central difference with larger epsilon to smooth out local interpolation errors
        eps_y = abs(y) * eps_rel if y != 0 else eps_rel
        eps_N = abs(N) * eps_rel if N != 0 else 1.0
        eps_H = abs(H) * eps_rel if H != 0 else 0.01

        # ∂Q/∂y
        Q_y_plus, _ = self.QT_from_yNH(y + eps_y, N, H)
        Q_y_minus, _ = self.QT_from_yNH(y - eps_y, N, H)
        dQ_dy = (Q_y_plus - Q_y_minus) / (2 * eps_y)

        # ∂Q/∂N
        Q_N_plus, _ = self.QT_from_yNH(y, N + eps_N, H)
        Q_N_minus, _ = self.QT_from_yNH(y, N - eps_N, H)
        dQ_dN = (Q_N_plus - Q_N_minus) / (2 * eps_N)

        # ∂Q/∂H
        Q_H_plus, _ = self.QT_from_yNH(y, N, H + eps_H)
        Q_H_minus, _ = self.QT_from_yNH(y, N, H - eps_H)
        dQ_dH = (Q_H_plus - Q_H_minus) / (2 * eps_H)

        return np.array([dQ_dy, dQ_dN, dQ_dH])

    def linearize_T(self, y, N, H, suter=False, eps_rel=2e-2):
        """
        Compute the linearization (Jacobian) of T(y,N,H) using numerical differentiation.

        Uses relatively large epsilon to average over multiple data regions, providing
        smoother gradients suitable for control applications.

        Returns the partial derivatives [∂T/∂y, ∂T/∂N, ∂T/∂H] evaluated at (y, N, H).

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            H: Head [m]
            suter: Use Suter transform (default: False)
            eps_rel: Relative epsilon for numerical differentiation (default: 2e-2)

        Returns:
            jacobian: Array [∂T/∂y, ∂T/∂N, ∂T/∂H] in units [Nm, Nm/rpm, Nm/m]
        """
        # Central difference with larger epsilon to smooth out local interpolation errors
        eps_y = abs(y) * eps_rel if y != 0 else eps_rel
        eps_N = abs(N) * eps_rel if N != 0 else 1.0
        eps_H = abs(H) * eps_rel if H != 0 else 0.01

        # ∂T/∂y
        _, T_y_plus = self.QT_from_yNH(y + eps_y, N, H)
        _, T_y_minus = self.QT_from_yNH(y - eps_y, N, H)
        dT_dy = (T_y_plus - T_y_minus) / (2 * eps_y)

        # ∂T/∂N
        _, T_N_plus = self.QT_from_yNH(y, N + eps_N, H)
        _, T_N_minus = self.QT_from_yNH(y, N - eps_N, H)
        dT_dN = (T_N_plus - T_N_minus) / (2 * eps_N)

        # ∂T/∂H
        _, T_H_plus = self.QT_from_yNH(y, N, H + eps_H)
        _, T_H_minus = self.QT_from_yNH(y, N, H - eps_H)
        dT_dH = (T_H_plus - T_H_minus) / (2 * eps_H)

        return np.array([dT_dy, dT_dN, dT_dH])

    def compute_H(self, y, N, Q, suter=True):
        """
        Compute head H given guide vane opening y, speed N, and discharge Q.

        This is the inverse of QT_from_yNH: given Q, solve for H analytically
        (no root-finding needed).

        For Suter method:
            θ = arctan2(υ, α) where α = N/N_n, υ = Q/Q_n
            W_H = interp_wh(y, θ)
            H = W_H × H_n × (α² + υ²)

        For standard method:
            θ = arctan2(υ, α)
            N11 = interp_n11(y, θ)
            H = (N × D / N11)²

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]
            suter: Use Suter transform (default: True)

        Returns:
            H: Head [m]
        """
        # Compute normalized parameters
        alpha = N / self.n_n
        upsilon = Q / self.q_n

        # Compute theta directly from (N, Q) - no iteration needed!
        # Use math.atan2 for scalar (avoids numpy array overhead)
        theta = math.atan2(upsilon, alpha)

        if suter:
            # Suter method: H = W_H × H_n × (α² + υ²)
            if self._use_fast_scalar:
                # Call numba kernel directly, bypassing Python wrapper
                wh = find_triangle_and_interpolate(
                    y,
                    theta,
                    self._wh_tri_vertices,
                    self._wh_tri_y,
                    self._wh_tri_theta,
                    self._wh_values,
                    self._wh_strip_starts,
                    self._wh_y_levels,
                )
            else:
                wh = self.interp_wh([[y, theta]])[0]

            if math.isnan(wh):
                raise ValueError(
                    f"Point (y={y}, theta={theta:.4f}) outside interpolation domain"
                )

            H = wh * self.h_n * (alpha**2 + upsilon**2)
        else:
            # Standard method: look up N11, then H = (N × D / N11)²
            n11 = self.interp_n11([[y, theta]])[0]

            if np.isnan(n11):
                raise ValueError(
                    f"Point (y={y}, theta={theta:.4f}) outside interpolation domain"
                )

            # Avoid division by zero
            if abs(n11) < 1e-10:
                raise ValueError(f"N11 ≈ 0 at (y={y}, theta={theta:.4f})")

            H = (N * self.d_ref / n11) ** 2

        return H

    def HT_from_yNQ(self, y, N, Q, suter=True):
        """
        Compute both H and T given guide vane opening y, speed N, and discharge Q.

        This is the H-based formulation that avoids problematic ∂Q/∂N derivatives.

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]
            suter: Use Suter transform (default: True)

        Returns:
            (H, T): Head [m] and Torque [Nm]
        """
        # Compute normalized parameters
        alpha = N / self.n_n
        upsilon = Q / self.q_n

        # Compute theta directly (arctan2 for 4-quadrant)
        theta = np.arctan2(upsilon, alpha)

        if suter:
            # Look up W_H and W_B
            wh = self.interp_wh([[y, theta]])[0]
            wb = self.interp_wb([[y, theta]])[0]

            if np.isnan(wh) or np.isnan(wb):
                raise ValueError(
                    f"Point (y={y}, theta={theta:.4f}) outside interpolation domain"
                )

            alpha_sq_plus_upsilon_sq = alpha**2 + upsilon**2

            # Compute H and T from Suter formulas
            H = wh * self.h_n * alpha_sq_plus_upsilon_sq
            T = wb * self.t_n * alpha_sq_plus_upsilon_sq
        else:
            # Standard method: look up N11 and T11
            n11 = self.interp_n11([[y, theta]])[0]

            if np.isnan(n11) or abs(n11) < 1e-10:
                raise ValueError(f"Invalid N11 at (y={y}, theta={theta:.4f})")

            H = (N * self.d_ref / n11) ** 2

            # For T, use the standard yn11t11 interpolator
            t11 = self.yn11t11([[y, n11]])[0]
            T = self.T(t11, H)

        return H, T

    def linearize_H(self, y, N, Q, suter=True, eps_rel=2e-2):
        """
        Compute the linearization (Jacobian) of H(y,N,Q) using numerical differentiation.

        This is for the H-based formulation: g(y, N, Q) → H

        Returns the partial derivatives [∂H/∂y, ∂H/∂N, ∂H/∂Q] evaluated at (y, N, Q).

        Note: For pumps where y=1.0 (fully open, constant), set ∂H/∂y = 0 since
        perturbing y would go outside the data domain. The pump characteristic
        is g_P(N, Q) → H with no y dependence.

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            Q: Discharge [m³/s]
            suter: Use Suter transform (default: True)
            eps_rel: Relative epsilon for numerical differentiation (default: 2e-2)

        Returns:
            jacobian: Array [∂H/∂y, ∂H/∂N, ∂H/∂Q] in units [m, m/rpm, m/(m³/s)]
        """
        # Central difference with larger epsilon to smooth out interpolation errors
        eps_y = abs(y) * eps_rel if y != 0 else eps_rel
        eps_N = abs(N) * eps_rel if N != 0 else 1.0
        eps_Q = abs(Q) * eps_rel if Q != 0 else 0.001

        # Get y data bounds
        y_max = np.max(self.y_data)
        y_min = np.min(self.y_data)

        # ∂H/∂y - handle boundary cases (e.g., pumps at y=1.0)
        if y + eps_y > y_max:
            # At upper boundary - use backward difference or set to 0
            if y - eps_y >= y_min:
                H_0 = self.compute_H(y, N, Q, suter=suter)
                H_y_minus = self.compute_H(y - eps_y, N, Q, suter=suter)
                dH_dy = (H_0 - H_y_minus) / eps_y
            else:
                # y is at both boundaries (very narrow range) - set to 0
                dH_dy = 0.0
        elif y - eps_y < y_min:
            # At lower boundary - use forward difference
            H_0 = self.compute_H(y, N, Q, suter=suter)
            H_y_plus = self.compute_H(y + eps_y, N, Q, suter=suter)
            dH_dy = (H_y_plus - H_0) / eps_y
        else:
            # Central difference
            H_y_plus = self.compute_H(y + eps_y, N, Q, suter=suter)
            H_y_minus = self.compute_H(y - eps_y, N, Q, suter=suter)
            dH_dy = (H_y_plus - H_y_minus) / (2 * eps_y)

        # ∂H/∂N
        H_N_plus = self.compute_H(y, N + eps_N, Q, suter=suter)
        H_N_minus = self.compute_H(y, N - eps_N, Q, suter=suter)
        dH_dN = (H_N_plus - H_N_minus) / (2 * eps_N)

        # ∂H/∂Q
        H_Q_plus = self.compute_H(y, N, Q + eps_Q, suter=suter)
        H_Q_minus = self.compute_H(y, N, Q - eps_Q, suter=suter)
        dH_dQ = (H_Q_plus - H_Q_minus) / (2 * eps_Q)

        return np.array([dH_dy, dH_dN, dH_dQ])

    def NT_from_QH(self, Q, H):
        """
        Compute N and T given discharge and head (y fixed to 1).

        Useful for pumps where guide vane is fully open (y=1).
        Uses direct 1D interpolation on the y=1 characteristic data.

        Args:
            Q: Discharge [m³/s]
            H: Head [m]

        Returns:
            (N, T): Rotational speed [rpm] and Torque [Nm]
        """
        # Convert to unit parameters
        q11 = self.Q11(Q, H)

        # Get y=1 data slice (use isclose for floating point comparison)
        mask = np.isclose(self.y_data, 1.0, rtol=1e-3)
        if not np.any(mask):
            raise ValueError(
                f"No y=1 data found. y values in data: {np.unique(self.y_data)}"
            )

        q11_y1 = self.q11_data[mask]
        n11_y1 = self.n11_data[mask]
        t11_y1 = self.t11_data[mask]

        # Sort by Q11 for interpolation
        sort_idx = np.argsort(q11_y1)
        q11_y1 = q11_y1[sort_idx]
        n11_y1 = n11_y1[sort_idx]
        t11_y1 = t11_y1[sort_idx]

        # Interpolate N11 and T11
        n11 = np.interp(q11, q11_y1, n11_y1)
        t11 = np.interp(q11, q11_y1, t11_y1)

        # Convert back to physical units
        N = self.N(n11, H)
        T = self.T(t11, H)

        return N, T

    def compute_Q_from_H(self, y, N, H, suter=True, Q_guess=None):
        """
        Inverse characteristic: given (y, N, H), find Q such that compute_H(y, N, Q) = H.

        Uses Brent's method for robust root-finding with safe handling of
        out-of-domain points.

        Args:
            y: Guide vane opening [1]
            N: Rotational speed [rpm]
            H: Head [m]
            suter: Use Suter transform (default: True)
            Q_guess: Initial guess for Q (optional, used to determine search direction)

        Returns:
            Q: Discharge [m³/s] such that compute_H(y, N, Q) ≈ H
        """
        from scipy.optimize import brentq

        # Large penalty for out-of-domain points (guides root-finder away)
        LARGE_RESIDUAL = 1e6

        def safe_residual(Q):
            """Residual that returns large value for out-of-domain instead of raising."""
            try:
                return self.compute_H(y, N, Q, suter=suter) - H
            except ValueError:
                # Out of interpolation domain - return large residual
                # Sign based on whether Q is too large or too small
                return LARGE_RESIDUAL if Q > 0 else -LARGE_RESIDUAL

        # Try progressively wider search intervals
        # Start small since operating Q may be much smaller than nominal Q_n
        Q_center = Q_guess if Q_guess is not None else 0.0

        # Search intervals: start small, expand outward
        search_ranges = [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,  # Small absolute values
            self.q_n * 0.1,
            self.q_n * 0.2,
            self.q_n * 0.5,  # Fractions of nominal
            self.q_n,
            self.q_n * 1.5,
            self.q_n * 2.0,  # Up to 2x nominal
        ]

        for half_width in search_ranges:
            Q_lo = Q_center - half_width
            Q_hi = Q_center + half_width

            f_lo = safe_residual(Q_lo)
            f_hi = safe_residual(Q_hi)

            # Skip if both are out-of-domain penalties
            if abs(f_lo) >= LARGE_RESIDUAL or abs(f_hi) >= LARGE_RESIDUAL:
                continue

            # Check for sign change (solution exists in bracket)
            if f_lo * f_hi < 0:
                # Found valid bracket - solve with safe residual
                Q_solution = brentq(safe_residual, Q_lo, Q_hi, xtol=1e-9)
                return Q_solution

        # Try asymmetric search (solution might be on one side of Q_center)
        for half_width in search_ranges:
            # Try negative side only
            Q_lo, Q_hi = -half_width * 2, 0
            f_lo, f_hi = safe_residual(Q_lo), safe_residual(Q_hi)
            if (
                abs(f_lo) < LARGE_RESIDUAL
                and abs(f_hi) < LARGE_RESIDUAL
                and f_lo * f_hi < 0
            ):
                return brentq(safe_residual, Q_lo, Q_hi, xtol=1e-9)

            # Try positive side only
            Q_lo, Q_hi = 0, half_width * 2
            f_lo, f_hi = safe_residual(Q_lo), safe_residual(Q_hi)
            if (
                abs(f_lo) < LARGE_RESIDUAL
                and abs(f_hi) < LARGE_RESIDUAL
                and f_lo * f_hi < 0
            ):
                return brentq(safe_residual, Q_lo, Q_hi, xtol=1e-9)

        # Last resort: report failure with diagnostic info
        diag_Qs = [0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, self.q_n, -self.q_n]
        diag_info = []
        for Q_test in diag_Qs:
            try:
                H_test = self.compute_H(y, N, Q_test, suter=suter)
                diag_info.append(f"Q={Q_test:.3f}->H={H_test:.2f}")
            except ValueError:
                diag_info.append(f"Q={Q_test:.3f}->OUT")

        raise ValueError(
            f"Cannot find Q for (y={y}, N={N}, H_target={H}). "
            f"Diagnostics: {', '.join(diag_info)}"
        )
