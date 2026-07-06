"""
Hill chart plotting utilities for turbine characteristics.

Provides functions to plot efficiency and power hill charts in the (N11, Q11) plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.characteristics import Characteristics


def plot_efficiency_hillchart(
    char: Characteristics,
    n_grid: int = 100,
    eta_levels: int = 15,
    figsize: tuple = (12, 9),
    cmap: str = "RdYlGn",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """
    Plot the efficiency hill chart for a turbine.

    Plots Q11 vs N11 with efficiency contours and guide vane opening (y) isolines.

    Args:
        char: Characteristics instance with loaded turbine data
        n_grid: Number of grid points for interpolation (default: 100)
        eta_levels: Number of efficiency contour levels (default: 15)
        figsize: Figure size (width, height) in inches
        cmap: Colormap for efficiency contours (default: "RdYlGn")
        fig: Existing figure (optional)
        ax: Existing axes (optional)

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq

    # Get turbine quadrant data for plot bounds and iso-y lines
    mask = char._turbine_mask
    n11_turb = char.n11_data[mask]
    q11_turb = char.q11_data[mask]
    y_turb = char.y_data[mask]

    # Create regular grid in (N11, Q11) space
    n11_min, n11_max = n11_turb.min(), n11_turb.max()
    q11_min, q11_max = q11_turb.min(), q11_turb.max()

    n11_grid = np.linspace(n11_min, n11_max, n_grid)
    q11_grid = np.linspace(q11_min, q11_max, n_grid)
    N11_mesh, Q11_mesh = np.meshgrid(n11_grid, q11_grid)

    # Get y bounds for root finding
    y_min, y_max = char.y_data.min(), char.y_data.max()

    # Interpolate efficiency using (y, theta) space
    eta_mesh = np.full_like(N11_mesh, np.nan)

    for i in range(n_grid):
        for j in range(n_grid):
            n11 = N11_mesh[i, j]
            q11 = Q11_mesh[i, j]

            # Compute theta from (N11, Q11)
            theta = np.arctan2(q11 / char.q_11_n, n11 / char.n_11_n)

            # Find y such that interp_n11(y, theta) = n11
            def residual(y):
                n11_interp = char.interp_n11([[y, theta]])[0]
                if np.isnan(n11_interp):
                    return 1e10
                return n11_interp - n11

            try:
                # Check if solution exists in [y_min, y_max]
                f_lo = residual(y_min)
                f_hi = residual(y_max)

                if f_lo * f_hi < 0:
                    y_sol = brentq(residual, y_min, y_max, xtol=1e-6)
                    eta_mesh[i, j] = char.interp_eta([[y_sol, theta]])[0]
            except (ValueError, RuntimeError):
                pass  # Leave as NaN

    # Create figure
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot efficiency contours (filled)
    eta_valid = eta_mesh[~np.isnan(eta_mesh)]
    if len(eta_valid) == 0:
        raise ValueError("No valid eta values interpolated. Check data coverage.")
    eta_min = np.nanmin(eta_valid)
    eta_max = np.nanmax(eta_valid)
    levels = np.linspace(eta_min, eta_max, eta_levels)

    contourf = ax.contourf(
        N11_mesh,
        Q11_mesh,
        eta_mesh,
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    # Plot efficiency contour lines
    contour = ax.contour(
        N11_mesh,
        Q11_mesh,
        eta_mesh,
        levels=levels,
        colors="k",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # Plot guide vane opening (y) isolines
    unique_y = np.unique(y_turb)
    colors_y = plt.cm.Blues(np.linspace(0.3, 0.9, len(unique_y)))

    for i, y_val in enumerate(unique_y):
        y_mask = y_turb == y_val
        n11_y = n11_turb[y_mask]
        q11_y = q11_turb[y_mask]

        # Sort by N11 for smooth line
        sort_idx = np.argsort(n11_y)
        n11_y = n11_y[sort_idx]
        q11_y = q11_y[sort_idx]

        ax.plot(
            n11_y,
            q11_y,
            color=colors_y[i],
            linewidth=1.5,
            linestyle="--",
        )

        # Add y label at end of line
        if len(n11_y) > 0:
            ax.annotate(
                f"y={y_val:.2f}",
                xy=(n11_y[-1], q11_y[-1]),
                fontsize=7,
                color=colors_y[i],
                ha="left",
            )

    # Mark BEP point
    bep = char.bep_turb()
    eta_bep, n11_bep, q11_bep, t11_bep, y_bep = bep
    ax.plot(
        n11_bep,
        q11_bep,
        "r*",
        markersize=15,
        markeredgecolor="k",
        markeredgewidth=1,
        label=f"BEP (η={eta_bep:.3f})",
        zorder=10,
    )

    # Labels and title
    ax.set_xlabel("N11", fontsize=12)
    ax.set_ylabel("Q11", fontsize=12)
    ax.set_title("Turbine Efficiency Hill Chart", fontsize=14)

    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_power_hillchart(
    char: Characteristics,
    n_grid: int = 100,
    p11_levels: int = 15,
    figsize: tuple = (12, 9),
    cmap: str = "YlOrRd",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """
    Plot the power hill chart for a turbine.

    Plots Q11 vs N11 with power coefficient (P11) contours and guide vane opening (y) isolines.

    The power coefficient is defined as:
        P11 = T11 × N11 × (π/30)

    which is proportional to shaft power P = T × ω.

    Args:
        char: Characteristics instance with loaded turbine data
        n_grid: Number of grid points for interpolation (default: 100)
        p11_levels: Number of power contour levels (default: 15)
        figsize: Figure size (width, height) in inches
        cmap: Colormap for power contours (default: "YlOrRd")
        fig: Existing figure (optional)
        ax: Existing axes (optional)

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq

    # Get turbine quadrant data for plot bounds and iso-y lines
    mask = char._turbine_mask
    n11_turb = char.n11_data[mask]
    q11_turb = char.q11_data[mask]
    t11_turb = char.t11_data[mask]
    y_turb = char.y_data[mask]

    # Create regular grid in (N11, Q11) space
    n11_min, n11_max = n11_turb.min(), n11_turb.max()
    q11_min, q11_max = q11_turb.min(), q11_turb.max()

    n11_grid = np.linspace(n11_min, n11_max, n_grid)
    q11_grid = np.linspace(q11_min, q11_max, n_grid)
    N11_mesh, Q11_mesh = np.meshgrid(n11_grid, q11_grid)

    # Get y bounds for root finding
    y_min, y_max = char.y_data.min(), char.y_data.max()

    # Interpolate T11 and compute P11 using (y, theta) space
    p11_mesh = np.full_like(N11_mesh, np.nan)

    # Build T11 interpolator if not already present
    if not hasattr(char, "interp_t11"):
        _build_t11_interpolator(char)

    for i in range(n_grid):
        for j in range(n_grid):
            n11 = N11_mesh[i, j]
            q11 = Q11_mesh[i, j]

            # Compute theta from (N11, Q11)
            theta = np.arctan2(q11 / char.q_11_n, n11 / char.n_11_n)

            # Find y such that interp_n11(y, theta) = n11
            def residual(y):
                n11_interp = char.interp_n11([[y, theta]])[0]
                if np.isnan(n11_interp):
                    return 1e10
                return n11_interp - n11

            try:
                # Check if solution exists in [y_min, y_max]
                f_lo = residual(y_min)
                f_hi = residual(y_max)

                if f_lo * f_hi < 0:
                    y_sol = brentq(residual, y_min, y_max, xtol=1e-6)
                    t11 = char.interp_t11([[y_sol, theta]])[0]
                    if not np.isnan(t11):
                        p11_mesh[i, j] = t11 * n11 * (np.pi / 30)
            except (ValueError, RuntimeError):
                pass  # Leave as NaN

    # Get BEP power for normalization
    bep = char.bep_turb()
    _, n11_bep, q11_bep, t11_bep, _ = bep
    p11_bep = t11_bep * n11_bep * (np.pi / 30)

    # Convert to percentage of BEP
    p11_pct_mesh = (p11_mesh / p11_bep) * 100

    # Create figure
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot power contours (filled)
    p11_pct_valid = p11_pct_mesh[~np.isnan(p11_pct_mesh)]
    if len(p11_pct_valid) == 0:
        raise ValueError("No valid P11 values interpolated. Check data coverage.")
    p11_pct_min = np.nanmin(p11_pct_valid)
    p11_pct_max = np.nanmax(p11_pct_valid)

    # Create levels at 5% intervals, ensuring 100% is included
    level_min = np.floor(p11_pct_min / 10) * 10
    level_max = np.ceil(p11_pct_max / 10) * 10
    levels = np.arange(level_min, level_max + 10, 10)

    contourf = ax.contourf(
        N11_mesh,
        Q11_mesh,
        p11_pct_mesh,
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    # Plot power contour lines
    contour = ax.contour(
        N11_mesh,
        Q11_mesh,
        p11_pct_mesh,
        levels=levels,
        colors="k",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f%%")

    # Plot guide vane opening (y) isolines
    unique_y = np.unique(y_turb)
    colors_y = plt.cm.Blues(np.linspace(0.3, 0.9, len(unique_y)))

    for i, y_val in enumerate(unique_y):
        y_mask = y_turb == y_val
        n11_y = n11_turb[y_mask]
        q11_y = q11_turb[y_mask]

        # Sort by N11 for smooth line
        sort_idx = np.argsort(n11_y)
        n11_y = n11_y[sort_idx]
        q11_y = q11_y[sort_idx]

        ax.plot(
            n11_y,
            q11_y,
            color=colors_y[i],
            linewidth=1.5,
            linestyle="--",
        )

        # Add y label at end of line
        if len(n11_y) > 0:
            ax.annotate(
                f"y={y_val:.2f}",
                xy=(n11_y[-1], q11_y[-1]),
                fontsize=7,
                color=colors_y[i],
                ha="left",
            )

    # Mark BEP point (bep already computed above for normalization)
    ax.plot(
        n11_bep,
        q11_bep,
        "r*",
        markersize=15,
        markeredgecolor="k",
        markeredgewidth=1,
        label="BEP (100%)",
        zorder=10,
    )

    # Labels and title
    ax.set_xlabel("N11", fontsize=12)
    ax.set_ylabel("Q11", fontsize=12)
    ax.set_title("Turbine Power Hill Chart (% of BEP)", fontsize=14)

    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def _build_t11_interpolator(char: Characteristics):
    """Build T11 interpolator in (y, theta) space if not already present."""
    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

    from src.structured_tri_interplator import StructuredTriInterpolator

    points = np.column_stack([char.y_data_ext, char.theta_data_ext])

    # Extend T11 data same way as other quantities
    t11_data = char.t11_data.copy()

    # Handle theta wrapping (same logic as in _build_suter_interpolators)
    wrap_threshold = np.pi * 0.8
    near_pos_pi = char.theta_data > wrap_threshold
    near_neg_pi = char.theta_data < -wrap_threshold

    t11_extended = t11_data.copy()
    if np.any(near_pos_pi):
        t11_extended = np.concatenate([t11_extended, t11_data[near_pos_pi]])
    if np.any(near_neg_pi):
        t11_extended = np.concatenate([t11_extended, t11_data[near_neg_pi]])

    if char.interp_method == "clough_tocher":
        char.interp_t11 = CloughTocher2DInterpolator(points, t11_extended)
    elif char.interp_method == "linear":
        char.interp_t11 = LinearNDInterpolator(points, t11_extended)
    else:  # "structured"
        char.interp_t11 = StructuredTriInterpolator(points, t11_extended)


def plot_trajectory_on_hillchart(
    char: Characteristics,
    N_T: np.ndarray,
    Q_T: np.ndarray,
    H_T: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    label: str = "Trajectory",
    linewidth: float = 2.0,
    marker_size: float = 20,
):
    """
    Plot a trajectory on a hill chart with H encoded as color.

    Args:
        char: Characteristics instance
        N_T: Turbine speed array [rpm]
        Q_T: Turbine flow array [m³/s]
        H_T: Turbine head array [m] - used for coloring
        fig: Existing figure (optional)
        ax: Existing axes (required for overlay)
        cmap: Colormap for H values (default: "viridis")
        label: Label for the trajectory
        linewidth: Line width for trajectory
        marker_size: Size of scatter markers

    Returns:
        fig, ax, cbar: Matplotlib figure, axes, and colorbar objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))

    # Convert to unit parameters
    N11 = np.array([char.N11(N, H) for N, H in zip(N_T, H_T)])
    Q11 = np.array([char.Q11(Q, H) for Q, H in zip(Q_T, H_T)])
    H_arr = np.array(H_T)

    # Create line segments for colored line
    points = np.array([N11, Q11]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize H for colormap
    norm = plt.Normalize(H_arr.min(), H_arr.max())

    # Create LineCollection with H-based colors
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=0.8)
    lc.set_array(H_arr[:-1])  # Color by H at segment start
    ax.add_collection(lc)

    # Add scatter points on top
    scatter = ax.scatter(
        N11, Q11, c=H_arr, cmap=cmap, norm=norm, s=marker_size, edgecolors="k", linewidths=0.5, zorder=5
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="H [m]", shrink=0.8)

    ax.legend(loc="upper left", fontsize=8)

    return fig, ax, cbar


def plot_swirl_numbers(
    char: Characteristics,
    swirl_values: list[float],
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """
    Plot constant swirl number lines on a hill chart.

    Args:
        char: Characteristics instance
        swirl_values: List of swirl values to plot
        fig: Existing figure (optional)
        ax: Existing axes (required, uses existing axis limits)

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Use existing axis limits
    n11_min, n11_max = ax.get_xlim()
    q11_min, q11_max = ax.get_ylim()

    _add_swirl_lines(ax, char, n11_min, n11_max, swirl_values, q11_min, q11_max)

    return fig, ax


def _add_swirl_lines(
    ax,
    char: Characteristics,
    n11_min: float,
    n11_max: float,
    swirl_values: list[float] | None = None,
    q11_min: float | None = None,
    q11_max: float | None = None,
):
    """
    Add constant swirl number lines to a hill chart.

    Swirl number S = U/Cm - cot(β̄)

    In unit parameters, constant S means constant Q11/N11 ratio.

    Args:
        ax: Matplotlib axes
        char: Characteristics instance
        n11_min: Minimum N11 for line extent
        n11_max: Maximum N11 for line extent
        swirl_values: List of S values to plot
        q11_min: Minimum Q11 for clipping (optional)
        q11_max: Maximum Q11 for clipping (optional)
    """
    import math

    if swirl_values is None:
        swirl_values = [-0.5, -0.25, 0, 0.25, 0.5]

    # Get Q11 bounds from data if not provided
    if q11_min is None or q11_max is None:
        mask = char._turbine_mask
        q11_turb = char.q11_data[mask]
        if q11_min is None:
            q11_min = q11_turb.min()
        if q11_max is None:
            q11_max = q11_turb.max()

    # Get blade angle
    beta_bar = char.beta_bar
    cot_beta = 1.0 / math.tan(beta_bar)

    # At BEP, S=0, so Q11_bep/N11_bep gives the ratio at zero swirl
    bep = char.bep_turb()
    _, n11_bep, q11_bep, _, _ = bep
    ratio_bep = q11_bep / n11_bep

    # k = cot(β̄) × ratio_bep (from S = k × (N11/Q11) - cot(β̄) = 0 at BEP)
    k = cot_beta * ratio_bep

    # For constant S: Q11/N11 = k / (S + cot(β̄))
    n11_line = np.linspace(n11_min, n11_max, 100)

    import matplotlib.pyplot as plt
    colors_s = plt.cm.Oranges(np.linspace(0.4, 0.7, len(swirl_values)))

    for i, S in enumerate(swirl_values):
        denom = S + cot_beta
        if abs(denom) < 1e-6:
            continue

        ratio = k / denom
        q11_line = ratio * n11_line

        # Clip to Q11 bounds
        valid_mask = (q11_line >= q11_min) & (q11_line <= q11_max)
        if not np.any(valid_mask):
            continue

        n11_clipped = n11_line[valid_mask]
        q11_clipped = q11_line[valid_mask]

        ax.plot(
            n11_clipped,
            q11_clipped,
            color=colors_s[i],
            linewidth=1.5,
            linestyle="--",
            label=f"S={S:.2f}",
        )

        # Label at end of line (if within bounds)
        if len(n11_clipped) > 0:
            ax.annotate(
                f"S={S:.2f}",
                xy=(n11_clipped[-1], q11_clipped[-1]),
                fontsize=7,
                color=colors_s[i],
                ha="left",
            )
