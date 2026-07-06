#!/usr/bin/env python3
"""
Visualize the (y, θ) trajectory through the Delaunay triangulation mesh.

This shows:
1. The Delaunay triangulation of characteristic data points
2. The trajectory of operating points over time
3. Which triangle the trajectory is in at each moment
4. Triangle boundaries that are crossed
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay

# Import turbine
from src.characteristics import Characteristics, StructuredTriInterpolator
from src.structured_tri_interplator import build_triangulation

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Visualize trajectory through Delaunay mesh"
)
parser.add_argument(
    "--native",
    "-n",
    type=int,
    default=None,
    help="Native cache index (skip interactive)",
)
parser.add_argument(
    "--list", "-l", action="store_true", help="List available caches and exit"
)
parser.add_argument(
    "--interp",
    "-i",
    choices=["linear", "structured", "clough_tocher"],
    default=None,
    help="Override interpolation method (default: from cache filename)",
)
args = parser.parse_args()

data_dir = Path("data/pf3")
output_dir = Path("output")
cache_dir = output_dir / "cache"


def get_fmu_cache_for_native(native_path):
    """Extract scenario from native cache name and find matching FMU cache.

    Native: native_<scenario>_t<time>_<interp>.pkl
    FMU:    fmu_<scenario>_t<time>.pkl
    """
    name = native_path.stem  # e.g., "native_fast_transition_t60.0_linear"
    # Match: native_<scenario>_t<time>_<interp>
    match = re.match(r"native_(.+_t[\d.]+)_(\w+)$", name)
    if match:
        scenario = match.group(1)  # e.g., "fast_transition_t60.0"
        interp = match.group(2)  # e.g., "linear"
        fmu_name = f"fmu_{scenario}.pkl"
        fmu_path = native_path.parent / fmu_name
        if fmu_path.exists():
            return fmu_path, interp
    return None, None


# Find native caches with matching FMU caches
native_caches_all = sorted(cache_dir.glob("native_*.pkl"))
available = []  # List of (native_path, fmu_path, interp_method, time)
for nc in native_caches_all:
    fmu_path, interp = get_fmu_cache_for_native(nc)
    if fmu_path:
        # Extract time from filename for sorting
        time_match = re.search(r"_t([\d.]+)_", nc.name)
        t = float(time_match.group(1)) if time_match else 0
        available.append((nc, fmu_path, interp, t))

# Sort by time (longest first)
available.sort(key=lambda x: x[3], reverse=True)

# --list mode: show available caches and exit
if args.list:
    print("\n=== Available caches (native + matching FMU) ===")
    for i, (nc, fc, interp, t) in enumerate(available):
        print(f"  [{i}] {nc.name}  (t={t}s, interp={interp})")
    print(f"\nUsage: python debug_trajectory_in_mesh.py -n 0")
    exit(0)

if not available:
    print("No matching cache pairs found. Please run compare_140state_vs_fmu.py first.")
    exit(1)

print("Loading simulation data...")

# Native cache selection (FMU auto-selected)
print("\n=== Available native caches ===")
for i, (nc, fc, interp, t) in enumerate(available):
    print(f"  [{i}] {nc.name}  (t={t}s, interp={interp})")

if args.native is not None:
    idx = args.native
    print(f"  -> Using -n {idx}")
elif len(available) == 1:
    idx = 0
    print(f"  -> Auto-selecting [{idx}] (only one available)")
else:
    idx = int(input(f"Select native cache [0-{len(available)-1}]: "))

native_cache, fmu_cache, cache_interp_method, _ = available[idx]

# Use interp method from cache, or override with command-line argument
interp_method = args.interp if args.interp else cache_interp_method
if args.interp and args.interp != cache_interp_method:
    print(
        f"  WARNING: Overriding cache interp '{cache_interp_method}' with '{args.interp}'"
    )

print(f"\nUsing: {native_cache.name}")
print(f"  FMU: {fmu_cache.name}")
print(f"  Interpolation: {interp_method}")

# Load turbine with the correct interpolation method
turbine = Characteristics(
    d_ref=0.3477,
    h_n=5.0,
    q_n=0.1959,
    t_n=224.057,
    n_n=369.3346,
    char_file=data_dir / "missing_files/STORE_4_quadrant_characteristic.txt",
    interp_method=interp_method,
)

with open(fmu_cache, "rb") as f:
    fmu_data = pickle.load(f)

with open(native_cache, "rb") as f:
    native_data = pickle.load(f)

# Extract time series from FMU (nested structure)
t_fmu = fmu_data["timestamps"]
H_T_fmu = fmu_data["outputs"]["FTURB1-H"]
Q_T_fmu = fmu_data["outputs"]["FTURB1-Q"]
y_T_fmu = fmu_data["inputs"]["y_T"]
N_T_fmu = fmu_data["inputs"]["N_T"]

# Extract time series from native (flat structure)
t_native = native_data["timestamps"]
H_T_native = native_data["H_T"]
Q_T_native = native_data["Q_T"]


# Compute (y, θ) trajectory for both FMU and native
def compute_theta(Q, N, q_n, n_n):
    """Compute Suter angle θ = arctan2(υ, α) in RADIANS [-π, π]

    NOTE: Must use radians to match the native model's interpolator (turbine.interp_wh)
    which is built with theta_data in radians.
    """
    alpha = N / n_n
    upsilon = Q / q_n
    # arctan2 handles all quadrants correctly, returns [-π, π]
    theta_rad = np.arctan2(upsilon, alpha)
    return theta_rad


# Interpolate native to FMU time grid for fair comparison
Q_T_native_interp = np.interp(t_fmu, t_native, Q_T_native)
H_T_native_interp = np.interp(t_fmu, t_native, H_T_native)

# Compute theta for FMU and native trajectories
theta_fmu = compute_theta(Q_T_fmu, N_T_fmu, turbine.q_n, turbine.n_n)
theta_native = compute_theta(Q_T_native_interp, N_T_fmu, turbine.q_n, turbine.n_n)

# Build triangulation from characteristic data
# NOTE: Use radians to match the native model's interpolator (turbine.interp_wh)
y_data = turbine.y_data
theta_data = turbine.theta_data  # Already in radians [-π, π]
wh_data = turbine.wh_data
points = np.column_stack([y_data, theta_data])

# Determine triangulation type based on interpolation method
# - "linear" and "clough_tocher" both use Delaunay internally
# - "structured" uses custom iso-y respecting triangulation
tri = Delaunay(points)  # Always needed for find_simplex

if interp_method == "structured":
    use_structured_viz = True
    print(f"Using STRUCTURED triangulation: {len(y_data)} points")
else:
    use_structured_viz = False
    print(
        f"Using DELAUNAY triangulation: {len(y_data)} points, {len(tri.simplices)} triangles"
    )
    if interp_method == "clough_tocher":
        print(
            "  Note: clough_tocher uses C1 smooth interpolation (visualization shows underlying Delaunay mesh)"
        )

# Debug: check FMU vs native Q variation
print(f"\nFMU data check:")
print(
    f"  y_T_fmu:  min={y_T_fmu.min():.4f}, max={y_T_fmu.max():.4f}, std={y_T_fmu.std():.6f}"
)
print(
    f"  Q_T_fmu:  min={Q_T_fmu.min():.4f}, max={Q_T_fmu.max():.4f}, std={Q_T_fmu.std():.6f}"
)
print(
    f"  N_T_fmu:  min={N_T_fmu.min():.2f}, max={N_T_fmu.max():.2f}, std={N_T_fmu.std():.4f}"
)
print(
    f"  theta_fmu: min={np.degrees(theta_fmu.min()):.2f}°, max={np.degrees(theta_fmu.max()):.2f}°, std={np.degrees(theta_fmu.std()):.2f}°"
)
print(f"\nNative data check:")
print(
    f"  Q_T_native: min={Q_T_native.min():.4f}, max={Q_T_native.max():.4f}, std={Q_T_native.std():.6f}"
)
print(
    f"  theta_native: min={np.degrees(theta_native.min()):.2f}°, max={np.degrees(theta_native.max()):.2f}°, std={np.degrees(theta_native.std()):.2f}°"
)


# Find which triangle each trajectory point is in
def find_triangle_indices(tri, trajectory_y, trajectory_theta):
    """Find which simplex (triangle) each point is in. Returns -1 if outside."""
    traj_points = np.column_stack([trajectory_y, trajectory_theta])
    return tri.find_simplex(traj_points)


tri_indices_fmu = find_triangle_indices(tri, y_T_fmu, theta_fmu)
tri_indices_native = find_triangle_indices(tri, y_T_fmu, theta_native)


# Find where triangles change
def find_triangle_changes(tri_indices, times):
    """Find times where the trajectory crosses to a different triangle."""
    changes = []
    for i in range(1, len(tri_indices)):
        if tri_indices[i] != tri_indices[i - 1]:
            changes.append(
                {
                    "time": times[i],
                    "from_tri": tri_indices[i - 1],
                    "to_tri": tri_indices[i],
                    "idx": i,
                }
            )
    return changes


changes_fmu = find_triangle_changes(tri_indices_fmu, t_fmu)
changes_native = find_triangle_changes(tri_indices_native, t_fmu)

print(f"\nFMU trajectory: {len(changes_fmu)} triangle changes")
print(f"Native trajectory: {len(changes_native)} triangle changes")


# Compute W_H error at each point
def compute_wh_error(y_arr, theta_arr_rad, H_arr, alpha_arr, upsilon_arr):
    """Back-calculate W_H from H and compare to interpolated W_H."""
    wh_interp = []
    wh_actual = []
    for y, theta_rad, H, alpha, upsilon in zip(
        y_arr, theta_arr_rad, H_arr, alpha_arr, upsilon_arr
    ):
        # Interpolated W_H (theta already in radians)
        wh_i = turbine.interp_wh([[y, theta_rad]])[0]
        wh_interp.append(wh_i)
        # Actual W_H from H = W_H × H_n × (α² + υ²)
        denom = alpha**2 + upsilon**2
        wh_a = H / (turbine.h_n * denom) if denom > 1e-10 else np.nan
        wh_actual.append(wh_a)
    return np.array(wh_interp), np.array(wh_actual)


alpha_fmu = N_T_fmu / turbine.n_n
upsilon_fmu = Q_T_fmu / turbine.q_n

wh_interp_fmu, wh_actual_fmu = compute_wh_error(
    y_T_fmu, theta_fmu, H_T_fmu, alpha_fmu, upsilon_fmu
)

# Create the visualization
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: (y, θ) mesh with trajectory
# ============================================================================
ax1 = fig.add_subplot(2, 2, 1)


# Convert radians to degrees [0, 360) FOR PLOTTING ONLY
def rad_to_deg_0_360(theta_rad):
    """Convert radians [-π, π] to degrees [0, 360) for plotting."""
    theta_deg = np.degrees(theta_rad)
    theta_deg = np.where(theta_deg < 0, theta_deg + 360, theta_deg)
    return theta_deg


theta_data_plot = rad_to_deg_0_360(theta_data)
theta_fmu_plot = rad_to_deg_0_360(theta_fmu)
theta_native_plot = rad_to_deg_0_360(theta_native)

# Build bridge triangles to fill the gap at the ±π boundary.
# Points near +π and -π both map to ~180° in [0,360) space, so bridge
# triangles connecting them are small and well-formed (no overlap issues).
from matplotlib.tri import Triangulation


def build_bridge_triangles(y_data, theta_data):
    """Create triangles bridging the ±π gap for each pair of adjacent y levels."""
    y_unique = np.sort(np.unique(y_data))
    bridges = []
    for i in range(len(y_unique) - 1):
        y_lo, y_hi = y_unique[i], y_unique[i + 1]
        idx_lo = np.where(np.abs(y_data - y_lo) < 1e-10)[0]
        idx_hi = np.where(np.abs(y_data - y_hi) < 1e-10)[0]
        idx_lo = idx_lo[np.argsort(theta_data[idx_lo])]
        idx_hi = idx_hi[np.argsort(theta_data[idx_hi])]
        if len(idx_lo) < 1 or len(idx_hi) < 1:
            continue
        # Connect last (θ ≈ +π) to first (θ ≈ -π)
        bridges.append([idx_lo[-1], idx_hi[-1], idx_lo[0]])
        bridges.append([idx_hi[-1], idx_hi[0], idx_lo[0]])
    return np.array(bridges) if bridges else np.empty((0, 3), dtype=int)


if use_structured_viz:
    base_triangles, _, _ = build_triangulation(y_data, theta_data)
    # Structured zipper sweeps min→max θ without wrapping; add bridge triangles
    bridge_triangles = build_bridge_triangles(y_data, theta_data)
    all_triangles = np.concatenate([base_triangles, bridge_triangles])
    tri_label = "STRUCTURED"
else:
    # Delaunay already creates triangles crossing the ±π boundary
    all_triangles = tri.simplices
    tri_label = "DELAUNAY"

# Filter out triangles with large theta span (wrap-around artifacts in [0,360) space)
valid_triangles = []
for simplex in all_triangles:
    theta_vertices = theta_data_plot[simplex]
    span = theta_vertices.max() - theta_vertices.min()
    if span < 90:  # Only keep triangles with reasonable theta span
        valid_triangles.append(simplex)
valid_triangles = np.array(valid_triangles)
n_filtered = len(all_triangles) - len(valid_triangles)
print(
    f"{tri_label}: {len(all_triangles)} triangles, filtered {n_filtered} with large theta span"
)

mpl_tri = Triangulation(y_data, theta_data_plot, valid_triangles)

# Scale colorbar to trajectory region (with some margin)
y_min_traj, y_max_traj = min(y_T_fmu.min(), y_T_fmu.min()), max(
    y_T_fmu.max(), y_T_fmu.max()
)
theta_min_traj_plot, theta_max_traj_plot = min(
    theta_fmu_plot.min(), theta_native_plot.min()
), max(theta_fmu_plot.max(), theta_native_plot.max())
# Find W_H range in trajectory region (use plot coordinates)
mask = (
    (y_data >= y_min_traj - 0.05)
    & (y_data <= y_max_traj + 0.05)
    & (theta_data_plot >= theta_min_traj_plot - 5)
    & (theta_data_plot <= theta_max_traj_plot + 5)
)
if mask.sum() > 0:
    wh_min, wh_max = wh_data[mask].min(), wh_data[mask].max()
else:
    wh_min, wh_max = wh_data.min(), wh_data.max()

# Fill triangles with W_H values using gouraud shading
interp_labels = {
    "structured": "W_H (structured)",
    "linear": "W_H (linear)",
    "clough_tocher": "W_H (clough_tocher)",
}
tpc = ax1.tripcolor(
    mpl_tri,
    wh_data,
    shading="gouraud",
    cmap="viridis",
    alpha=0.7,
    vmin=wh_min,
    vmax=wh_max,
)
ax1.triplot(mpl_tri, "k-", linewidth=0.3, alpha=0.3)
plt.colorbar(tpc, ax=ax1, label=interp_labels.get(interp_method, "W_H"))

# Draw characteristic data points
ax1.scatter(y_data, theta_data_plot, c="black", s=10, alpha=0.5, zorder=5)

# Draw FMU trajectory
sc_fmu = ax1.scatter(
    y_T_fmu,
    theta_fmu_plot,
    c=t_fmu,
    cmap="plasma",
    s=15,
    marker="o",
    label="FMU trajectory",
    zorder=10,
    edgecolors="white",
    linewidths=0.3,
)

# Draw native trajectory
ax1.scatter(
    y_T_fmu,
    theta_native_plot,
    c=t_fmu,
    cmap="plasma",
    s=15,
    marker="s",
    label="Native trajectory",
    zorder=10,
    alpha=0.7,
    edgecolors="black",
    linewidths=0.3,
)

# Mark triangle boundary crossings for native
for change in changes_native:
    idx = change["idx"]
    ax1.scatter(
        [y_T_fmu[idx]],
        [theta_native_plot[idx]],
        c="red",
        s=100,
        marker="X",
        zorder=15,
        edgecolors="black",
        linewidths=1,
    )
    ax1.annotate(
        f"t={change['time']:.1f}s",
        (y_T_fmu[idx], theta_native_plot[idx]),
        fontsize=8,
        xytext=(5, 5),
        textcoords="offset points",
        color="red",
    )

ax1.set_xlabel("y (guide vane opening)")
ax1.set_ylabel("θ (Suter angle) [°]")
if interp_method == "structured":
    mesh_label = "STRUCTURED (iso-y)"
elif interp_method == "clough_tocher":
    mesh_label = "CLOUGH-TOCHER (C1 smooth)"
else:
    mesh_label = "DELAUNAY (linear)"
ax1.set_title(
    f"Trajectory through {mesh_label} mesh\n(red X = triangle boundary crossing for native)"
)
ax1.legend(loc="upper right")

# Dynamic axis limits based on trajectory with 10% margin (in degrees)
y_all = np.concatenate([y_T_fmu, y_T_fmu])  # Both use y_T_fmu
theta_all_plot = np.concatenate([theta_fmu_plot, theta_native_plot])
y_margin = (y_all.max() - y_all.min()) * 0.15 + 0.02
theta_margin = (theta_all_plot.max() - theta_all_plot.min()) * 0.15 + 2  # degrees
ax1.set_xlim(y_all.min() - y_margin, y_all.max() + y_margin)
ax1.set_ylim(theta_all_plot.min() - theta_margin, theta_all_plot.max() + theta_margin)

# ============================================================================
# Plot 2: H_T error over time with triangle crossings marked
# ============================================================================
ax2 = fig.add_subplot(2, 2, 2)

H_error_mm = (H_T_native_interp - H_T_fmu) * 1000

ax2.plot(t_fmu, H_error_mm, "b-", linewidth=1, label="H_T error (native - FMU)")
ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

# Mark triangle crossings
for change in changes_native:
    ax2.axvline(x=change["time"], color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax2.annotate(
        f"tri {change['from_tri']}→{change['to_tri']}",
        (change["time"], ax2.get_ylim()[1] * 0.9),
        fontsize=7,
        rotation=90,
        va="top",
        color="red",
    )

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("H_T error [mm]")
ax2.set_title(
    "H_T error with triangle boundary crossings\n(red lines = native trajectory changes triangle)"
)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Classic N11 vs Q11 turbine chart with trajectory
# ============================================================================
ax3 = fig.add_subplot(2, 2, 3)

# Get unique y values from characteristic data and plot iso-y curves
y_unique = np.unique(turbine.y_data)
cmap_y = plt.cm.coolwarm
norm_y = plt.Normalize(y_unique.min(), y_unique.max())

# Plot characteristic curves (iso-y lines in N11-Q11 space)
for y_val in y_unique:
    mask = np.abs(turbine.y_data - y_val) < 1e-6
    q11_curve = turbine.q11_data[mask]
    n11_curve = turbine.n11_data[mask]
    # Sort by N11 for smooth line
    sort_idx = np.argsort(n11_curve)
    ax3.plot(
        n11_curve[sort_idx],
        q11_curve[sort_idx],
        "-",
        color=cmap_y(norm_y(y_val)),
        linewidth=1.5,
        alpha=0.6,
    )

# Add colorbar for y values
sm = plt.cm.ScalarMappable(cmap=cmap_y, norm=norm_y)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, label="y (guide vane)")

# Compute N11 and Q11 for FMU trajectory
# N11 = (N * d_ref) / sqrt(H), Q11 = Q / (d_ref² * sqrt(H))
sqrt_H_fmu = np.sqrt(np.maximum(H_T_fmu, 0.01))  # Avoid sqrt of negative
Q11_traj_fmu = Q_T_fmu / (turbine.d_ref**2 * sqrt_H_fmu)
N11_traj_fmu = (N_T_fmu * turbine.d_ref) / sqrt_H_fmu

# Compute N11 and Q11 for native trajectory
sqrt_H_native = np.sqrt(np.maximum(H_T_native_interp, 0.01))
Q11_traj_native = Q_T_native_interp / (turbine.d_ref**2 * sqrt_H_native)
N11_traj_native = (N_T_fmu * turbine.d_ref) / sqrt_H_native

# Plot FMU trajectory
sc_fmu3 = ax3.scatter(
    N11_traj_fmu,
    Q11_traj_fmu,
    c=t_fmu,
    cmap="plasma",
    s=15,
    marker="o",
    label="FMU",
    zorder=10,
    edgecolors="white",
    linewidths=0.3,
)

# Plot native trajectory
ax3.scatter(
    N11_traj_native,
    Q11_traj_native,
    c=t_fmu,
    cmap="plasma",
    s=15,
    marker="s",
    label="Native",
    zorder=10,
    alpha=0.7,
    edgecolors="black",
    linewidths=0.3,
)

# Mark start and end points
ax3.scatter(
    [N11_traj_fmu[0]],
    [Q11_traj_fmu[0]],
    c="lime",
    s=100,
    marker="*",
    zorder=15,
    edgecolors="black",
    linewidths=1,
    label="Start",
)
ax3.scatter(
    [N11_traj_fmu[-1]],
    [Q11_traj_fmu[-1]],
    c="red",
    s=100,
    marker="*",
    zorder=15,
    edgecolors="black",
    linewidths=1,
    label="End",
)

ax3.set_xlabel("N11")
ax3.set_ylabel("Q11")
ax3.set_title(
    "Classic turbine chart: Q11 vs N11\n(colors = time, background = iso-y curves)"
)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3)

# Set axis limits with margin around trajectory
q11_all = np.concatenate([Q11_traj_fmu, Q11_traj_native])
n11_all = np.concatenate([N11_traj_fmu, N11_traj_native])
q11_margin = (q11_all.max() - q11_all.min()) * 0.2 + 0.01
n11_margin = (n11_all.max() - n11_all.min()) * 0.2 + 1
ax3.set_xlim(n11_all.min() - n11_margin, n11_all.max() + n11_margin)
ax3.set_ylim(q11_all.min() - q11_margin, q11_all.max() + q11_margin)

# ============================================================================
# Plot 4: W_H interpolation error
# ============================================================================
ax4 = fig.add_subplot(2, 2, 4)

wh_error = wh_interp_fmu - wh_actual_fmu

ax4.plot(t_fmu, wh_error, "b-", linewidth=1, label="W_H error (interp - actual)")
ax4.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

# Mark triangle crossings
for change in changes_native:
    ax4.axvline(x=change["time"], color="red", linestyle=":", linewidth=1, alpha=0.7)

ax4.set_xlabel("Time [s]")
ax4.set_ylabel("W_H error")
ax4.set_title(
    "W_H interpolation error\n(computed from FMU H vs interpolated from FMU (y,θ))"
)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_file = output_dir / f"trajectory_in_mesh_{interp_method}.png"
plt.savefig(output_file, dpi=300)
print(f"\nSaved: {output_file}")
plt.show()

# Brief summary
outside_mask = tri_indices_native < 0
if np.any(outside_mask):
    outside_times = t_fmu[outside_mask]
    print(
        f"WARNING: {np.sum(outside_mask)} points outside convex hull (t={outside_times[0]:.1f}s to {outside_times[-1]:.1f}s)"
    )
else:
    print("✓ Trajectory stays inside convex hull")
