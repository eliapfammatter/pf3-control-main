import numba
import numpy as np


@numba.njit(cache=True)
def find_triangle_and_interpolate(
    y, theta, tri_vertices, tri_y, tri_theta, tri_values, strip_starts, y_levels
):
    """
    Find which triangle (y, theta) falls in and interpolate using barycentric coords.

    Uses the structured y-strip layout for fast triangle search:
    binary search on y_levels to find the strip, then linear scan within the strip.

    Args:
        y: Query point y-coordinate (guide vane opening)
        theta: Query point theta-coordinate (Suter angle, radians)
        tri_vertices: (n_tri, 3) int array (into tri_y, tri_theta and tri_values) of vertex indices defining each triangle
        tri_y: (n_points,) float array of y-coordinates for all vertices
        tri_theta: (n_points,) float array of theta-coordinates for all vertices
        tri_values: (n_points,) float array of values to interpolate at each vertex
        strip_starts: (n_levels,) int array where strip_starts[i] is the first
            triangle index for the strip between y_levels[i] and y_levels[i+1];
            strip_starts[-1] is the total number of triangles (sentinel)
        y_levels: (n_levels-1,) float array of sorted unique y values defining strips

    Returns:
        Interpolated value at (y, theta), or NaN if outside all triangles.
    """
    n_levels = len(y_levels)

    # Binary search for the y-strip: find i such that y_levels[i] <= y < y_levels[i+1]
    if y < y_levels[0] or y > y_levels[-1]:
        return np.nan

    lo, hi = 0, n_levels - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if y_levels[mid] <= y:
            lo = mid
        else:
            hi = mid - 1
    strip_idx = lo

    # Get triangle range for this strip
    t_start = strip_starts[strip_idx]
    t_end = strip_starts[strip_idx + 1]

    # Also check adjacent strips (point may be on boundary)
    if strip_idx > 0:
        t_start = strip_starts[strip_idx - 1]
    if strip_idx < n_levels - 2:
        t_end = strip_starts[strip_idx + 2]

    # Linear scan through candidate triangles
    for t in range(t_start, t_end):
        i0 = tri_vertices[t, 0]
        i1 = tri_vertices[t, 1]
        i2 = tri_vertices[t, 2]

        # Triangle vertex coordinates
        y0, t0 = tri_y[i0], tri_theta[i0]
        y1, t1 = tri_y[i1], tri_theta[i1]
        y2, t2 = tri_y[i2], tri_theta[i2]

        # Barycentric coordinates via determinant method
        denom = (y1 - y2) * (t0 - t2) + (t2 - t1) * (y0 - y2)
        if abs(denom) < 1e-30:
            continue

        inv_denom = 1.0 / denom
        lam0 = ((y1 - y2) * (theta - t2) + (t2 - t1) * (y - y2)) * inv_denom
        lam1 = ((y2 - y0) * (theta - t2) + (t0 - t2) * (y - y2)) * inv_denom
        lam2 = 1.0 - lam0 - lam1

        # Check if point is inside triangle (with small tolerance)
        tol = -1e-10
        if lam0 >= tol and lam1 >= tol and lam2 >= tol:
            return lam0 * tri_values[i0] + lam1 * tri_values[i1] + lam2 * tri_values[i2]

    return np.nan


@numba.njit(cache=True)
def find_triangle_interp_with_grad(
    y, theta, tri_vertices, tri_y, tri_theta, tri_values, strip_starts, y_levels
):
    """
    Find triangle, interpolate, AND compute analytic gradients.

    Same algorithm as find_triangle_and_interpolate, but also returns
    the partial derivatives dv/dy and dv/dtheta.

    Returns:
        (value, dv_dy, dv_dtheta) - interpolated value and gradients,
        or (NaN, 0.0, 0.0) if outside all triangles.
    """
    n_levels = len(y_levels)

    # Bounds check
    if y < y_levels[0] or y > y_levels[n_levels - 1]:
        return np.nan, 0.0, 0.0

    # Binary search for y-strip
    lo, hi = 0, n_levels - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if y_levels[mid] <= y:
            lo = mid
        else:
            hi = mid - 1
    strip_idx = lo

    # Triangle range (include adjacent strips)
    t_start = strip_starts[strip_idx]
    t_end = strip_starts[strip_idx + 1]
    if strip_idx > 0:
        t_start = strip_starts[strip_idx - 1]
    if strip_idx < n_levels - 2:
        t_end = strip_starts[strip_idx + 2]

    # Linear scan with early exit
    for t in range(t_start, t_end):
        i0 = tri_vertices[t, 0]
        i1 = tri_vertices[t, 1]
        i2 = tri_vertices[t, 2]

        y0, t0 = tri_y[i0], tri_theta[i0]
        y1, t1 = tri_y[i1], tri_theta[i1]
        y2, t2 = tri_y[i2], tri_theta[i2]
        v0, v1, v2 = tri_values[i0], tri_values[i1], tri_values[i2]

        denom = (y1 - y2) * (t0 - t2) + (t2 - t1) * (y0 - y2)
        if abs(denom) < 1e-30:
            continue

        inv_denom = 1.0 / denom
        lam0 = ((y1 - y2) * (theta - t2) + (t2 - t1) * (y - y2)) * inv_denom
        lam1 = ((y2 - y0) * (theta - t2) + (t0 - t2) * (y - y2)) * inv_denom
        lam2 = 1.0 - lam0 - lam1

        tol = -1e-10
        if lam0 >= tol and lam1 >= tol and lam2 >= tol:
            # Interpolated value
            value = lam0 * v0 + lam1 * v1 + lam2 * v2

            # Analytic derivatives of barycentric coords w.r.t. y
            dlam0_dy = (t2 - t1) * inv_denom
            dlam1_dy = (t0 - t2) * inv_denom
            dlam2_dy = -dlam0_dy - dlam1_dy

            # Analytic derivatives of barycentric coords w.r.t. theta
            dlam0_dt = (y1 - y2) * inv_denom
            dlam1_dt = (y2 - y0) * inv_denom
            dlam2_dt = -dlam0_dt - dlam1_dt

            # Chain rule: dv/dy = sum(v_i * dlam_i/dy)
            dv_dy = v0 * dlam0_dy + v1 * dlam1_dy + v2 * dlam2_dy
            dv_dt = v0 * dlam0_dt + v1 * dlam1_dt + v2 * dlam2_dt

            return value, dv_dy, dv_dt

    return np.nan, 0.0, 0.0


@numba.njit(cache=True)
def interpolate_points(
    y_arr, theta_arr, tri_vertices, tri_y, tri_theta, tri_values, strip_starts, y_levels
):
    """Vectorized version: interpolate an array of points."""
    n = len(y_arr)
    result = np.empty(n)
    for i in range(n):
        result[i] = find_triangle_and_interpolate(
            y_arr[i],
            theta_arr[i],
            tri_vertices,
            tri_y,
            tri_theta,
            tri_values,
            strip_starts,
            y_levels,
        )
    return result


def build_triangulation(y_data, theta_data):
    """
    Build triangle vertex indices respecting iso-y curves.

    Returns:
        triangles: (n_tri, 3) int array of vertex indices into the original data y_data, theta_data and values
        strip_starts: (n_levels,) int array — strip_starts[i] is the first
                      triangle index belonging to the strip between
                      y_levels[i] and y_levels[i+1]
        y_levels: sorted unique y values
    """
    y_unique = np.unique(y_data)
    y_unique = np.sort(y_unique)

    # Build index mapping: for each y, get indices sorted by theta
    y_to_indices = {}
    for y_val in y_unique:
        mask = np.abs(y_data - y_val) < 1e-10
        indices = np.where(mask)[0]
        theta_order = np.argsort(theta_data[indices])
        y_to_indices[y_val] = indices[theta_order]

    triangles = []
    strip_starts = []

    # Create triangular strips between adjacent y levels
    for i in range(len(y_unique) - 1):
        strip_starts.append(len(triangles))

        y_lo = y_unique[i]
        y_hi = y_unique[i + 1]

        idx_lo = y_to_indices[y_lo]
        idx_hi = y_to_indices[y_hi]

        theta_lo = theta_data[idx_lo]
        theta_hi = theta_data[idx_hi]

        # Greedy zipper: advance along whichever curve has the smaller theta
        i_lo, i_hi = 0, 0
        n_lo, n_hi = len(idx_lo), len(idx_hi)

        while i_lo < n_lo - 1 or i_hi < n_hi - 1:
            v_lo = idx_lo[i_lo]
            v_hi = idx_hi[i_hi]

            if i_lo >= n_lo - 1:
                i_hi += 1
                triangles.append([v_lo, v_hi, idx_hi[i_hi]])
            elif i_hi >= n_hi - 1:
                i_lo += 1
                triangles.append([v_lo, v_hi, idx_lo[i_lo]])
            elif theta_lo[i_lo + 1] <= theta_hi[i_hi + 1]:
                i_lo += 1
                triangles.append([v_lo, v_hi, idx_lo[i_lo]])
            else:
                i_hi += 1
                triangles.append([v_lo, v_hi, idx_hi[i_hi]])

    # Sentinel: total number of triangles
    strip_starts.append(len(triangles))

    return (
        np.ascontiguousarray(triangles, dtype=np.int64),
        np.ascontiguousarray(strip_starts, dtype=np.int64),
        np.ascontiguousarray(y_unique, dtype=np.float64),
    )


class StructuredTriInterpolator:
    """
    Structured triangulation interpolator that respects iso-y curves.

    Instead of arbitrary Delaunay triangles that can span large y ranges,
    this creates triangles that:
    1. Connect points along iso-y curves (constant guide vane opening)
    2. Form strips between adjacent y values

    Uses numba-jitted barycentric interpolation with y-strip spatial indexing.

    Usage:
        interp = StructuredTriInterpolator(y_data, theta_data, values)
        result = interp([[y, theta]])  # same API as LinearNDInterpolator
    """

    def __init__(self, points, values):
        """
        Args:
            y_data: Array of y values (guide vane openings)
            theta_data: Array of theta values (Suter angles)
            values: Array of values to interpolate
        """
        y_data = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        theta_data = np.ascontiguousarray(points[:, 1], dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)

        triangles, strip_starts, y_levels = build_triangulation(y_data, theta_data)

        self._tri_vertices = triangles
        self._tri_y = y_data
        self._tri_theta = theta_data
        self._values = values
        self._strip_starts = strip_starts
        self._y_levels = y_levels

        # Store for external access (plotting, debugging)
        self.triangles = triangles
        self.y_data = y_data
        self.theta_data = theta_data

    def __call__(self, points):
        """
        Evaluate interpolator at points.

        Args:
            points: Array of shape (n, 2) with columns [y, theta]

        Returns:
            Array of interpolated values, shape (n,)
        """
        points = np.atleast_2d(points)
        y_arr = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        theta_arr = np.ascontiguousarray(points[:, 1], dtype=np.float64)
        return interpolate_points(
            y_arr,
            theta_arr,
            self._tri_vertices,
            self._tri_y,
            self._tri_theta,
            self._values,
            self._strip_starts,
            self._y_levels,
        )
