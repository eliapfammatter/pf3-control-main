"""
Fast triangle interpolation for CasADi using compiled C.

Usage:
    from src.triinterp import TriInterpCasadi

    # Pass interpolator object, not characteristic
    interp_wh = TriInterpCasadi(turbine.characteristic.interp_wh)
    interp_wb = TriInterpCasadi(turbine.characteristic.interp_wb)

    wh = interp_wh(y, theta)  # CasADi symbolic expression
    wb = interp_wb(y, theta)

The interpolator uses content-based hashing to support compiled NLP caching.
Data is saved to binary files and loaded by hash, so the same data always
produces the same hash (deterministic across runs).
"""

import ctypes
import hashlib
from pathlib import Path

import casadi as ca
import numpy as np

# Path to compiled library
_LIB_PATH = Path(__file__).parent / "libtriinterp.so"
_lib = None
_interp_func = None  # Shared CasADi external function

# Cache directory for binary data files
CACHE_DIR = Path(__file__).parent / ".cache"


def _ensure_lib():
    """Load the shared library and set up cache directory."""
    global _lib, _interp_func
    if _lib is None:
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"Library not found: {_LIB_PATH}\n"
                f"Build with: cd {_LIB_PATH.parent} && make"
            )
        # RTLD_GLOBAL makes symbols available to other dynamically loaded libs (CasADi JIT)
        _lib = ctypes.CDLL(str(_LIB_PATH), mode=ctypes.RTLD_GLOBAL)

        # triinterp_create signature (legacy pointer-based interface)
        _lib.triinterp_create.argtypes = [
            ctypes.c_int,  # n_tri
            ctypes.c_int,  # n_points
            ctypes.c_int,  # n_levels
            ctypes.POINTER(ctypes.c_int64),  # tri_vertices
            ctypes.POINTER(ctypes.c_double),  # tri_y
            ctypes.POINTER(ctypes.c_double),  # tri_theta
            ctypes.POINTER(ctypes.c_double),  # tri_values
            ctypes.POINTER(ctypes.c_int64),  # strip_starts
            ctypes.POINTER(ctypes.c_double),  # y_levels
        ]
        _lib.triinterp_create.restype = ctypes.c_int64

        _lib.triinterp_destroy.argtypes = [ctypes.c_int64]
        _lib.triinterp_destroy.restype = None

        # Set cache directory for hash-based loading
        _lib.triinterp_set_cache_dir.argtypes = [ctypes.c_char_p]
        _lib.triinterp_set_cache_dir.restype = None

        # Ensure cache directory exists and tell C library where it is
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _lib.triinterp_set_cache_dir(str(CACHE_DIR).encode("utf-8"))

        # Create shared CasADi external function (one for all instances)
        _interp_func = ca.external("interp_wh", str(_LIB_PATH))

    return _lib


def _np_ptr(arr, dtype):
    """Get ctypes pointer to contiguous numpy array."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    if dtype == np.int64:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), arr
    else:
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr


class TriInterpCasadi:
    """
    CasADi-compatible triangle interpolator using compiled C.

    Uses content-based hashing: the interpolator data is hashed and saved
    to a binary file. The hash (as 64-bit int) is passed to the C function
    instead of a pointer. This allows compiled NLP solvers to work across
    Python sessions (the hash is deterministic for the same data).
    """

    def __init__(self, interpolator):
        """
        Initialize from a StructuredTriInterpolator object.

        Args:
            interpolator: StructuredTriInterpolator with attributes:
                _tri_vertices, _tri_y, _tri_theta, _values,
                _strip_starts, _y_levels
        """
        _ensure_lib()

        # Get triangulation data directly from interpolator
        tri_vertices = np.ascontiguousarray(
            interpolator._tri_vertices.flatten(), dtype=np.int64
        )
        tri_y = np.ascontiguousarray(interpolator._tri_y, dtype=np.float64)
        tri_theta = np.ascontiguousarray(interpolator._tri_theta, dtype=np.float64)
        tri_values = np.ascontiguousarray(interpolator._values, dtype=np.float64)
        strip_starts = np.ascontiguousarray(interpolator._strip_starts, dtype=np.int64)
        y_levels = np.ascontiguousarray(interpolator._y_levels, dtype=np.float64)

        n_tri = len(interpolator._tri_vertices)
        n_points = len(tri_y)
        n_levels = len(strip_starts)

        # Compute deterministic hash from data content
        self._hash_hex = self._compute_hash(
            n_tri,
            n_points,
            n_levels,
            tri_vertices,
            tri_y,
            tri_theta,
            tri_values,
            strip_starts,
            y_levels,
        )
        # Use first 13 hex chars = 52 bits as integer ID
        # (52 bits fits exactly in double's mantissa, no precision loss)
        # we need this because we can only pass double as external value to casadi interface.
        self._hash_int = int(self._hash_hex[:13], 16)

        # Save binary file if not already cached
        bin_path = CACHE_DIR / f"{self._hash_hex[:13]}.bin"
        if not bin_path.exists():
            self._save_binary(
                bin_path,
                n_tri,
                n_points,
                n_levels,
                tri_vertices,
                tri_y,
                tri_theta,
                tri_values,
                strip_starts,
                y_levels,
            )

    @staticmethod
    def _compute_hash(
        n_tri,
        n_points,
        n_levels,
        tri_vertices,
        tri_y,
        tri_theta,
        tri_values,
        strip_starts,
        y_levels,
    ) -> str:
        """Compute SHA256 hash of all interpolator data."""
        h = hashlib.sha256()
        # Include metadata
        h.update(np.array([n_tri, n_points, n_levels], dtype=np.int32).tobytes())
        # Include all arrays
        h.update(tri_vertices.tobytes())
        h.update(tri_y.tobytes())
        h.update(tri_theta.tobytes())
        h.update(tri_values.tobytes())
        h.update(strip_starts.tobytes())
        h.update(y_levels.tobytes())
        return h.hexdigest()

    @staticmethod
    def _save_binary(
        path: Path,
        n_tri,
        n_points,
        n_levels,
        tri_vertices,
        tri_y,
        tri_theta,
        tri_values,
        strip_starts,
        y_levels,
    ):
        """Save interpolator data as binary file for C to load."""
        with open(path, "wb") as f:
            # Write metadata (3 int32s)
            np.array([n_tri, n_points, n_levels], dtype=np.int32).tofile(f)
            # Write arrays in order
            tri_vertices.tofile(f)
            tri_y.tofile(f)
            tri_theta.tofile(f)
            tri_values.tofile(f)
            strip_starts.tofile(f)
            y_levels.tofile(f)

    def __call__(self, y, theta):
        """
        Evaluate interpolation at (y, theta).

        Args:
            y: Guide vane opening (CasADi symbolic or numeric)
            theta: Suter angle in radians (CasADi symbolic or numeric)

        Returns:
            Interpolated W_H value (CasADi expression)
        """
        # Pass hash as ID (not pointer!) - deterministic across runs
        hash_as_double = np.float64(self._hash_int).item()
        return _interp_func(hash_as_double, y, theta)
