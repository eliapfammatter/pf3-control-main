"""
Measurement Series - Core time series data structure with lazy calibration.

Provides MeasurementSeries dataclass for handling time series data with:
- Lazy calibration (computed on first access)
- Time masking and cropping
- Support for uniform and non-uniform sampling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class MeasurementSeries:
    """A measurement series with optional calibration, time, and masking.

    Supports two modes:
    1. Uniform sampling: Provide freq and t_start, time computed lazily
    2. Non-uniform sampling: Provide time_raw directly (actual timestamps)

    Calibration:
    - Pass calibrate_fn parameter for calibrated data
    - Calibration is lazy (computed on first .val access)
    - Raw data always accessible via .raw property

    Attributes:
        raw_data: Raw measurement values (before calibration)
        freq: Sample frequency in Hz (used for uniform mode, informational for non-uniform)
        t_start: Start time offset in seconds
        name: Optional name/label for the series
        unit: Optional unit string
        time_raw: Optional actual timestamps (for non-uniform sampling)
        calibrate_fn: Optional calibration function (raw -> calibrated)

    Properties:
        raw: Raw values with mask applied
        val: Calibrated values with mask applied (lazy)
        time: Time array with offset and mask applied
    """

    raw_data: np.ndarray
    freq: float
    t_start: float = 0.0
    name: str = ""
    unit: str = ""
    time_raw: np.ndarray | None = None  # Actual timestamps (non-uniform)
    calibrate_fn: Callable[[np.ndarray], np.ndarray] | None = None

    # Private fields (not in __init__)
    _time_computed: np.ndarray | None = field(default=None, repr=False, init=False)
    _val_cached: np.ndarray | None = field(default=None, repr=False, init=False)
    _mask: np.ndarray | None = field(default=None, repr=False, init=False)

    @property
    def raw(self) -> np.ndarray:
        """Raw values with mask applied (before calibration)."""
        if self._mask is None:
            return self.raw_data
        return self.raw_data[self._mask]

    @property
    def val(self) -> np.ndarray:
        """Calibrated values with mask applied (lazy calibration)."""
        v = self._get_calibrated()
        if self._mask is None:
            return v
        return v[self._mask]

    def _get_calibrated(self) -> np.ndarray:
        """Get full calibrated array (lazy computation)."""
        if self.calibrate_fn is None:
            return self.raw_data
        if self._val_cached is None:
            self._val_cached = self.calibrate_fn(self.raw_data)
        return self._val_cached

    @property
    def time(self) -> np.ndarray:
        """Time array with offset and mask applied."""
        t = self._get_time_array()
        if self._mask is None:
            return t
        return t[self._mask]

    def _get_time_array(self) -> np.ndarray:
        """Get full time array (with offset applied)."""
        if self.time_raw is not None:
            # Non-uniform: use actual timestamps + offset
            return self.time_raw + self.t_start
        else:
            # Uniform: compute lazily
            if self._time_computed is None:
                self._time_computed = np.arange(len(self.raw_data)) / self.freq
            return self._time_computed + self.t_start

    @property
    def dt(self) -> float:
        """Nominal sample period in seconds."""
        return 1.0 / self.freq

    def __len__(self) -> int:
        """Length of masked values."""
        if self._mask is None:
            return len(self.raw_data)
        return int(self._mask.sum())

    def crop(self, t_start: float, t_end: float, new_origin: float = 0.0) -> None:
        """
        Crop data to time range and reset time origin.

        After cropping:
        - Only data within [t_start, t_end] is accessible via .val, .time, .raw
        - Time is shifted so the cropped region starts at new_origin (default: 0)

        Args:
            t_start: Start of crop range (in current time coordinates)
            t_end: End of crop range (in current time coordinates)
            new_origin: Time value for the start of cropped region (default: 0)
        """
        t = self._get_time_array()
        self._mask = (t >= t_start) & (t <= t_end)
        # Shift time so crop start becomes new_origin
        self.t_start -= (t_start - new_origin)

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Set mask directly."""
        self._mask = mask

    def clear_mask(self) -> None:
        """Clear mask."""
        self._mask = None

    def raw_time(self) -> np.ndarray:
        """Get full time array without mask."""
        return self._get_time_array()

    def val_unmasked(self) -> np.ndarray:
        """Get full calibrated values without mask."""
        return self._get_calibrated()

    def raw_unmasked(self) -> np.ndarray:
        """Get full raw (uncalibrated) values without mask."""
        return self.raw_data


class MaskableDataMixin:
    """Mixin to propagate cropping to all MeasurementSeries fields."""

    def crop(self, t_start: float, t_end: float, new_origin: float = 0.0) -> None:
        """
        Crop all MeasurementSeries fields to time range and reset time origin.

        Args:
            t_start: Start of crop range (in current time coordinates)
            t_end: End of crop range (in current time coordinates)
            new_origin: Time value for the start of cropped region (default: 0)
        """
        for name in dir(self):
            if name.startswith("_"):
                continue
            val = object.__getattribute__(self, name)
            if isinstance(val, MeasurementSeries):
                val.crop(t_start, t_end, new_origin)

    def clear_masks(self) -> None:
        """Clear masks on all MeasurementSeries fields (keeps time offset)."""
        for name in dir(self):
            if name.startswith("_"):
                continue
            val = object.__getattribute__(self, name)
            if isinstance(val, MeasurementSeries):
                val.clear_mask()
