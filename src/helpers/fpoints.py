import numpy as np


class FPoints:
    """
    Class to read and interpolate SIMSEN FPOINTS files.

    Example:
        >>> fpoints = FPoints("data/REGP") # where data/REGP is a FPOINTS file.
        >>> N = fpoints(15.0)  # Get interpolated value at t=15s
        >>> N
        -305.7861
    """

    def __init__(self, filepath):
        """
        Initialize FPoints reader.

        Parameters
        ----------
        filepath : str
            Path to FPOINTS file
        """
        self.filepath = filepath
        self.x_unit = None
        self.y_unit = None
        self.x_data = []
        self.y_data = []

        # Parse the file
        self._parse_file()

        # Precompute slopes for fast scalar interpolation
        self._dx = np.diff(self.x_data)
        self._dy = np.diff(self.y_data)
        self._slopes = self._dy / self._dx

    @property
    def time(self) -> np.ndarray:
        """Return time array."""
        return self.x_data

    @property
    def values(self) -> np.ndarray:
        """Return values array."""
        return self.y_data

    def resample(self, dt: float, t_end: float) -> "FPoints":
        """
        Resample to a regular time grid.

        Parameters
        ----------
        dt : float
            Time step for resampling
        t_end : float
            End time for resampled trajectory

        Returns
        -------
        FPoints
            Resampled FPoints with regular time grid
        """
        t_new = np.arange(0, t_end + dt, dt)
        y_new = np.array([self(t) for t in t_new])
        return FPoints.from_arrays(t_new, y_new, self.x_unit, self.y_unit)

    @classmethod
    def from_arrays(cls, time: np.ndarray, values: np.ndarray, x_unit: str = "s", y_unit: str = "-"):
        """
        Create an FPoints object from time and value arrays.

        Parameters
        ----------
        time : array-like
            Time points
        values : array-like
            Values at each time point
        x_unit : str
            Unit for time axis (default: 's')
        y_unit : str
            Unit for values (default: '-')

        Returns
        -------
        FPoints
            FPoints object with the given data
        """
        obj = cls.__new__(cls)
        obj.filepath = None
        obj.x_unit = x_unit
        obj.y_unit = y_unit
        obj.x_data = np.asarray(time)
        obj.y_data = np.asarray(values)

        obj._dx = np.diff(obj.x_data)
        obj._dy = np.diff(obj.y_data)
        with np.errstate(divide='ignore', invalid='ignore'):
            obj._slopes = np.where(obj._dx != 0, obj._dy / obj._dx, 0.0)

        return obj

    @classmethod
    def constant(cls, value):
        """
        Create an FPoints object with constant value.

        Parameters
        ----------
        value : float
            Constant value to return for all time steps

        Returns
        -------
        FPoints
            FPoints object that returns constant value

        Example
        -------
        >>> fpoints = FPoints.constant(5.0)
        >>> fpoints(10.0)
        5.0
        >>> fpoints(100.0)
        5.0
        """
        # Create a minimal FPoints object that doesn't read from file
        obj = cls.__new__(cls)
        obj.filepath = None
        obj.x_unit = "s"
        obj.y_unit = "-"
        obj.x_data = np.array([0.0, 1e10])  # Wide time range
        obj.y_data = np.array([value, value])  # Constant value

        # Precompute slopes (will be zero)
        obj._dx = np.diff(obj.x_data)
        obj._dy = np.diff(obj.y_data)
        obj._slopes = obj._dy / obj._dx

        return obj

    @classmethod
    def steps_abs(cls, initial_value, t_steps, values, t_end=1e10):
        """
        Create an FPoints object with absolute step changes.

        Starts at initial_value, then jumps to each absolute value at corresponding t_step.

        Parameters
        ----------
        initial_value : float
            Value before any steps
        t_steps : list of float
            Times at which each step occurs
        values : list of float
            Absolute values to step to at each time (must be same length as t_steps)
        t_end : float, optional
            End time (default: 1e10)

        Returns
        -------
        FPoints
            FPoints object with step changes

        Examples
        --------
        >>> fpoints = FPoints.steps_abs(-313.0, [2.0], [-320.0])
        >>> fpoints(1.0)
        -313.0
        >>> fpoints(5.0)
        -320.0

        >>> fpoints = FPoints.steps_abs(100.0, [2.0, 5.0, 8.0], [110.0, 105.0, 125.0])
        >>> fpoints(0.0)   # before any steps
        100.0
        >>> fpoints(3.0)   # after first step
        110.0
        >>> fpoints(6.0)   # after second step
        105.0
        >>> fpoints(10.0)  # after third step
        125.0
        """
        if len(values) != len(t_steps):
            raise ValueError(
                f"t_steps and values must have same length, "
                f"got {len(t_steps)} and {len(values)}"
            )

        obj = cls.__new__(cls)
        obj.filepath = None
        obj.x_unit = "s"
        obj.y_unit = "-"

        # Build x_data and y_data arrays
        # Format: [0, t1, t1, t2, t2, ..., tn, tn, t_end]
        # Values: [v0, v0, v1, v1, v2, ..., vn-1, vn, vn]
        x_list = [0.0]
        y_list = [initial_value]

        current_value = initial_value
        for t, val in zip(t_steps, values):
            # Before step: hold previous value
            x_list.append(t)
            y_list.append(current_value)
            # After step: new value
            current_value = val
            x_list.append(t)
            y_list.append(current_value)

        # Final segment to t_end
        x_list.append(t_end)
        y_list.append(current_value)

        obj.x_data = np.array(x_list)
        obj.y_data = np.array(y_list)

        obj._dx = np.diff(obj.x_data)
        obj._dy = np.diff(obj.y_data)
        with np.errstate(divide='ignore', invalid='ignore'):
            obj._slopes = np.where(obj._dx != 0, obj._dy / obj._dx, 0.0)

        return obj

    @classmethod
    def linear(cls, start_value, end_value, t_start, t_end):
        """
        Create an FPoints object with a single linear ramp.

        Parameters
        ----------
        start_value : float
            Value at t_start
        end_value : float
            Value at t_end
        t_start : float
            Start time of ramp
        t_end : float
            End time of ramp

        Returns
        -------
        FPoints
            FPoints object that ramps linearly from start_value to end_value

        Examples
        --------
        >>> fpoints = FPoints.linear(100.0, 200.0, 2.0, 4.0)
        >>> fpoints(2.0)
        100.0
        >>> fpoints(3.0)
        150.0
        >>> fpoints(4.0)
        200.0
        """
        return cls.from_arrays(
            np.array([t_start, t_end]),
            np.array([start_value, end_value]),
        )

    @classmethod
    def steps_rel(cls, initial_value, t_steps, deltas, t_end=1e10):
        """
        Create an FPoints object with relative step changes.

        Starts at initial_value, then applies each delta at corresponding t_step.
        Values accumulate: after step i, value = initial_value + sum(deltas[:i+1]).

        Parameters
        ----------
        initial_value : float
            Value before any steps
        t_steps : list of float
            Times at which each step occurs
        deltas : list of float
            Change in value at each step (must be same length as t_steps)
        t_end : float, optional
            End time (default: 1e10)

        Returns
        -------
        FPoints
            FPoints object with step changes

        Examples
        --------
        >>> fpoints = FPoints.steps_rel(-313.0, [2.0], [-7.0])
        >>> fpoints(1.0)
        -313.0
        >>> fpoints(5.0)
        -320.0

        >>> fpoints = FPoints.steps_rel(100.0, [2.0, 5.0, 8.0], [10.0, -5.0, 20.0])
        >>> fpoints(0.0)   # before any steps
        100.0
        >>> fpoints(3.0)   # after first step: 100 + 10 = 110
        110.0
        >>> fpoints(6.0)   # after second step: 110 - 5 = 105
        105.0
        >>> fpoints(10.0)  # after third step: 105 + 20 = 125
        125.0
        """
        # Convert deltas to absolute values
        values = []
        current = initial_value
        for delta in deltas:
            current += delta
            values.append(current)
        return cls.steps_abs(initial_value, t_steps, values, t_end)

    def _parse_file(self):
        """Parse DATA section of FPOINTS file."""
        with open(self.filepath, "r") as f:
            content = f.read()

        in_data_section = False
        for line in content.split("\n"):
            if "- DATA :" in line:
                in_data_section = True
                continue

            if not in_data_section:
                continue

            line = line.strip()
            if line.startswith("x") and ":" in line:
                # Split by whitespace: ['x1', '[s]', ':', '0.0', 'y1', '[rpm]', ':', '-313.2579']
                parts = line.split()

                if len(parts) >= 8:
                    # Extract units from first data point
                    if self.x_unit is None:
                        self.x_unit = parts[1].strip("[]")
                        self.y_unit = parts[5].strip("[]")

                    # Extract values (simple index access)
                    x_val = float(parts[3])
                    y_val = float(parts[7])

                    self.x_data.append(x_val)
                    self.y_data.append(y_val)

        # Convert to numpy arrays
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

        if len(self.x_data) == 0:
            raise ValueError(f"No data points found in {self.filepath}")

    def __call__(self, x):
        """
        Evaluate the function at given x value(s).

        Uses searchsorted + precomputed slopes for fast scalar lookup.
        Extrapolates linearly beyond data range.

        Parameters
        ----------
        x : float or array-like
            Point(s) at which to evaluate the function

        Returns
        -------
        float or ndarray
            Interpolated y value(s)
        """
        idx = np.searchsorted(self.x_data, x) - 1
        # Clamp to valid segment range [0, n-2] (extrapolates linearly)
        last = len(self.x_data) - 2
        if np.ndim(x) == 0:
            # Scalar fast path (common case in ODE integration)
            if idx < 0:
                idx = 0
            elif idx > last:
                idx = last
        else:
            idx = np.clip(idx, 0, last)
        return self.y_data[idx] + self._slopes[idx] * (x - self.x_data[idx])


if __name__ == "__main__":
    from datetime import datetime

    import plotly.graph_objects as go

    # Example usage
    fpoints = FPoints("data/pf3/REGP")
    print(f"Loaded FPOINTS: {fpoints.filepath}")
    print(f"  Data points: {len(fpoints.x_data)}")
    print(
        f"  X range: [{fpoints.x_data[0]:.2f}, {fpoints.x_data[-1]:.2f}] {fpoints.x_unit}"
    )
    print(
        f"  Y range: [{fpoints.y_data.min():.2f}, {fpoints.y_data.max():.2f}] {fpoints.y_unit}"
    )
    print()

    # Evaluate at specific points
    print("Evaluating at specific time points:")
    for t in [0, 10, 20, 30, 40, 50]:
        speed = fpoints(t)
        print(f"  t = {t:5.1f} s → N = {speed:8.2f} rpm")

    print()
    print("Interpolating between data points:")
    print(f"  t = 15.5 s → N = {fpoints(15.5):.2f} rpm")
    print(f"  t = 25.8 s → N = {fpoints(25.8):.2f} rpm")

    # Plot the function with Plotly
    print()
    print("Generating interactive plot...")

    # Smooth interpolated curve
    x_smooth = np.linspace(fpoints.x_data[0], fpoints.x_data[-1], 500)
    y_smooth = fpoints(x_smooth)

    fig = go.Figure()

    # Add interpolated curve
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name="Interpolated",
            line=dict(color="blue", width=2),
        )
    )

    # Add original data points
    fig.add_trace(
        go.Scatter(
            x=fpoints.x_data,
            y=fpoints.y_data,
            mode="markers",
            name="Data points",
            marker=dict(size=6, color="red", opacity=0.7),
        )
    )

    fig.update_layout(
        title=f"FPOINTS: {fpoints.filepath}",
        xaxis_title=f"Time [{fpoints.x_unit}]",
        yaxis_title=f"Speed [{fpoints.y_unit}]",
        hovermode="x unified",
        showlegend=True,
    )

    # Save as HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"plots/fpoints_test_{timestamp}.html"
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {output_file}")

    # Show in browser
    fig.show()
