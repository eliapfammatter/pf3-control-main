import numpy as np
import pytest as pytest
from data_types import Trajectory


def test_trajectory_interpolates_linearly():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 10.0, 20.0])
    traj = Trajectory(times=times, values=values)
    assert traj(0.5) == np.interp(0.5, times, values)
    assert traj(1.5) == np.interp(1.5, times, values)


def test_trajectory_clamps_outside_sample_range():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 10.0, 20.0])
    traj = Trajectory(times=times, values=values)
    assert traj(-1.0) == values[0]
    assert traj(5.0) == values[-1]


def test_trajectory_raises_on_empty_samples():
    traj = Trajectory(times=np.array([]), values=np.array([]))
    with pytest.raises(ValueError):
        traj(0.0)
