import numpy as np
from data_types import ModelInputs, TrajectorySet, Trajectory
from fpoints_controller import FPointsController


def make_traj(values):
    times = np.array([0.0, 1.0, 2.0])
    return Trajectory(times=times, values=np.array(values))


def test_call_interpolates_trajectories():
    traj_set = TrajectorySet(
        y_T=make_traj([0.0, 0.5, 1.0]),
        N_T=make_traj([100.0, 200.0, 300.0]),
        H_ref=make_traj([1.0, 2.0, 3.0]),
        N_P=make_traj([50.0, 60.0, 70.0]),
    )
    controller = FPointsController(traj_set)
    inputs = controller(1.0, None)
    assert isinstance(inputs, ModelInputs)
    assert inputs.y_T == 0.5
    assert inputs.N_T == 200.0
    assert inputs.N_P == 60.0


def test_n_p_defaults_to_zero_when_missing():
    traj_set = TrajectorySet(
        y_T=make_traj([0.0, 0.5, 1.0]),
        N_T=make_traj([100.0, 200.0, 300.0]),
        H_ref=make_traj([1.0, 2.0, 3.0]),
        N_P=None,
    )
    controller = FPointsController(traj_set)
    inputs = controller(1.0, None)
    assert inputs.N_P == 0.0


def test_reset_is_noop():
    traj_set = TrajectorySet(
        y_T=make_traj([0.0, 0.5, 1.0]),
        N_T=make_traj([100.0, 200.0, 300.0]),
        H_ref=make_traj([1.0, 2.0, 3.0]),
        N_P=None,
    )
    controller = FPointsController(traj_set)
    controller.reset()  # should not raise
