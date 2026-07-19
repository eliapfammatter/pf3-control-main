import numpy as np

from data_types import ModelInputs, ModelOutputs, ModelState, Trajectory, TrajectorySet
from fpoints_controller import FPointsController


def make_trajectory_set(n_p_values=(100.0, 200.0, 300.0)):
    times = np.array([0.0, 1.0, 2.0])
    return TrajectorySet(
        y_T=Trajectory(times=times, values=np.array([0.0, 0.0, 0.0])),
        N_T=Trajectory(times=times, values=np.array([0.0, 0.0, 0.0])),
        H_ref=Trajectory(times=times, values=np.array([0.0, 0.0, 0.0])),
        N_P=Trajectory(times=times, values=np.array(n_p_values)) if n_p_values is not None else None,
    )


def make_controller(n_p_values=(100.0, 200.0, 300.0)):
    return FPointsController(make_trajectory_set(n_p_values))


def make_state(**overrides):
    defaults = dict(y_T=0.0, N_T=0.0, N_P=0.0)
    defaults.update(overrides)
    return ModelState(
        t=0.0,
        inputs=ModelInputs(**defaults),
        outputs=ModelOutputs(H_T=0.0, Q_T=0.0, H_P1=0.0, H_P2=0.0),
    )


def test_call_interpolates_pump_speed_between_sample_points():
    controller = make_controller(n_p_values=(100.0, 200.0, 300.0))
    result = controller(0.5, make_state())
    assert result.N_P == 150.0


def test_call_ignores_state():
    controller = make_controller(n_p_values=(100.0, 200.0, 300.0))
    state_a = make_state(N_T=10.0)
    state_b = make_state(N_T=999.0)
    assert controller(0.5, state_a) == controller(0.5, state_b)


def test_call_accepts_none_state():
    controller = make_controller(n_p_values=(100.0, 200.0, 300.0))
    assert controller(0.5, None).N_P == 150.0


def test_call_holds_pump_speed_at_zero_when_not_scripted():
    controller = make_controller(n_p_values=None)
    result = controller(0.5, make_state())
    assert result.N_P == 0.0


def test_reset_is_noop():
    controller = make_controller()
    controller.reset()
