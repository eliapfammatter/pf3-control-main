from controllers import ControllerState
from fpoints_controller import FPointsController
from helpers.fpoints import FPoints


def make_controller(values=(100.0, 200.0, 300.0)):
    fpoints = FPoints.from_arrays(time=[0.0, 1.0, 2.0], values=list(values))
    fpoints.filepath = "data/pf3/REGP"
    return FPointsController(fpoints)


def make_state(**overrides):
    defaults = dict(
        H_T=0.0, H_P1=0.0, H_P2=0.0, N_T=0.0, y_T=0.0,
        N_P=0.0, Q_T=0.0, Q_P1=0.0, Q_P2=0.0, H_tank=0.0,
    )
    defaults.update(overrides)
    return ControllerState(**defaults)


def test_compute_pump_speed_interpolates_between_sample_points():
    controller = make_controller(values=(100.0, 200.0, 300.0))
    assert controller.compute_pump_speed(0.5, make_state()) == 150.0


def test_compute_pump_speed_ignores_state():
    controller = make_controller(values=(100.0, 200.0, 300.0))
    state_a = make_state(N_T=10.0)
    state_b = make_state(N_T=999.0)
    assert (
        controller.compute_pump_speed(0.5, state_a)
        == controller.compute_pump_speed(0.5, state_b)
    )


def test_reset_is_noop():
    controller = make_controller()
    controller.reset()  


def test_name_includes_file_name():
    controller = make_controller()
    assert controller.name == "FPOINTS (REGP)"
