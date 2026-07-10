import numpy as np
import pytest

from src.generated.zero_shot.data_types import ModelInputs, ModelOutputs
from src.generated.zero_shot.orchestrator import run_simulation


class IntegratorPlant:
    """Plant that accumulates y_T over time, like a simple integrator."""

    def __init__(self):
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def step(self, t, dt, inputs):
        self.value += inputs.y_T * dt
        return ModelOutputs(H_T=self.value, Q_T=inputs.N_T, H_P1=0.0, H_P2=0.0)


class FeedbackInputFn:
    """Input fn that reacts to the previous step's H_T output."""

    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def __call__(self, t, state):
        self.calls += 1
        prev_h = state.outputs.H_T if state is not None else 0.0
        return ModelInputs(y_T=1.0, N_T=prev_h, N_P=0.0)


def test_run_simulation_integration_end_to_end():
    plant, input_fn = IntegratorPlant(), FeedbackInputFn()
    artefact = run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.25)

    assert len(artefact.t) == 4
    np.testing.assert_allclose(artefact.t, [0.0, 0.25, 0.5, 0.75])
    # H_T is a running integral of y_T=1.0 over dt=0.25
    np.testing.assert_allclose(artefact.outputs["H_T"], [0.25, 0.5, 0.75, 1.0])
    # input N_T at step i mirrors output H_T at step i-1 (0.0 on the first step)
    np.testing.assert_allclose(artefact.inputs["N_T"][1:], artefact.outputs["H_T"][:-1])
    assert artefact.inputs["N_T"][0] == 0.0
    assert input_fn.calls == 4


def test_run_simulation_resets_state_between_runs():
    plant, input_fn = IntegratorPlant(), FeedbackInputFn()
    run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.25)
    artefact = run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.25)

    assert artefact.outputs["H_T"][0] == pytest.approx(0.25)
    assert input_fn.calls == 4


@pytest.mark.parametrize(
    "t_span, dt",
    [((0.0, 1.0), 0.0), ((0.0, 1.0), -0.1), ((1.0, 0.0), 0.1), ((1.0, 1.0), 0.1)],
)
def test_run_simulation_invalid_arguments_raise(t_span, dt):
    with pytest.raises(ValueError):
        run_simulation(IntegratorPlant(), FeedbackInputFn(), t_span=t_span, dt=dt)
