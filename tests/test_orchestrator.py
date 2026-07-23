import typing

import numpy as np
from src.data_types import ModelInputs, ModelOutputs
from src.orchestrator import Plant, InputFn, run_simulation


class DummyPlant:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def step(self, t, dt, inputs):
        return ModelOutputs(H_T=inputs.y_T, Q_T=inputs.N_T, H_P1=0.0, H_P2=0.0)


class DummyInputFn:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def __call__(self, t, state):
        return ModelInputs(y_T=t, N_T=t * 2, N_P=0.0)


def test_protocols_are_typing_protocol():
    assert getattr(Plant, "_is_protocol", False)
    assert getattr(InputFn, "_is_protocol", False)


def test_run_simulation_calls_reset():
    plant, input_fn = DummyPlant(), DummyInputFn()
    run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.5)
    assert plant.reset_called
    assert input_fn.reset_called


def test_run_simulation_returns_artefact():
    plant, input_fn = DummyPlant(), DummyInputFn()
    artefact = run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.5)
    assert isinstance(artefact.t, np.ndarray)
    assert len(artefact.t) > 0
    assert isinstance(artefact.inputs, dict)
    assert isinstance(artefact.outputs, dict)
    assert isinstance(artefact.metadata, dict)


def test_run_simulation_step_values():
    plant, input_fn = DummyPlant(), DummyInputFn()
    artefact = run_simulation(plant, input_fn, t_span=(0.0, 1.0), dt=0.5)
    assert artefact.outputs["H_T"][0] == artefact.t[0]
