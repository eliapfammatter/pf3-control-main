"""Integration test running the zero-shot FPointsController against real SIMSEN via FMI.

Skipped unless the FMU is present and RFMI_SERVER_SIMSEN points to a running
SimsenRFMIServer.exe. See README.md "SIMSEN FMU setup" for setup instructions.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src.generated.zero_shot.data_types import ModelInputs, ModelOutputs, Trajectory, TrajectorySet
from src.generated.zero_shot.fpoints_controller import FPointsController
from src.generated.zero_shot.orchestrator import run_simulation

fmpy = pytest.importorskip("fmpy")
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

REPO_ROOT = Path(__file__).resolve().parent.parent
FMU_PATH = REPO_ROOT / "data/pf3/PF3_FMI.fmu"

_missing = []
if not FMU_PATH.exists():
    _missing.append(f"FMU not found at {FMU_PATH}")
if not os.environ.get("RFMI_SERVER_SIMSEN"):
    _missing.append("RFMI_SERVER_SIMSEN is not set")

pytestmark = pytest.mark.skipif(bool(_missing), reason="; ".join(_missing))

INPUT_NAMES = ["PUMP1-N", "PUMP2-N", "TURB-N", "TURB-y"]
N_P_INIT = -313.2579
N_T_INIT = 369.3346
Y_T_INIT = 0.4706


class SimsenPlant:
    """Plant wrapping the real SIMSEN FMU, conforming to the orchestrator's Plant protocol."""

    def __init__(self, fmu_path: Path = FMU_PATH):
        self.fmu_path = fmu_path
        self._fmu = None
        self._in_vars = {}
        self._out_vars = {}

    def reset(self):
        if self._fmu is not None:
            self._fmu.terminate()
            self._fmu.freeInstance()

        model_description = read_model_description(self.fmu_path, validate=False)
        self._in_vars = {
            v.name: v.valueReference
            for v in model_description.modelVariables
            if v.causality == "input"
        }
        self._out_vars = {
            v.name: v.valueReference
            for v in model_description.modelVariables
            if v.causality == "output"
        }

        unzipdir = extract(self.fmu_path)
        self._fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.modelIdentifier,
            instanceName="fpoints_controller_integration_test",
        )
        self._fmu.instantiate()
        self._fmu.setupExperiment(startTime=0.0)
        self._fmu.enterInitializationMode()
        self._fmu.setReal(
            [self._in_vars[k] for k in INPUT_NAMES],
            [N_P_INIT, N_P_INIT, N_T_INIT, Y_T_INIT],
        )
        self._fmu.exitInitializationMode()

    def step(self, t, dt, inputs: ModelInputs) -> ModelOutputs:
        self._fmu.setReal(
            [self._in_vars[k] for k in INPUT_NAMES],
            [inputs.N_P, inputs.N_P, inputs.N_T, inputs.y_T],
        )
        self._fmu.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
        H_T = self._fmu.getReal([self._out_vars["FTURB1-H"]])[0]
        Q_T = self._fmu.getReal([self._out_vars["FTURB1-Q"]])[0]
        H_P1 = self._fmu.getReal([self._out_vars["FPUMP1-H"]])[0]
        H_P2 = self._fmu.getReal([self._out_vars["FPUMP2-H"]])[0]
        return ModelOutputs(H_T=H_T, Q_T=Q_T, H_P1=H_P1, H_P2=H_P2)


def make_constant_trajectories() -> TrajectorySet:
    times = np.array([0.0, 1.0])
    return TrajectorySet(
        y_T=Trajectory(times=times, values=np.array([Y_T_INIT, Y_T_INIT])),
        N_T=Trajectory(times=times, values=np.array([N_T_INIT, N_T_INIT])),
        H_ref=None,
        N_P=Trajectory(times=times, values=np.array([N_P_INIT, N_P_INIT])),
    )


def test_run_simulation_with_fpoints_controller_against_real_simsen():
    plant = SimsenPlant()
    controller = FPointsController(make_constant_trajectories())
    try:
        artefact = run_simulation(plant, controller, t_span=(0.0, 1.0), dt=0.1)
    finally:
        if plant._fmu is not None:
            plant._fmu.terminate()
            plant._fmu.freeInstance()

    assert len(artefact.t) == 10
    assert np.all(np.isfinite(artefact.outputs["H_T"]))
    assert np.all(np.isfinite(artefact.outputs["Q_T"]))
