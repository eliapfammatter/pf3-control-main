"""Quick check: does generated/zero_shot/data_types.py work with real SIMSEN output?

Connects to the running SIMSEN FMU (same way run_fmi_control.py does), takes one
simulation step with fixed inputs, and wraps the raw FMU values into the
generated ModelInputs / ModelOutputs dataclasses. If this runs with no errors
and prints sane numbers, the data_types module is compatible with SIMSEN's
FMU interface.

Run from the repo root (pf3-control-main), with your venv active and
RFMI_SERVER_SIMSEN set:

    python check_data_types_simsen.py
"""

import sys
from pathlib import Path

from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

# Make the generated zero-shot module importable.
sys.path.insert(0, str(Path("generated/zero_shot").resolve()))
from data_types import ModelInputs, ModelOutputs  # noqa: E402

FMU_PATH = Path("data/pf3/PF3_FMI.fmu")

# Known-good rig values (same as used in run_fmi_control.py).
Y_T_TEST = 0.4706
N_T_TEST = 369.3346
N_P_TEST = -313.2579

START_TIME = 0.0
STEP_SIZE = 0.1


def main() -> int:
    if not FMU_PATH.exists():
        print(f"error: FMU not found at {FMU_PATH}", file=sys.stderr)
        return 2

    model_description = read_model_description(FMU_PATH, validate=False)

    in_vars, out_vars = {}, {}
    for var in model_description.modelVariables:
        if var.causality == "input":
            in_vars[var.name] = var.valueReference
        elif var.causality == "output":
            out_vars[var.name] = var.valueReference

    unzipdir = extract(FMU_PATH)
    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.modelIdentifier,
        instanceName="data_types_check",
    )

    fmu.instantiate()
    fmu.setupExperiment(startTime=START_TIME)
    fmu.enterInitializationMode()
    fmu.setReal(
        [in_vars[k] for k in ["PUMP1-N", "PUMP2-N", "TURB-N", "TURB-y"]],
        [N_P_TEST, N_P_TEST, N_T_TEST, Y_T_TEST],
    )
    fmu.exitInitializationMode()

    fmu.doStep(currentCommunicationPoint=START_TIME, communicationStepSize=STEP_SIZE)

    H_T = fmu.getReal([out_vars["FTURB1-H"]])[0]
    Q_T = fmu.getReal([out_vars["FTURB1-Q"]])[0]
    H_P1 = fmu.getReal([out_vars["FPUMP1-H"]])[0]
    H_P2 = fmu.getReal([out_vars["FPUMP2-H"]])[0]

    fmu.terminate()
    fmu.freeInstance()

    # This is the actual compatibility check: build the generated dataclasses
    # straight from the real FMU values.
    inputs = ModelInputs(y_T=Y_T_TEST, N_T=N_T_TEST, N_P=N_P_TEST)
    outputs = ModelOutputs(H_T=H_T, Q_T=Q_T, H_P1=H_P1, H_P2=H_P2)

    print("ModelInputs :", inputs)
    print("ModelOutputs:", outputs)
    print("\nOK: data_types.py accepted real SIMSEN values with no errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
