"""
Compare 140-state Native Model vs SIMSEN FMU.

Supports multiple scenarios with different FPOINTS inputs and FMUs.

Usage:
    python compare_140state_vs_fmu.py [--stop-time 20] [--no-cache] [--scenario SCENARIO]

Scenarios:
    fast_transition - Original fast_transition scenario (default)
    multistep       - Multi-step test with inputs held after t=5.2s

View live updates:
    feh -R 1 --scale-down output/compare_140state_vs_fmu.png
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

from src.characteristics import Characteristics
from src.helpers import FPoints
from src.pf3_full_network import build_pf3_full
from src.pf3_lumped_casadi import PF3LumpedCasadi, PF3Simulator
from src.pf3_reduced_model import build_pf3_reduced_model

# =============================================================================
# Scenario Configuration
# =============================================================================
DATA_DIR = Path("data/pf3")
OUTPUT_DIR = Path("output")
CACHE_DIR = OUTPUT_DIR / "cache"

# Define available scenarios
SCENARIOS = {
    "fast_transition": {
        "fmu": DATA_DIR / "PF3_2_pumps_PAR_STORE_fmi.fmu",
        "fpoints_y_T": DATA_DIR / "REGY",
        "fpoints_N_T": DATA_DIR / "REGN",
        "fpoints_N_P": DATA_DIR / "REGP",
        "description": "Original fast_transition scenario",
    },
    "multistep": {
        "fmu": DATA_DIR / "multi-step-test" / "PF3_2_pumps_PAR_STORE_fmi_multistep.fmu",
        "fpoints_y_T": DATA_DIR / "multi-step-test" / "REGY",
        "fpoints_N_T": DATA_DIR / "multi-step-test" / "REGN",
        "fpoints_N_P": DATA_DIR / "multi-step-test" / "REGP",
        "description": "Multi-step test with inputs held after t=5.2s",
    },
}

# Available REGP files for pump speed trajectory
REGP_OPTIONS = {
    "REGP": DATA_DIR / "REGP",
    "REGP_NMPC": DATA_DIR / "REGP_NMPC",
}


def create_turbine_and_pump(interp_method: str = "linear"):
    """Create turbine and pump PumpTurbine objects.

    Args:
        interp_method: Interpolation method ("linear", "clough_tocher", or "structured")
    """
    turbine = Characteristics(
        d_ref=0.3477,
        h_n=5.0,
        q_n=0.1959,
        t_n=224.057,
        n_n=369.3346,
        char_file=Path("data/pf3/missing_files/STORE_4_quadrant_characteristic.txt"),
        interp_method=interp_method,
    )
    pump = Characteristics(
        d_ref=0.535,
        h_n=50.0,
        q_n=1.30,
        t_n=1280.3,
        n_n=790.0,
        char_file=Path("data/pf3/missing_files/PF3_FP_D_535_ext.txt"),
        interp_method=interp_method,
    )
    return turbine, pump


def run_fmu_simulation(
    fmu_file: Path,
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    fpoints_N_P: FPoints,
    start_time: float,
    stop_time: float,
    step_size: float = 0.01,
) -> Dict[str, Any]:
    """Run SIMSEN FMU simulation."""

    model_description = read_model_description(fmu_file, validate=False)

    in_vars = {}
    out_vars = {}
    for var in model_description.modelVariables:
        if var.causality == "input":
            in_vars[var.name] = var.valueReference
        elif var.causality == "output":
            out_vars[var.name] = var.valueReference

    print(f"\nFMU: {model_description.modelIdentifier}")
    print(f"Inputs: {list(in_vars.keys())}")

    unzipdir = extract(fmu_file)
    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.modelIdentifier,
        instanceName="simsen",
    )

    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    # Set initial pump speed (after initialization mode)
    pump_input_names = [k for k in in_vars.keys() if "PUMP" in k and "-N" in k]
    N_P_init = fpoints_N_P(start_time)
    if pump_input_names:
        fmu.setReal(
            [in_vars[k] for k in pump_input_names], [N_P_init] * len(pump_input_names)
        )

    # Storage
    timestamps = []
    outputs = {name: [] for name in out_vars}
    inputs_record = {"N_P": [], "y_T": [], "N_T": []}

    time = start_time
    while time < stop_time:
        N_P = fpoints_N_P(time)

        if pump_input_names:
            fmu.setReal(
                [in_vars[k] for k in pump_input_names], [N_P] * len(pump_input_names)
            )

        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
        time += step_size

        timestamps.append(time)
        inputs_record["N_P"].append(N_P)
        inputs_record["y_T"].append(fpoints_y_T(time))
        inputs_record["N_T"].append(fpoints_N_T(time))

        for name in out_vars:
            outputs[name].append(fmu.getReal([out_vars[name]])[0])

        if int(time * 10) % 10 == 0:
            H_T = outputs["FTURB1-H"][-1]
            print(f"[FMU] t={time:6.1f}s | N_P={N_P:.1f} | H_T={H_T:.3f}m")

    fmu.terminate()
    fmu.freeInstance()

    return {
        "timestamps": np.array(timestamps),
        "outputs": {k: np.array(v) for k, v in outputs.items()},
        "inputs": {k: np.array(v) for k, v in inputs_record.items()},
    }


def run_native_140state(
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    fpoints_N_P: FPoints,
    start_time: float,
    stop_time: float,
    chunk_duration: float = 1.0,
    interp_method: str = "linear",
) -> Dict[str, Any]:
    """Run native 140-state distributed model."""

    print("\nBuilding native 140-state network...")
    network = build_pf3_full(
        interp_method=interp_method,
        data_dir=Path("data/pf3/passive_elements"),
    )

    y0 = network.get_initial_state()
    print(f"   State dimension: {len(y0)}")

    # DEBUG: Check STANK configuration
    print(f"\n   DEBUG STANK:")
    print(f"     STANK node in network: {'STANK' in network.nodes}")
    if "STANK" in network.nodes:
        stank_node = network.nodes["STANK"]
        print(f"     STANK node_type: {stank_node.node_type}")
        print(f"     STANK node C: {stank_node.C}")
        print(f"     STANK node H0: {stank_node.H0}")
    print(f"     _cap_nodes: {network._cap_nodes}")
    print(f"     STANK in _cap_nodes: {'STANK' in network._cap_nodes}")
    if "STANK" in network._H_indices:
        stank_H_idx = network._H_indices["STANK"]
        stank_cap_idx = network._cap_nodes.index("STANK")
        print(f"     STANK H state index: {stank_H_idx}")
        print(f"     C_diag[STANK]: {network._C_diag[stank_cap_idx]}")
        # Check A_c row for STANK - which branches connect?
        A_c_stank = network._A_c[stank_cap_idx, :]
        nonzero_idx = np.where(A_c_stank != 0)[0]
        print(f"     A_c[STANK] has {len(nonzero_idx)} nonzero entries:")
        for bidx in nonzero_idx:
            bname = network._branch_order[bidx]
            print(f"       Branch {bidx} ({bname}): coef={A_c_stank[bidx]:+.0f}")
    else:
        print(f"     WARNING: STANK not in _H_indices!")

    def y_T_func(t):
        return fpoints_y_T(t)

    def N_T_func(t):
        return fpoints_N_T(t)

    def N_P_func(t):
        return fpoints_N_P(t)

    print(f"\nRunning native simulation t=[{start_time}, {stop_time}]s...")

    current_time = start_time
    current_state = y0
    all_t = []
    all_y = []

    while current_time < stop_time:
        chunk_end = min(current_time + chunk_duration, stop_time)

        # Use RK45 with 0.1ms max step (like SIMSEN), output at 1ms for plotting
        chunk_t_eval = np.arange(current_time, chunk_end + 0.0005, 0.001)
        chunk_t_eval = chunk_t_eval[chunk_t_eval <= chunk_end]

        def external_inputs(t):
            return {
                "y_T": y_T_func(t),
                "N_T": N_T_func(t),
                "N_P": N_P_func(t),
                "y_P1": 1.0,
                "y_P2": 1.0,
            }

        result = network.simulate(
            t_span=(current_time, chunk_end),
            y0=current_state,
            t_eval=chunk_t_eval,
            external_inputs_func=external_inputs,
            method="RK45",
            max_step=0.0001,  # 0.1ms max step like SIMSEN
        )

        if len(all_t) == 0:
            all_t.extend(result.t.tolist())
            all_y.append(result.y)
        else:
            new_mask = result.t > all_t[-1] + 1e-9
            if np.any(new_mask):
                all_t.extend(result.t[new_mask].tolist())
                all_y.append(result.y[:, new_mask])

        current_state = result.y[:, -1]
        current_time = chunk_end
        print(f"   t = {current_time:.1f}s / {stop_time:.1f}s")

    t_arr = np.array(all_t)
    y_arr = np.hstack(all_y)

    # Extract Q_T (turbine flow at CONE)
    Q_cone_idx = None
    Q_P1_idx = None
    Q_P2_idx = None
    for name, idx in network._Q_indices.items():
        if "CONE" in name and Q_cone_idx is None:
            Q_cone_idx = idx
        elif "LP1" in name and Q_P1_idx is None:
            Q_P1_idx = idx
        elif "LP2" in name and Q_P2_idx is None:
            Q_P2_idx = idx

    Q_T = y_arr[Q_cone_idx, :] if Q_cone_idx is not None else y_arr[0, :]
    Q_P1 = y_arr[Q_P1_idx, :] if Q_P1_idx is not None else np.zeros_like(t_arr)
    Q_P2 = y_arr[Q_P2_idx, :] if Q_P2_idx is not None else np.zeros_like(t_arr)

    # Compute H_T from characteristic + dynamic term (L × dQ/dt)
    # This matches SIMSEN's FTURB-H output which is the total head across the LRH element
    from src.hydraulic_elements.pump_turbine import PumpTurbine

    turbine, pump = create_turbine_and_pump(interp_method=interp_method)
    fturb1 = PumpTurbine.from_dat(Path("data/pf3/passive_elements") / "FTURB1.DAT")
    turb_L = fturb1.hydraulic_L()

    n = len(t_arr)
    H_T = np.zeros(n)
    dQ_dt = np.zeros(n)
    dt = np.diff(t_arr)
    dQ_dt[1:-1] = (Q_T[2:] - Q_T[:-2]) / (t_arr[2:] - t_arr[:-2])
    if n > 1:
        dQ_dt[0] = (Q_T[1] - Q_T[0]) / dt[0] if dt[0] > 0 else 0
        dQ_dt[-1] = (Q_T[-1] - Q_T[-2]) / dt[-1] if dt[-1] > 0 else 0
    for i in range(n):
        H_T[i] = (
            turbine.compute_H(y_T_func(t_arr[i]), N_T_func(t_arr[i]), Q_T[i])
            + turb_L * dQ_dt[i]
        )

    # Pump heads (pumps have Lequ=0, so no dynamic term needed)
    H_P1 = np.array(
        [pump.compute_H(1.0, N_P_func(ti), -Q_P1[i]) for i, ti in enumerate(t_arr)]
    )
    H_P2 = np.array(
        [pump.compute_H(1.0, N_P_func(ti), -Q_P2[i]) for i, ti in enumerate(t_arr)]
    )

    # Extract H_STANK from state vector (capacitive node)
    H_STANK_idx = network._H_indices.get("STANK", None)
    if H_STANK_idx is not None:
        H_STANK = y_arr[H_STANK_idx, :]
    else:
        H_STANK = np.zeros_like(t_arr)
        print("   WARNING: STANK H index not found")

    # Extract flows at STANK connection: Q_L1 (first seg) and Q_ELBOW (last seg)
    # Find branch indices for L1_Q0 and ELBOW's last segment
    Q_L1_at_STANK = np.zeros_like(t_arr)
    Q_ELBOW_at_STANK = np.zeros_like(t_arr)

    # Find L1_Q0 (first segment of L1, connected to STANK)
    L1_Q0_idx = network._Q_indices.get("L1_Q0", None)
    if L1_Q0_idx is not None:
        Q_L1_at_STANK = y_arr[L1_Q0_idx, :]
        print(f"   Found L1_Q0 at index {L1_Q0_idx}, Q0={Q_L1_at_STANK[0]:.6f} m³/s")
    else:
        print("   WARNING: L1_Q0 not found in Q_indices")

    # Find ELBOW's last segment (connected to STANK)
    # ELBOW is a distributed pipe, need to find ELBOW_Q{Nb}
    elbow_last_idx = None
    for name, idx in network._Q_indices.items():
        if name.startswith("ELBOW_Q") and not any(c.isalpha() for c in name[7:]):
            # Extract segment number
            try:
                seg_num = int(name[7:])
                if elbow_last_idx is None or seg_num > int(
                    list(network._Q_indices.keys())[elbow_last_idx].split("_Q")[1]
                ):
                    elbow_last_idx = idx
                    elbow_last_name = name
            except (ValueError, IndexError):
                pass

    # Actually, let's just find all ELBOW_Q* and pick the highest number
    elbow_branches = [
        (name, idx)
        for name, idx in network._Q_indices.items()
        if name.startswith("ELBOW_Q")
    ]
    if elbow_branches:
        # Sort by segment number and get the last one
        elbow_branches.sort(key=lambda x: int(x[0].split("_Q")[1]))
        elbow_last_name, elbow_last_idx = elbow_branches[-1]
        Q_ELBOW_at_STANK = y_arr[elbow_last_idx, :]
        print(
            f"   Found {elbow_last_name} at index {elbow_last_idx}, Q0={Q_ELBOW_at_STANK[0]:.6f} m³/s"
        )
    else:
        print("   WARNING: ELBOW_Q* not found in Q_indices")

    # Compute Q_STANK = Q_ELBOW - Q_L1 (flow INTO tank)
    # Based on incidence matrix: A[STANK, L1_Q0] = +1, A[STANK, ELBOW_last] = -1
    # dH/dt = (1/A) * (-A_row @ Q) = (1/A) * (Q_ELBOW - Q_L1)
    Q_STANK_native = Q_ELBOW_at_STANK - Q_L1_at_STANK
    print(
        f"   Q_STANK at t=0: Q_ELBOW({Q_ELBOW_at_STANK[0]:.6f}) - Q_L1({Q_L1_at_STANK[0]:.6f}) = {Q_STANK_native[0]:.6f} m³/s"
    )

    # Extract penstock heads along the path: STANK → L1 → L2 → L3 → L4 → NODE1
    # In SIMSEN, HNb = head at last internal capacitor node
    # For Nb=0 pipes (lumped), there are no Hc nodes - need junction head at outlet
    # Pipe topology: L1(STANK→j_L1_E1), L2(j_E1_L2→j_L2_E2), L3(j_E2_L3→j_E3_L4), L4(j_E3_L4→NODE1)
    penstock_heads = {}

    # First, compute junction heads for all timesteps (needed for Nb=0 pipes)
    print("   Computing junction heads for penstock extraction...")
    all_junction_heads = {}  # {junction_name: array of heads}
    for i, ti in enumerate(t_arr):
        state = y_arr[:, i]
        external_inputs = {
            "y_T": y_T_func(ti),
            "N_T": N_T_func(ti),
            "N_P": N_P_func(ti),
        }
        try:
            junc_heads = network.get_junction_heads(state, ti, external_inputs)
            for jname, jhead in junc_heads.items():
                if jname not in all_junction_heads:
                    all_junction_heads[jname] = np.zeros_like(t_arr)
                all_junction_heads[jname][i] = jhead
        except Exception as e:
            if i == 0:
                print(f"   WARNING: Junction head computation failed: {e}")

    print(f"   Available junctions: {list(all_junction_heads.keys())}")

    # Map pipe names to their outlet nodes (for Nb=0 case)
    pipe_outlet_junctions = {
        "L1": "j_L1_E1",
        "L2": "j_L2_E2",
        "L3": "j_E3_L4",
        "L4": "NODE1",
    }

    for pipe_name in ["L1", "L2", "L3", "L4"]:
        # Find the last Hc node for this pipe
        hc_nodes = [
            (name, idx)
            for name, idx in network._H_indices.items()
            if name.startswith(f"{pipe_name}_Hc")
        ]
        if hc_nodes:
            # Distributed pipe (Nb > 0): use last internal Hc node
            hc_nodes.sort(key=lambda x: int(x[0].split("_Hc")[1]))
            last_hc_name, last_hc_idx = hc_nodes[-1]
            penstock_heads[f"{pipe_name}-HNb"] = y_arr[last_hc_idx, :]
            print(
                f"   Found {last_hc_name} for {pipe_name}-HNb, H0={y_arr[last_hc_idx, 0]:.6f} m"
            )
        else:
            # Lumped pipe (Nb=0): no internal Hc nodes exist
            # Skip comparison - FMU outputs inlet head, but there's no equivalent
            # internal node in the native model to compare against
            print(f"   Skipping {pipe_name}-HNb (Nb=0, no internal Hc nodes)")

    # Extract ELBOW-H1, CONE-H1, and L12-H1 (first Hc node of each)
    for pipe_name in ["ELBOW", "CONE", "L12"]:
        hc_nodes = [
            (name, idx)
            for name, idx in network._H_indices.items()
            if name.startswith(f"{pipe_name}_Hc")
        ]
        if hc_nodes:
            # Sort by Hc number and get the first one (H1)
            hc_nodes.sort(key=lambda x: int(x[0].split("_Hc")[1]))
            first_hc_name, first_hc_idx = hc_nodes[0]
            penstock_heads[f"{pipe_name}-H1"] = y_arr[first_hc_idx, :]
            print(
                f"   Found {first_hc_name} for {pipe_name}-H1, H0={y_arr[first_hc_idx, 0]:.6f} m"
            )
        else:
            penstock_heads[f"{pipe_name}-H1"] = np.zeros_like(t_arr)
            print(f"   WARNING: No Hc nodes found for {pipe_name}")

    # Use precomputed junction heads for NODE1, NODE2
    H_NODE1 = all_junction_heads.get("NODE1", np.zeros_like(t_arr))
    H_NODE2 = all_junction_heads.get("NODE2", np.zeros_like(t_arr))
    print(f"   NODE1 H0={H_NODE1[0]:.6f} m, NODE2 H0={H_NODE2[0]:.6f} m")

    return {
        "timestamps": t_arr,
        "H_T": H_T,
        "Q_T": Q_T,
        "H_P1": H_P1,
        "H_P2": H_P2,
        "Q_P1": Q_P1,
        "Q_P2": Q_P2,
        "y_T": np.array([y_T_func(ti) for ti in t_arr]),
        "N_T": np.array([N_T_func(ti) for ti in t_arr]),
        "N_P": np.array([N_P_func(ti) for ti in t_arr]),
        # Debug outputs
        "Q_STANK": Q_STANK_native,  # Flow into STANK = Q_ELBOW - Q_L1
        "Q_L1_at_STANK": Q_L1_at_STANK,
        "Q_ELBOW_at_STANK": Q_ELBOW_at_STANK,
        # Penstock heads along path
        **penstock_heads,
        "H_STANK": H_STANK,
        "H_NODE1": H_NODE1,
        "H_NODE2": H_NODE2,
    }


def run_lumped_casadi_2state(
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    fpoints_N_P: FPoints,
    start_time: float,
    stop_time: float,
    step_size: float = 0.01,
) -> Dict[str, Any]:
    """Run fast 2-state CasADi lumped model with RK4 integration.

    This uses compiled CasADi expressions for maximum speed.
    The 2-state model enforces mass balance by construction (Q_loop = Q_P1 + Q_P2).
    """
    print("\nBuilding CasADi 2-state lumped model...")
    model = PF3LumpedCasadi(Path("data/pf3/passive_elements"))

    print("   Compiling RK4 simulator...")
    t_compile_start = time.time()
    simulator = PF3Simulator(model)
    print(f"   Compilation time: {time.time() - t_compile_start:.2f}s")

    # Input function from FPOINTS
    def u_func(t_val):
        return [fpoints_y_T(t_val), fpoints_N_T(t_val), fpoints_N_P(t_val)]

    # Get initial state from DAT files
    x0 = model.get_initial_state()
    print(f"   Initial state: Q_P1={x0[0]:.6f}, Q_P2={x0[1]:.6f}")

    print(
        f"   Running simulation t=[{start_time}, {stop_time}]s with dt={step_size*1000:.1f}ms..."
    )
    t_sim_start = time.time()
    result = simulator.simulate(
        x0, t_span=(start_time, stop_time), dt=step_size, u_func=u_func
    )
    t_sim_elapsed = time.time() - t_sim_start
    print(
        f"   Simulation time: {t_sim_elapsed:.2f}s ({stop_time/t_sim_elapsed:.1f}x realtime)"
    )

    t_arr = result["t"]
    Q_P1 = result["Q_1"]
    Q_P2 = result["Q_2"]
    Q_T = result["Q_T"]
    dQ_T = result["dQ_T"]

    # Compute H_T from Q_loop using turbine characteristic
    # Note: This is the quasi-steady head (no dynamic L*dQ/dt term since we don't track dQ/dt)
    # Use float() to convert CasADi DM (1x1) to Python scalar
    H_T = np.zeros_like(Q_T)
    for i, t_val in enumerate(t_arr):
        y_T_val = fpoints_y_T(t_val)
        N_T_val = fpoints_N_T(t_val)
        H_T[i] = float(model.fturb.compute_H(y_T_val, N_T_val, Q_T[i], dQ_T[i]))

    # Pump heads - use float() to convert CasADi DM (1x1) to Python scalar
    H_P1 = np.array(
        [
            float(model.pump1.compute_H(1.0, fpoints_N_P(t_arr[i]), -Q_P1[i], 0))
            for i in range(len(t_arr))
        ]
    )
    H_P2 = np.array(
        [
            float(model.pump2.compute_H(1.0, fpoints_N_P(t_arr[i]), -Q_P2[i], 0))
            for i in range(len(t_arr))
        ]
    )

    print(f"   CasADi 2-state simulation complete ({len(t_arr)} points)")

    return {
        "timestamps": t_arr,
        "H_T": H_T,
        "Q_T": Q_T,
        "H_P1": H_P1,
        "H_P2": H_P2,
        "Q_P1": Q_P1,
        "Q_P2": Q_P2,
        "y_T": np.array([fpoints_y_T(ti) for ti in t_arr]),
        "N_T": np.array([fpoints_N_T(ti) for ti in t_arr]),
        "N_P": np.array([fpoints_N_P(ti) for ti in t_arr]),
    }


def run_scipy_2state(
    fpoints_y_T: FPoints,
    fpoints_N_T: FPoints,
    fpoints_N_P: FPoints,
    start_time: float,
    stop_time: float,
    step_size: float = 0.01,
) -> Dict[str, Any]:
    """Run 2-state reduced model with SciPy solve_ivp.

    Uses PF3ReducedModel with states [Q_T, Q_d] and MNA-derived physics.
    Same physics as PF3LumpedCasadi.
    """
    print("\nBuilding SciPy 2-state reduced model...")
    model = build_pf3_reduced_model(data_dir=Path("data/pf3/passive_elements"))
    model.print_summary()

    # Get initial state from DAT files
    x0 = model.get_initial_state()
    print(f"   Initial state: Q_T={x0[0]:.6f}, Q_d={x0[1]:.6f}")

    # Create time evaluation points
    t_eval = np.arange(start_time, stop_time + step_size / 2, step_size)

    print(
        f"   Running simulation t=[{start_time}, {stop_time}]s with dt={step_size*1000:.1f}ms..."
    )
    t_sim_start = time.time()
    result = model.simulate(
        t_span=(start_time, stop_time),
        y0=x0,
        t_eval=t_eval,
        y_T_func=lambda t: fpoints_y_T(t),
        N_T_func=lambda t: fpoints_N_T(t),
        N_P_func=lambda t: fpoints_N_P(t),
        method="RK45",
    )
    t_sim_elapsed = time.time() - t_sim_start
    print(
        f"   Simulation time: {t_sim_elapsed:.2f}s ({stop_time/t_sim_elapsed:.1f}x realtime)"
    )

    t_arr = result.t
    Q_T = result.y[0, :]
    Q_d = result.y[1, :]

    # Recover individual pump flows
    Q_P1 = (Q_T + Q_d) / 2.0
    Q_P2 = (Q_T - Q_d) / 2.0

    # Compute H_T using turbine characteristic + dynamic term
    H_T = model.compute_H_T_from_timeseries(
        t_arr,
        Q_T,
        y_T_func=lambda t: fpoints_y_T(t),
        N_T_func=lambda t: fpoints_N_T(t),
    )

    # Pump heads
    H_P1 = np.array(
        [
            model.pump.compute_H(1.0, fpoints_N_P(t_arr[i]), -Q_P1[i])
            for i in range(len(t_arr))
        ]
    )
    H_P2 = np.array(
        [
            model.pump.compute_H(1.0, fpoints_N_P(t_arr[i]), -Q_P2[i])
            for i in range(len(t_arr))
        ]
    )

    print(f"   SciPy 2-state simulation complete ({len(t_arr)} points)")

    return {
        "timestamps": t_arr,
        "H_T": H_T,
        "Q_T": Q_T,
        "H_P1": H_P1,
        "H_P2": H_P2,
        "Q_P1": Q_P1,
        "Q_P2": Q_P2,
        "y_T": np.array([fpoints_y_T(ti) for ti in t_arr]),
        "N_T": np.array([fpoints_N_T(ti) for ti in t_arr]),
        "N_P": np.array([fpoints_N_P(ti) for ti in t_arr]),
    }


def plot_comparison(
    fmu_results: Dict,
    native_results: Dict | None = None,
    casadi_results: Dict | None = None,
    scipy_results: Dict | None = None,
    output_file: Path | None = None,
    show: bool = True,
    scenario_name: str | None = None,
):
    """Create comparison plot."""

    plt.close("all")

    # Create figure with 4 rows
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows), layout="constrained")
    title = "FMU vs Native vs ROM-CasADi vs ROM-Native"
    if scenario_name:
        title = f"[{scenario_name}] {title}"
    fig.suptitle(title)

    t_fmu = fmu_results["timestamps"]
    t_native = native_results["timestamps"] if native_results else None
    t_casadi = casadi_results["timestamps"] if casadi_results else None
    t_scipy = scipy_results["timestamps"] if scipy_results else None

    # Row 1: H_T and Q_T
    ax = axes[0, 0]
    ax.plot(
        t_fmu, fmu_results["outputs"]["FTURB1-H"], "b-", lw=1.5, label="FMU (reference)"
    )
    if native_results:
        ax.plot(t_native, native_results["H_T"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi,
            casadi_results["H_T"],
            "c-",
            lw=1,
            alpha=0.8,
            label="ROM-CasADi",
        )
    if scipy_results:
        ax.plot(
            t_scipy,
            scipy_results["H_T"],
            "g:",
            lw=1.5,
            alpha=0.8,
            label="ROM-Native",
        )
    ax.set_ylabel("H_T [m]")
    ax.set_title("Turbine Head")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_fmu, fmu_results["outputs"]["FTURB1-Q"], "b-", lw=1.5, label="FMU")
    if native_results:
        ax.plot(t_native, native_results["Q_T"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi, casadi_results["Q_T"], "c-", lw=1, alpha=0.8, label="ROM-CasADi"
        )
    if scipy_results:
        ax.plot(
            t_scipy, scipy_results["Q_T"], "g:", lw=1.5, alpha=0.8, label="ROM-Native"
        )
    ax.set_ylabel("Q_T [m³/s]")
    ax.set_title("Turbine Flow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Pump heads
    ax = axes[1, 0]
    ax.plot(t_fmu, fmu_results["outputs"]["FPUMP1-H"], "b-", lw=1.5, label="FMU")
    if native_results:
        ax.plot(t_native, native_results["H_P1"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi, casadi_results["H_P1"], "c-", lw=1, alpha=0.8, label="ROM-CasADi"
        )
    if scipy_results:
        ax.plot(
            t_scipy, scipy_results["H_P1"], "g:", lw=1.5, alpha=0.8, label="ROM-Native"
        )
    ax.set_ylabel("H_P1 [m]")
    ax.set_title("Pump 1 Head")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_fmu, fmu_results["outputs"]["FPUMP2-H"], "b-", lw=1.5, label="FMU")
    if native_results:
        ax.plot(t_native, native_results["H_P2"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi, casadi_results["H_P2"], "c-", lw=1, alpha=0.8, label="ROM-CasADi"
        )
    if scipy_results:
        ax.plot(
            t_scipy, scipy_results["H_P2"], "g:", lw=1.5, alpha=0.8, label="ROM-Native"
        )
    ax.set_ylabel("H_P2 [m]")
    ax.set_title("Pump 2 Head")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: Pump flows (native uses opposite sign convention, negate to match FMU)
    ax = axes[2, 0]
    ax.plot(t_fmu, fmu_results["outputs"]["FPUMP1-Q"], "b-", lw=1.5, label="FMU")
    if native_results:
        ax.plot(t_native, -native_results["Q_P1"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi, -casadi_results["Q_P1"], "c-", lw=1, alpha=0.8, label="ROM-CasADi"
        )
    if scipy_results:
        ax.plot(
            t_scipy, -scipy_results["Q_P1"], "g:", lw=1.5, alpha=0.8, label="ROM-Native"
        )
    ax.set_ylabel("Q_P1 [m³/s]")
    ax.set_title("Pump 1 Flow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_fmu, fmu_results["outputs"]["FPUMP2-Q"], "b-", lw=1.5, label="FMU")
    if native_results:
        ax.plot(t_native, -native_results["Q_P2"], "r--", lw=1.5, label="Native")
    if casadi_results:
        ax.plot(
            t_casadi, -casadi_results["Q_P2"], "c-", lw=1, alpha=0.8, label="ROM-CasADi"
        )
    if scipy_results:
        ax.plot(
            t_scipy, -scipy_results["Q_P2"], "g:", lw=1.5, alpha=0.8, label="ROM-Native"
        )
    ax.set_ylabel("Q_P2 [m³/s]")
    ax.set_title("Pump 2 Flow")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 4: H_T error and Q_T error
    ax = axes[3, 0]
    # Interpolate to FMU timestamps for error calculation
    if native_results:
        H_T_native_interp = np.interp(t_fmu, t_native, native_results["H_T"])
        H_T_err_native = (fmu_results["outputs"]["FTURB1-H"] - H_T_native_interp) * 1000
        ax.plot(t_fmu, H_T_err_native, "r-", lw=1.5, label="FMU - Native")

    if casadi_results:
        H_T_casadi_interp = np.interp(t_fmu, t_casadi, casadi_results["H_T"])
        H_T_err_casadi = (fmu_results["outputs"]["FTURB1-H"] - H_T_casadi_interp) * 1000
        ax.plot(t_fmu, H_T_err_casadi, "c-", lw=1, alpha=0.8, label="FMU - ROM-CasADi")

    if scipy_results:
        H_T_scipy_interp = np.interp(t_fmu, t_scipy, scipy_results["H_T"])
        H_T_err_scipy = (fmu_results["outputs"]["FTURB1-H"] - H_T_scipy_interp) * 1000
        ax.plot(t_fmu, H_T_err_scipy, "g:", lw=1.5, alpha=0.8, label="FMU - ROM-Native")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("H_T Error [mm]")
    ax.set_title("Turbine Head Error vs FMU")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    # Q_T error
    if native_results:
        Q_T_native_interp = np.interp(t_fmu, t_native, native_results["Q_T"])
        Q_T_err_native = (
            fmu_results["outputs"]["FTURB1-Q"] - Q_T_native_interp
        ) * 1000  # L/s
        ax.plot(t_fmu, Q_T_err_native, "r-", lw=1.5, label="FMU - Native")

    if casadi_results:
        Q_T_casadi_interp = np.interp(t_fmu, t_casadi, casadi_results["Q_T"])
        Q_T_err_casadi = (fmu_results["outputs"]["FTURB1-Q"] - Q_T_casadi_interp) * 1000
        ax.plot(t_fmu, Q_T_err_casadi, "c-", lw=1, alpha=0.8, label="FMU - ROM-CasADi")

    if scipy_results:
        Q_T_scipy_interp = np.interp(t_fmu, t_scipy, scipy_results["Q_T"])
        Q_T_err_scipy = (fmu_results["outputs"]["FTURB1-Q"] - Q_T_scipy_interp) * 1000
        ax.plot(t_fmu, Q_T_err_scipy, "g:", lw=1.5, alpha=0.8, label="FMU - ROM-Native")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Q_T Error [L/s]")
    ax.set_title("Turbine Flow Error vs FMU")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(
    fmu_results: Dict,
    native_results: Dict | None = None,
    casadi_results: Dict | None = None,
    scipy_results: Dict | None = None,
):
    """Print comparison summary."""

    t_fmu = fmu_results["timestamps"]
    H_T_fmu = fmu_results["outputs"]["FTURB1-H"]
    Q_T_fmu = fmu_results["outputs"]["FTURB1-Q"]

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: FMU vs Native vs ROM-CasADi vs ROM-Native")
    print("=" * 80)

    print("\n--- TURBINE HEAD H_T ---")
    print(
        f"{'Model':<20} {'RMS Error [mm]':>15} {'Max |Error| [mm]':>18} {'Final Error [mm]':>18}"
    )
    print("-" * 75)

    if native_results:
        H_T_native = np.interp(
            t_fmu, native_results["timestamps"], native_results["H_T"]
        )
        err = H_T_fmu - H_T_native
        print(
            f"{'Native':<20} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
        )

    if casadi_results:
        H_T_casadi = np.interp(
            t_fmu, casadi_results["timestamps"], casadi_results["H_T"]
        )
        err = H_T_fmu - H_T_casadi
        print(
            f"{'ROM-CasADi':<20} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
        )

    if scipy_results:
        H_T_scipy = np.interp(t_fmu, scipy_results["timestamps"], scipy_results["H_T"])
        err = H_T_fmu - H_T_scipy
        print(
            f"{'ROM-Native':<20} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
        )

    print("\n--- TURBINE FLOW Q_T ---")
    print(
        f"{'Model':<20} {'RMS Error [L/s]':>15} {'Max |Error| [L/s]':>18} {'Final Error [L/s]':>18}"
    )
    print("-" * 75)

    if native_results:
        Q_T_native = np.interp(
            t_fmu, native_results["timestamps"], native_results["Q_T"]
        )
        err = Q_T_fmu - Q_T_native
        print(
            f"{'Native':<20} {np.sqrt(np.mean(err**2))*1000:>15.3f} {np.max(np.abs(err))*1000:>18.3f} {err[-1]*1000:>18.3f}"
        )

    if casadi_results:
        Q_T_casadi = np.interp(
            t_fmu, casadi_results["timestamps"], casadi_results["Q_T"]
        )
        err = Q_T_fmu - Q_T_casadi
        print(
            f"{'ROM-CasADi':<20} {np.sqrt(np.mean(err**2))*1000:>15.3f} {np.max(np.abs(err))*1000:>18.3f} {err[-1]*1000:>18.3f}"
        )

    if scipy_results:
        Q_T_scipy = np.interp(t_fmu, scipy_results["timestamps"], scipy_results["Q_T"])
        err = Q_T_fmu - Q_T_scipy
        print(
            f"{'ROM-Native':<20} {np.sqrt(np.mean(err**2))*1000:>15.3f} {np.max(np.abs(err))*1000:>18.3f} {err[-1]*1000:>18.3f}"
        )

    # Debug section: compare internal states if available (only when native results exist)
    if native_results:
        fmu_outputs = fmu_results["outputs"]
        has_debug = (
            "STANK-Hc" in fmu_outputs
            or "L4-HNb" in fmu_outputs
            or "L12-H1" in fmu_outputs
        )

        if has_debug:
            print("\n--- DEBUG: INTERNAL STATES (FMU vs Native 140-state) ---")
            print(
                f"{'Variable':<25} {'RMS Error [mm]':>15} {'Max |Error| [mm]':>18} {'Final Error [mm]':>18}"
            )
            print("-" * 80)

            t_native = native_results["timestamps"]

            if "STANK-Hc" in fmu_outputs and "H_STANK" in native_results:
                H_STANK_native = np.interp(t_fmu, t_native, native_results["H_STANK"])
                err = fmu_outputs["STANK-Hc"] - H_STANK_native
                print(
                    f"{'STANK-Hc (surge tank)':<25} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
                )

            if "L4-HNb" in fmu_outputs and "H_NODE1" in native_results:
                H_NODE1_native = np.interp(t_fmu, t_native, native_results["H_NODE1"])
                err = fmu_outputs["L4-HNb"] - H_NODE1_native
                print(
                    f"{'L4-HNb (NODE1)':<25} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
                )

            if "L12-H1" in fmu_outputs and "H_NODE2" in native_results:
                H_NODE2_native = np.interp(t_fmu, t_native, native_results["H_NODE2"])
                err = fmu_outputs["L12-H1"] - H_NODE2_native
                print(
                    f"{'L12-H1 (NODE2)':<25} {np.sqrt(np.mean(err**2))*1000:>15.2f} {np.max(np.abs(err))*1000:>18.2f} {err[-1]*1000:>18.2f}"
                )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare differne simulation models")
    parser.add_argument(
        "--stop-time", type=float, default=10, help="Simulation stop time [s]"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--skip-native", action="store_true", help="Skip native 140-state model"
    )
    parser.add_argument(
        "--skip-casadi", action="store_true", help="Skip CasADi 2-state model"
    )
    parser.add_argument(
        "--skip-scipy", action="store_true", help="Skip SciPy 2-state model"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful when running in Docker on macOS)",
    )
    parser.add_argument(
        "--interp",
        choices=["linear", "clough_tocher", "structured"],
        default="linear",
        help="Suter interpolation method (default: linear)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="fast_transition",
        help=f"Scenario to run: {list(SCENARIOS.keys())} (default: fast_transition)",
    )
    parser.add_argument(
        "--regp",
        choices=list(REGP_OPTIONS.keys()),
        default="REGP",
        help=f"REGP file for pump speed: {list(REGP_OPTIONS.keys())} (default: REGP)",
    )
    args = parser.parse_args()

    # Get scenario configuration
    scenario = SCENARIOS[args.scenario]
    fmu_file = scenario["fmu"]
    fpoints_y_T_path = scenario["fpoints_y_T"]
    fpoints_N_T_path = scenario["fpoints_N_T"]
    # Use REGP option (overrides scenario default)
    fpoints_N_P_path = REGP_OPTIONS[args.regp]
    regp_name = args.regp

    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    start_time = 0.0
    stop_time = args.stop_time

    print("=" * 80)
    print(f"COMPARE 140-STATE NATIVE vs FMU ({args.scenario} scenario)")
    print("=" * 80)
    print(f"Scenario: {args.scenario} - {scenario['description']}")
    print(f"REGP file: {regp_name} ({fpoints_N_P_path})")
    print(f"Stop time: {stop_time}s")
    print(f"FMU: {fmu_file}")
    print(f"Interpolation method: {args.interp}")

    # Load FPOINTS
    print("\nLoading FPOINTS...")
    fpoints_y_T = FPoints(fpoints_y_T_path)
    fpoints_N_T = FPoints(fpoints_N_T_path)
    fpoints_N_P = FPoints(fpoints_N_P_path)

    print(
        f"  y_T(0)={fpoints_y_T(0):.4f}, y_T(5.2)={fpoints_y_T(5.2):.4f}, y_T(10)={fpoints_y_T(10):.4f}"
    )
    print(
        f"  N_T(0)={fpoints_N_T(0):.1f}, N_T(5.2)={fpoints_N_T(5.2):.1f}, N_T(10)={fpoints_N_T(10):.1f}"
    )
    print(
        f"  N_P(0)={fpoints_N_P(0):.1f}, N_P(5.2)={fpoints_N_P(5.2):.1f}, N_P(10)={fpoints_N_P(10):.1f}"
    )

    # Cache paths include scenario name and REGP to avoid cross-contamination
    fmu_cache = CACHE_DIR / f"fmu_{args.scenario}_{regp_name}_t{stop_time}.pkl"
    native_cache = (
        CACHE_DIR / f"native_{args.scenario}_{regp_name}_t{stop_time}_{args.interp}.pkl"
    )

    # Run FMU
    fmu_results = None
    if not args.no_cache and fmu_cache.exists():
        print(f"\nLoading FMU from cache: {fmu_cache}")
        with open(fmu_cache, "rb") as f:
            fmu_results = pickle.load(f)

    if fmu_results is None:
        print("\n" + "-" * 60)
        print("Running SIMSEN FMU...")
        print("-" * 60)
        fmu_results = run_fmu_simulation(
            fmu_file,
            fpoints_y_T,
            fpoints_N_T,
            fpoints_N_P,
            start_time,
            stop_time,
            step_size=0.01,
        )
        with open(fmu_cache, "wb") as f:
            pickle.dump(fmu_results, f)
        print(f"Cached to: {fmu_cache}")

    # Run Native 140-state
    native_results = None
    if args.skip_native:
        print("\n--skip-native specified, skipping native 140-state model.")
    elif not args.no_cache and native_cache.exists():
        print(f"\nLoading Native from cache: {native_cache}")
        with open(native_cache, "rb") as f:
            native_results = pickle.load(f)
    else:
        print("\n" + "-" * 60)
        print("Running Native 140-state model...")
        print("-" * 60)
        native_results = run_native_140state(
            fpoints_y_T,
            fpoints_N_T,
            fpoints_N_P,
            start_time,
            stop_time,
            chunk_duration=1.0,
            interp_method=args.interp,
        )
        with open(native_cache, "wb") as f:
            pickle.dump(native_results, f)
        print(f"Cached to: {native_cache}")

    # Run CasADi 2-state (fast RK4)
    casadi_results = None
    if not args.skip_casadi:
        print("\n" + "-" * 60)
        print("Running CasADi 2-state model (fast RK4)...")
        print("-" * 60)
        casadi_results = run_lumped_casadi_2state(
            fpoints_y_T,
            fpoints_N_T,
            fpoints_N_P,
            start_time,
            stop_time,
            step_size=0.01,
        )

    # Run SciPy 2-state (RK45)
    scipy_results = None
    if not args.skip_scipy:
        print("\n" + "-" * 60)
        print("Running SciPy 2-state model (RK45)...")
        print("-" * 60)
        scipy_results = run_scipy_2state(
            fpoints_y_T,
            fpoints_N_T,
            fpoints_N_P,
            start_time,
            stop_time,
            step_size=0.01,
        )

    # Summary
    print_summary(fmu_results, native_results, casadi_results, scipy_results)

    # Plot (skip if --no-plot specified)
    if not args.no_plot:
        main_suffix = f"_{args.scenario}" if args.scenario else ""
        output_file = OUTPUT_DIR / f"compare_models_vs_fmu{main_suffix}.png"
        plot_comparison(
            fmu_results,
            native_results,
            casadi_results,
            scipy_results,
            output_file,
            show=True,
            scenario_name=args.scenario,
        )
    else:
        print("\nSkipping plot (--no-plot specified)")

    print("\nDone!")


if __name__ == "__main__":
    main()
