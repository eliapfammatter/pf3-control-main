"""
PF3 Hydraulic Network - Distributed model using incidence matrix formulation.

Factory function that builds a HydraulicNetwork with individual pipe segments
(~140 states), preserving water hammer dynamics.

TOPOLOGY (following SIMSEN dot convention - dot = inlet):
=========================================================
The circuit has TWO NODES (NODE1, NODE2) with THREE PARALLEL PATHS.

PENSTOCK (STANK → NODE1):
    STANK → L1 → ELBOW1 → L2 → ELBOW2 → L3 → ELBOW3 → L4 → NODE1

TURBINE + DRAFT TUBE (STANK → NODE2):
    STANK → ELBOW → CONE → FTURB1 → L19_2 → L19_1 → L18 → ELBOW15 →
    L17 → ELBOW14 → L16 → L15 → L14 → ELBOW13 → L13 → ELBOW12 → L12 → NODE2

PUMP 1 PATH (NODE1 → NODE2, short path):
    NODE1 → ELBOW4 → L5 → ELBOW5 → LP1 → PUMP1 → NODE2

PUMP 2 PATH (NODE1 → NODE2, long path):
    NODE1 → ELBOW7 → L6 → ELBOW8 → L7 → ELBOW9 → LP2 → PUMP2 →
    L10 → ELBOW10 → L11 → ELBOW11 → NODE2

Usage:
    from src.pf3_full_network import build_pf3_full

    network = build_pf3_full(interp_method="structured")
    y0 = network.get_initial_state()
    result = network.simulate(t_span=(0, 10), y0=y0, external_inputs_func=...)
"""

from pathlib import Path

from .hydraulic_elements import DiscreteLoss, Pipe, PumpTurbine, Tank
from .incidence_network import ElementNetworkBuilder, HydraulicNetwork


def build_pf3_full(
    interp_method: str = "linear",
    data_dir: Path = Path("data/pf3/passive_elements"),
) -> HydraulicNetwork:
    """
    Build ~140-state distributed PF3 network.

    Uses individual pipe segments with internal head nodes, preserving
    water hammer dynamics (~1-2 Hz oscillations).

    Parameters
    ----------
    interp_method : str
        Interpolation method for characteristic curves ("linear", "structured", etc.)
    data_dir : Path
        Directory containing SIMSEN DAT files.

    Returns
    -------
    HydraulicNetwork
        Built network ready for simulation.
    """
    data_dir = Path(data_dir)

    def load_pipe(name):
        return Pipe.from_dat(data_dir / f"{name}.DAT")

    def load_dloss(name):
        return DiscreteLoss.from_dat(data_dir / f"{name}.DAT")

    # =========================================================================
    # Load elements
    # =========================================================================

    # Penstock pipes
    L1, L2, L3, L4 = load_pipe("L1"), load_pipe("L2"), load_pipe("L3"), load_pipe("L4")

    # Turbine path pipes
    L12, L13, L14 = load_pipe("L12"), load_pipe("L13"), load_pipe("L14")
    L15, L16, L17 = load_pipe("L15"), load_pipe("L16"), load_pipe("L17")
    L18, L19_1, L19_2 = load_pipe("L18"), load_pipe("L19_1"), load_pipe("L19_2")
    CONE, ELBOW_pipe = load_pipe("CONE"), load_pipe("ELBOW")

    # Pump 1 path
    L5, LP1 = load_pipe("L5"), load_pipe("LP1")

    # Pump 2 path
    L6, L7, LP2, L10, L11 = (
        load_pipe("L6"),
        load_pipe("L7"),
        load_pipe("LP2"),
        load_pipe("L10"),
        load_pipe("L11"),
    )

    # Discrete losses
    E1, E2, E3 = load_dloss("ELBOW1"), load_dloss("ELBOW2"), load_dloss("ELBOW3")
    E4, E5 = load_dloss("ELBOW4"), load_dloss("ELBOW5")
    E7, E8, E9 = load_dloss("ELBOW7"), load_dloss("ELBOW8"), load_dloss("ELBOW9")
    E10, E11 = load_dloss("ELBOW10"), load_dloss("ELBOW11")
    E12, E13, E14, E15 = (
        load_dloss("ELBOW12"),
        load_dloss("ELBOW13"),
        load_dloss("ELBOW14"),
        load_dloss("ELBOW15"),
    )

    # Tank
    tank_path = data_dir / "STANK.DAT"
    tank = (
        Tank.from_dat(tank_path)
        if tank_path.exists()
        else Tank(name="STANK", A=10.0, H0=0.0)
    )

    # Machines - load with interp_method
    fturb1 = PumpTurbine.from_dat(data_dir / "FTURB1.DAT", interp_method=interp_method)
    fturb1._y_key = "y_T"
    fturb1._N_key = "N_T"

    pump1 = PumpTurbine.from_dat(data_dir / "PUMP1.DAT", interp_method=interp_method)
    pump1._y_key = "y_P1"
    pump1._N_key = "N_P"

    pump2 = PumpTurbine.from_dat(data_dir / "PUMP2.DAT", interp_method=interp_method)
    pump2._y_key = "y_P2"
    pump2._N_key = "N_P"

    # =========================================================================
    # Build network topology
    # =========================================================================

    builder = ElementNetworkBuilder()
    builder.add(tank)

    # --- PENSTOCK: STANK → NODE1 ---
    builder.add(L1, from_node="STANK", to_node="j_L1_E1")
    builder.add(E1, from_node="j_L1_E1", to_node="j_E1_L2")
    builder.add(L2, from_node="j_E1_L2", to_node="j_L2_E2")
    builder.add(E2, from_node="j_L2_E2", to_node="j_E2_L3")
    builder.add(L3, from_node="j_E2_L3", to_node="j_L3_E3")
    builder.add(E3, from_node="j_L3_E3", to_node="j_E3_L4")
    builder.add(L4, from_node="j_E3_L4", to_node="NODE1")

    # --- TURBINE: CONE → ELBOW → STANK ---
    builder.add(CONE, from_node="j_CONE_TURB", to_node="j_ELBOW_CONE")
    builder.add(ELBOW_pipe, from_node="j_ELBOW_CONE", to_node="STANK")

    # --- FRANCIS TURBINE (separate branch, Lequ > 0) ---
    builder.add(fturb1, from_node="j_TURB", to_node="j_CONE_TURB", negate_Q0=True)

    # --- DRAFT TUBE: NODE2 → L12 → ... → L19_2 → TURB ---
    builder.add(L19_2, from_node="j_L19_2_L19_1", to_node="j_TURB")
    builder.add(L19_1, from_node="j_L19_1_L18", to_node="j_L19_2_L19_1")
    builder.add(L18, from_node="j_E15_L18", to_node="j_L19_1_L18")
    builder.add(E15, from_node="j_L17_E15", to_node="j_E15_L18")
    builder.add(L17, from_node="j_E14_L17", to_node="j_L17_E15")
    builder.add(E14, from_node="j_L16_E14", to_node="j_E14_L17")
    builder.add(L16, from_node="j_L16_L15", to_node="j_L16_E14")
    builder.add(L15, from_node="j_L15_L14", to_node="j_L16_L15")
    builder.add(L14, from_node="j_E13_L14", to_node="j_L15_L14")
    builder.add(E13, from_node="j_L13_E13", to_node="j_E13_L14")
    builder.add(L13, from_node="j_E12_L13", to_node="j_L13_E13")
    builder.add(E12, from_node="j_L12_E12", to_node="j_E12_L13")
    builder.add(L12, from_node="NODE2", to_node="j_L12_E12")

    # --- PUMP 1 PATH: NODE1 → NODE2 (short) ---
    builder.add(E4, from_node="NODE1", to_node="j_E4_L5")
    builder.add(L5, from_node="j_E4_L5", to_node="j_L5_E5")
    builder.add(E5, from_node="j_L5_E5", to_node="j_E5_LP1")
    builder.add(LP1, from_node="j_E5_LP1", to_node="NODE2")

    # --- PUMP 2 PATH: NODE1 → NODE2 (long) ---
    builder.add(E7, from_node="NODE1", to_node="j_E7_L6")
    builder.add(L6, from_node="j_E7_L6", to_node="j_L6_E8")
    builder.add(E8, from_node="j_L6_E8", to_node="j_E8_L7")
    builder.add(L7, from_node="j_E8_L7", to_node="j_L7_E9")
    builder.add(E9, from_node="j_L7_E9", to_node="j_E9_LP2")
    builder.add(LP2, from_node="j_E9_LP2", to_node="j_LP2_L10")
    builder.add(L10, from_node="j_LP2_L10", to_node="j_L10_E10")
    builder.add(E10, from_node="j_L10_E10", to_node="j_E10_L11")
    builder.add(L11, from_node="j_E10_L11", to_node="j_L11_E11")
    builder.add(E11, from_node="j_L11_E11", to_node="NODE2")

    # --- PUMPS (H_sources on pipes, Lequ=0) ---
    builder.add(pump1, on_element="LP1", negate_Q0=True)
    builder.add(pump2, on_element="LP2", negate_Q0=True)

    return builder.build()
